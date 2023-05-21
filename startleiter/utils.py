import metpy.calc as mpcalc
from metpy.units import units

import os

import numpy as np
import pandas as pd
from sqlalchemy import create_engine


# silence invalid value warning
np.seterr(invalid="ignore")


def get_engine():
    db_url = os.environ.get("DATABASE_URL")
    db_url = db_url.replace("postgres://", "postgresql://")
    return create_engine(db_url, echo=False)


def read_flights(engine, target_id):
    cols = [
        "datetime",
        "site_id",
        "length_km",
        "max_altitude_m",
        "airtime",
        "glider_cat",
    ]
    df = pd.read_sql("flight", engine, index_col="id", columns=cols)
    return df[df.site_id == target_id].drop("site_id", axis=1)


def read_predictions(engine, target_id, date=None):
    df = pd.read_sql("prediction", engine)
    df = df[df.site_id == target_id].drop("site_id", axis=1)
    if date:
        timestamp = pd.to_datetime(date).replace(hour=0, minute=0, second=0)
        date_id = df.reftime == timestamp
        df = df[date_id].drop("reftime", axis=1)
    return df


def prepare_flights(df_flight):
    # minimum length: 2 km
    df_flight = df_flight[df_flight.length_km > 2]

    # maximum altitude: 5000 m
    df_flight = df_flight[
        np.logical_or(df_flight.max_altitude_m < 5000, df_flight.max_altitude_m.isna())
    ]

    # minimum altitude: 1000 m
    df_flight = df_flight[
        np.logical_or(df_flight.max_altitude_m > 1000, df_flight.max_altitude_m.isna())
    ]

    # exclude rigid wings
    df_flight = df_flight.drop(df_flight[df_flight.glider_cat == "HGFAI-1 HG"].index)
    df_flight = df_flight.drop(df_flight[df_flight.glider_cat == "RW5FAI-5 RW"].index)

    # datetime to date
    df_flight["date"] = df_flight.datetime.dt.date

    # convert airtime to float
    df_flight["airtime_hours"] = df_flight["airtime"] / np.timedelta64(1, "s") / 3600

    # no. flights in last 24 hours
    df_flight = df_flight.reset_index().sort_values("datetime").set_index("datetime")
    df_flight["occurrences_last_24h"] = 1
    df_flight["occurrences_last_24h"] = (
        df_flight["occurrences_last_24h"].rolling("24H").sum()
    )
    df_flight = df_flight.reset_index().sort_values("id").set_index("id")

    # compute the day of the week
    df_flight["dayofweek"] = df_flight.datetime.dt.dayofweek
    mapday = {0: "Mon", 1: "Tue", 2: "Wed", 3: "Thu", 4: "Fri", 5: "Sat", 6: "Sun"}
    df_flight["dayofweek"] = df_flight["dayofweek"].map(mapday)

    # compute the month
    df_flight["month"] = df_flight.datetime.dt.month
    mapmonths = {
        1: "Jan",
        2: "Feb",
        3: "Mar",
        4: "Apr",
        5: "May",
        6: "Jun",
        7: "Jul",
        8: "Aug",
        9: "Sep",
        10: "Oct",
        11: "Nov",
        12: "Dec",
    }
    df_flight["month"] = df_flight["month"].map(mapmonths)

    # compute the season
    df_flight["season"] = df_flight.datetime.dt.month % 12 // 3 + 1
    mapseas = {1: "DJF", 2: "MAM", 3: "JJA", 4: "SON"}
    df_flight["season"] = df_flight["season"].map(mapseas)

    return df_flight


def get_flights(target_id, engine):
    df = read_flights(target_id, engine)
    return prepare_flights(df)


def get_predictions(target_id, engine):
    df = read_predictions(target_id, engine)
    return df


def to_wind_components(dataset, inverse=False):
    """
    Convert wind direction and speed to (and from) wind components (u and v).

    Parameters
    ----------
    dataset: xarray.Dataset
    inverse: bool, optional

    Returns
    -------
    xarray.Dataset

    """
    dataset = dataset.copy()
    if not inverse:
        wind_components = mpcalc.wind_components(
            dataset["SKNT"].values * units.knots, dataset["DRCT"].values * units.deg
        )
        dataset["U"] = (dataset.coords, wind_components[0].magnitude)
        dataset["V"] = (dataset.coords, wind_components[1].magnitude)
        dataset["U"].attrs["units"] = "knot"
        dataset["U"] = dataset["U"].astype("float32")
        dataset["V"].attrs["units"] = "knot"
        dataset["V"] = dataset["V"].astype("float32")
        dataset = dataset.drop_vars(("SKNT", "DRCT"))

    else:
        dataset["SKNT"] = (
            dataset.coords,
            mpcalc.wind_speed(
                dataset["U"].values * units.knots, dataset["V"].values * units.knots
            ).magnitude,
        )
        dataset["DRCT"] = (
            dataset.coords,
            mpcalc.wind_direction(
                dataset["U"].values * units.knots, dataset["V"].values * units.knots
            ).magnitude,
        )
        dataset["SKNT"].attrs["units"] = "knot"
        dataset["SKNT"] = dataset["SKNT"].astype("float32")
        dataset["DRCT"].attrs["units"] = "deg"
        dataset["DRCT"] = dataset["DRCT"].astype("float32")
        dataset = dataset.drop_vars(("U", "V"))

    return dataset
