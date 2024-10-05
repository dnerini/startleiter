import logging
import re

import requests
import numpy as np
import pandas as pd
import xarray as xr
import metpy.calc as mpcalc
from metpy.units import units

import startleiter.scraping as scr
from startleiter import config as CFG


_LOGGER = logging.getLogger(__name__)

BASE_URL = "https://api.open-meteo.com"
SEARCH_URL = BASE_URL + "/v1/dwd-icon"
DEFAULT_QUERY = {
    "latitude": 47.45,
    "longitude": 8.58,
    "hourly": "pressure_msl",
}

# https://api.open-meteo.com/v1/dwd-icon?latitude=47.45&longitude=8.58&hourly=pressure_msl


def scrape(station_name, hourly_parameter):
    """
    Parameters
    ----------
    station: str
        The station shortname or its identifier
    hourly_parameter: str, e.g. "pressure_msl"
        See https://open-meteo.com/en/docs/dwd-api

    Returns
    -------

    """
    lat = CFG["stations"][station_name]["latitude"]
    long = CFG["stations"][station_name]["longitude"]
    this_query = {
        "latitude": lat,
        "longitude": long,
        "hourly": hourly_parameter,
    }
    query_url = scr.build_query(SEARCH_URL, DEFAULT_QUERY, this_query)
    _LOGGER.info(query_url)
    resp = requests.get(query_url)
    df = pd.DataFrame(resp.json()["hourly"])
    df["time"] = pd.to_datetime(df["time"])
    df = df.set_index("time")
    return df.astype("float32")


def sounding_parse_df(df):
    pressure_vars = [col for col in df.columns if re.search(r"\d+hPa", col)]
    df_long = df.reset_index().melt(
        id_vars=["time"],
        value_vars=pressure_vars,
        var_name="variable",
        value_name="value",
    )
    df_long["pressure"] = df_long["variable"].apply(
        lambda x: int(re.search(r"(\d+)hPa", x).group(1))
    )
    df_long["variable"] = df_long["variable"].apply(lambda x: x.rsplit("_", 1)[0])
    df_pivoted = df_long.pivot_table(
        index=["time", "pressure"], columns="variable", values="value"
    ).reset_index()
    return df_pivoted.set_index(["time", "pressure"]).to_xarray()


def sounding_convert_units(ds):
    relhum = xr.where(ds["relative_humidity"] > 1, ds.relative_humidity, 1).values
    dewpoint = mpcalc.dewpoint_from_relative_humidity(
        ds["temperature"].values * units.degC, relhum * units.percent
    )
    ds["DWPT"] = (ds.coords, dewpoint.magnitude)
    ds["DWPT"].attrs["units"] = "degC"
    wspeed = ds["wind_speed"].values * units.kilometer / units.hour
    ds["SKNT"] = (ds.coords, wspeed.to(units.knots).magnitude)
    ds = ds.assign_coords(leadtime=("time", (ds.time - ds.time.isel(time=0)).data))
    ds = ds.swap_dims({"time": "leadtime"})
    ds = ds.rename(
        {
            "wind_direction": "DRCT",
            "pressure": "PRES",
            "temperature": "TEMP",
            "time": "validtime",
        }
    )
    ds = ds.drop_vars(("wind_speed", "relative_humidity"))
    return ds


def scrape_sounding(lat, lon, leadtime):
    """
    Parameters
    ----------
    station: str
        The station shortname or its identifier

    Returns
    -------

    """
    this_query = {
        "latitude": lat,
        "longitude": lon,
        "hourly": "temperature_2m,relative_humidity_2m,temperature_1000hPa,temperature_975hPa,temperature_950hPa,temperature_925hPa,temperature_900hPa,temperature_850hPa,temperature_800hPa,temperature_700hPa,temperature_600hPa,temperature_500hPa,temperature_400hPa,temperature_300hPa,temperature_250hPa,temperature_200hPa,temperature_150hPa,temperature_100hPa,temperature_70hPa,temperature_50hPa,temperature_30hPa,relative_humidity_1000hPa,relative_humidity_975hPa,relative_humidity_950hPa,relative_humidity_925hPa,relative_humidity_900hPa,relative_humidity_850hPa,relative_humidity_800hPa,relative_humidity_700hPa,relative_humidity_600hPa,relative_humidity_500hPa,relative_humidity_400hPa,relative_humidity_300hPa,relative_humidity_250hPa,relative_humidity_200hPa,relative_humidity_150hPa,relative_humidity_100hPa,relative_humidity_70hPa,relative_humidity_50hPa,wind_speed_1000hPa,wind_speed_975hPa,wind_speed_950hPa,wind_speed_925hPa,wind_speed_900hPa,wind_speed_850hPa,wind_speed_800hPa,wind_speed_700hPa,wind_speed_600hPa,wind_speed_500hPa,wind_speed_400hPa,wind_speed_300hPa,wind_speed_250hPa,wind_speed_200hPa,wind_speed_150hPa,wind_speed_100hPa,wind_speed_70hPa,wind_speed_50hPa,wind_speed_30hPa,wind_direction_1000hPa,wind_direction_975hPa,wind_direction_950hPa,wind_direction_925hPa,wind_direction_900hPa,wind_direction_850hPa,wind_direction_800hPa,wind_direction_700hPa,wind_direction_600hPa,wind_direction_500hPa,wind_direction_400hPa,wind_direction_300hPa,wind_direction_250hPa,wind_direction_200hPa,wind_direction_150hPa,wind_direction_100hPa,wind_direction_70hPa,wind_direction_50hPa,wind_direction_30hPa",
    }
    query_url = scr.build_query(SEARCH_URL, DEFAULT_QUERY, this_query)
    _LOGGER.debug(query_url)
    resp = requests.get(query_url)
    df = pd.DataFrame(resp.json()["hourly"])
    df["time"] = pd.to_datetime(df["time"])
    df = df.set_index("time")
    df = df.astype("float32")
    ds = sounding_parse_df(df)
    ds = sounding_convert_units(ds)
    ds = ds.sel(leadtime=leadtime)
    ref_pres = np.logspace(np.log10(200), 3, 64, base=10)[::-1] // 1
    ds = ds.interp(PRES=ref_pres)
    validtime = pd.to_datetime(ds.validtime.values)
    ds = ds.drop_vars(("leadtime", "validtime"), errors="ignore")
    return validtime, ds
