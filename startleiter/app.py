import logging
import pickle
from datetime import datetime, timedelta
from functools import lru_cache
from io import BytesIO
from typing import Literal, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
import xarray as xr
from fastapi import FastAPI
from starlette.responses import StreamingResponse, RedirectResponse

from startleiter import config as CFG
from startleiter import openmeteo, uwyo, rucsoundings
from startleiter.decorators import try_wait
from startleiter.explainer import compute_shap
from startleiter.plots import explainable_plot
from startleiter.utils import to_wind_components

LOGGER = logging.getLogger(__name__)
app = FastAPI()


AVAILABLE_SITES = Literal[
    "Cimetta",
    "Carì",
    "Monte Tamaro",
    "Monte Generoso",
    "Mornera",
    "Monte Lema",
    "Santa Maria",
]

# note: last bin is open, ie > 1600 m and > 150 km
ALT_BINS = [5, 400, 800, 1200, 1600, 2000, 2400]
DIST_BINS = [10, 25, 50, 75, 100, 125, 150, 200]

PRESSURE_MIN_hPa = 400
FILL_NA_VALUE = -5

STATIONS = CFG["stations"]

SITES = CFG["sites"]
SITE_IDS = {
    "Cimetta": 1,
    "Carì": 2,
    "Monte Tamaro": 3,
    "Monte Generoso": 4,
    "Mornera": 5,
    "Monte Lema": 6,
    "Santa Maria": 7,
}

MODEL_FLYABILITY = tf.keras.models.load_model("models/flyability.h5")
MODEL_MAX_ALT = tf.keras.models.load_model("models/fly_max_alt.h5")
MODEL_MAX_DIST = tf.keras.models.load_model("models/fly_max_dist.h5")

FLYABILITY_CALIBRATION_CURVE = pickle.load(
    open("models/flyability_calibration_curve.pkl", "rb")
)

MOMENTS_FLYABILITY = xr.load_dataset("models/flyability_moments.nc")
MOMENTS_MAX_ALT = xr.load_dataset("models/fly_max_alt_moments.nc")
MOMENTS_MAX_DIST = xr.load_dataset("models/fly_max_dist_moments.nc")

BACKGROUND = np.load("models/flyability_background.npy")

FLY_PROB_THR = 0.1


@app.get("/", include_in_schema=False)
async def basic_view():
    return RedirectResponse("/docs")


def parse_time(time: str, leadtime_days) -> tuple[datetime, int, datetime]:
    if time == "latest":
        time = datetime.utcnow()
    elif time == "yesterday":
        time = datetime.utcnow()
        time -= timedelta(days=1)
        leadtime_days = None
    elif time == "today":
        time = datetime.utcnow()
        leadtime_days = None
    elif time == "tomorrow":
        time = datetime.utcnow()
        leadtime_days = 1
    else:
        time = pd.to_datetime(time)
    leadtime_days = leadtime_days or 0
    if time.date() > datetime.utcnow().date():
        raise ValueError("Argument 'time' cannot be in the future!")
    if leadtime_days < 0:
        raise ValueError("Argument 'leadtime_days' cannot be negative!")
    if leadtime_days > 5:
        raise ValueError("Cannot predict more than 5 days ahead!")
    validtime = time + timedelta(days=leadtime_days)
    return time, leadtime_days, validtime


@try_wait(maxattempts=4)
def get_last_sounding(station, time):
    data = list(uwyo.scrape(station, time).items())[0]
    return data[0], data[1]["data"]


@try_wait(maxattempts=4)
def get_last_sounding_forecast(station, time, leadtime_hrs):
    leadtime = timedelta(hours=leadtime_hrs)
    data = list(rucsoundings.scrape(station, time, leadtime).items())[0]
    return data[0], data[1]["data"]


def get_last_pressure_diff_forecast(time: datetime, leadtime_days: int):
    qff_klo = openmeteo.scrape("Kloten", "pressure_msl")
    qff_lug = openmeteo.scrape("Lugano", "pressure_msl")
    qff_gve = openmeteo.scrape("Geneva", "pressure_msl")
    qff_diff_gve = qff_klo - qff_gve
    qff_diff_gve = qff_diff_gve.rename(columns={"pressure_msl": "KLO-GVE"})
    qff_diff_lug = qff_klo - qff_lug
    qff_diff_lug = qff_diff_lug.rename(columns={"pressure_msl": "KLO-LUG"})
    time = time.replace(minute=0, second=0, microsecond=0)
    time_idx = pd.date_range(
        time + pd.Timedelta(hours=1),
        time + pd.Timedelta(days=leadtime_days),
        freq="1H",
    )
    qff_diff_gve = qff_diff_gve.loc[time_idx]
    qff_diff_lug = qff_diff_lug.loc[time_idx]
    qff_diff = pd.concat((qff_diff_gve, qff_diff_lug), axis=1)
    assert qff_diff.shape[0] == leadtime_days * 24
    return qff_diff.to_xarray().rename({"index": "date"})


def extract_features(ds):
    ds = to_wind_components(ds)
    # dew point temperature depression
    ds["DWPD"] = ds["TEMP"] - ds["DWPT"]
    ds["WOY"] = ds.attrs["validtime"].isocalendar().week
    (ds,) = xr.broadcast(ds)
    return ds[["TEMP", "DWPD", "U", "V", "WOY"]].rename({"PRES": "level"})


def standardize(da, moments, inverse=False):
    """Standardize the input data with training mean and standard deviation."""
    if not inverse:
        return (da - moments.mu) / moments.sigma
    else:
        return da * moments.sigma + moments.mu


def get_sounding(station: str, time: datetime, leadtime_days: int) -> xr.Dataset:
    """Get the input data"""
    time = time.replace(hour=0, minute=0, second=0, microsecond=0)
    LOGGER.info(f"Time: {time}")
    station = STATIONS[station]
    if leadtime_days > 0:
        validtime, sounding = get_last_sounding_forecast(
            station["name"], time, leadtime_hrs=int(leadtime_days) * 24
        )
        sounding.attrs["source"] = f"GFS sounding +{leadtime_days * 24:.0f} h"
    else:
        validtime, sounding = get_last_sounding(station["stid"], time)
        sounding.attrs["source"] = f"Radiosounding 00Z {station['long_name']}"
    sounding.attrs["validtime"] = validtime
    sounding = extract_features(sounding)
    sounding = sounding.sel(level=slice(1000, PRESSURE_MIN_hPa))
    return sounding


def get_surface(time: datetime, leadtime_days: int) -> xr.Dataset:
    time = time.replace(hour=0, minute=0, second=0, microsecond=0)
    leadtime_days = leadtime_days or 0
    time += pd.Timedelta(days=leadtime_days)
    surface = get_last_pressure_diff_forecast(time, leadtime_days=1)
    return surface[["KLO-GVE", "KLO-LUG"]]


def get_features(time, leadtime_days):
    features = get_sounding("Cameri", time, leadtime_days)
    surface = get_surface(time, leadtime_days)
    surface = surface.pad(
        date=(0, features.sizes["level"] - surface.sizes["date"]),
        constant_values=np.nan,
    )
    surface = surface.rename({"date": "level"}).drop("level")
    features = features.merge(surface)
    return features.to_array().transpose("level", "variable").astype("float32")


def preprocess(features, site, moments):
    """Preprocess inputs"""
    inputs_features = standardize(features, moments)
    inputs_features = inputs_features.fillna(FILL_NA_VALUE)
    inputs_embedding = xr.DataArray(
        np.ones((1, 1)) * SITE_IDS[site],
        dims=("validtime", "variable"),
        coords={"variable": ["ID"]},
    )
    return xr.concat((inputs_features, inputs_embedding), "variable")


@lru_cache(maxsize=42)
def predict(site: str, time: datetime, leadtime_days: int):
    """Predict flyability, max altitude and max distance."""

    features = get_features(time, leadtime_days)

    # flyability
    inputs = preprocess(features, site, MOMENTS_FLYABILITY)
    fly_prob = float(MODEL_FLYABILITY.predict(inputs.values[None, ..., 0])[0][0])
    fly_prob = float(FLYABILITY_CALIBRATION_CURVE.predict([fly_prob]))

    # max altitude and distance
    if fly_prob < FLY_PROB_THR:
        max_alt_gain = 0
        max_dist = 0
    else:
        inputs = preprocess(features, site, MOMENTS_MAX_ALT)
        max_alt_gain = ALT_BINS[
            int(MODEL_MAX_ALT.predict(inputs.values[None, ..., 0])[0].argmax())
        ]
        inputs = preprocess(features, site, MOMENTS_MAX_DIST)
        max_dist = DIST_BINS[
            int(MODEL_MAX_DIST.predict(inputs.values[None, ..., 0])[0].argmax())
        ]

    max_alt = (max_alt_gain + SITES[site]["elevation"]) // 100 * 100

    return fly_prob, max_alt, max_dist


@lru_cache(maxsize=21)
def explain(site: str, time: datetime, leadtime_days: int):
    fly_prob, max_alt, max_dist = predict(site, time, leadtime_days)
    features = get_features(time, leadtime_days)
    inputs = preprocess(features, site, MOMENTS_FLYABILITY)
    shap_values = compute_shap(
        BACKGROUND, MODEL_FLYABILITY, inputs.values[None, ..., 0]
    )[0]
    fig = explainable_plot(
        SITES[site],
        features,
        shap_values,
        fly_prob,
        max_alt,
        max_dist,
        min_pressure_hPa=PRESSURE_MIN_hPa,
    )
    return fig


@app.get("/site")
@app.get("/cimetta")  # deprecated
async def predict_site(
    site: AVAILABLE_SITES = "Cimetta",
    time: str = "latest",
    leadtime_days: Optional[int] = None,
):
    time, leadtime_days, validtime = parse_time(time, leadtime_days)
    fly_prob, max_alt, max_dist = predict(site, time, leadtime_days)

    return {
        "site": site,
        "validtime": f"{validtime:%Y-%m-%d}",
        "flying_probability": fly_prob,
        "max_altitude_masl": max_alt,
        "max_distance_km": max_dist,
    }


@app.get("/site_plot")
@app.get("/cimetta_plot")  # deprecated
async def plot_site(
    site: AVAILABLE_SITES = "Cimetta",
    time: str = "latest",
    leadtime_days: Optional[int] = None,
):
    time, leadtime_days, _ = parse_time(time, leadtime_days)

    fig = explain(site, time, leadtime_days)

    image_file = BytesIO()
    plt.savefig(image_file)
    plt.close(fig)
    image_file.seek(0)
    return StreamingResponse(image_file, media_type="image/png")
