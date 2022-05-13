import logging
from datetime import datetime, timedelta
from functools import lru_cache
from io import BytesIO
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
import xarray as xr
from fastapi import FastAPI
from starlette.responses import StreamingResponse, RedirectResponse

from startleiter import config as CFG
from startleiter import uwyo, rucsoundings
from startleiter.decorators import try_wait
from startleiter.explainer import compute_shap, explainable_plot
from startleiter.utils import to_wind_components

LOGGER = logging.getLogger(__name__)
app = FastAPI()


# TODO: do not hardcode
STATIONS = CFG["stations"]
CIMETTA_ELEVATION = 1600
ALT_BIN = 300
DIST_BIN = 15


@lru_cache
@try_wait()
def get_last_sounding(station, time):
    data = list(uwyo.scrape(station, time).items())[0]
    return data[0], data[1]["data"]


@lru_cache
def get_last_sounding_forecast(station, time, leadtime_hrs):
    leadtime = timedelta(hours=leadtime_hrs)
    data = list(rucsoundings.scrape(station, time, leadtime).items())[0]
    return data[0], data[1]["data"]


def preprocess(ds):
    ds = to_wind_components(ds)
    # dew point temperature depression
    ds["DWPD"] = ds["TEMP"] - ds["DWPT"]
    return (
        ds[["TEMP", "DWPD", "U", "V"]]
        .rename({"PRES": "level"})
        .bfill(dim="level", limit=3)
        .to_array()
        .transpose("level", "variable")
        .astype("float32")
    )


def standardize(da, moments, inverse=False):
    """Standardize the input data with training mean and standard deviation."""
    if not inverse:
        return (da - moments.mu) / moments.sigma
    else:
        return da * moments.sigma + moments.mu


def pipeline(station: str, time: str, leadtime_days: int) -> xr.Dataset:
    """Get and preprocess the input data"""
    if time == "latest":
        time = datetime.utcnow()
    else:
        time = pd.to_datetime(time)
    time = time.replace(hour=0, minute=0, second=0, microsecond=0)
    LOGGER.info(f"Time: {time}")
    station = STATIONS[station]
    if leadtime_days is not None:
        validtime, sounding = get_last_sounding_forecast(
            station["name"], time, leadtime_hrs=int(leadtime_days) * 24
        )
        sounding.attrs["source"] = f"GFS sounding +{leadtime_days * 24:.0f} h"
    else:
        validtime, sounding = get_last_sounding(station["stid"], time)
        sounding.attrs["source"] = f"Radiosounding 00Z {station['long_name']}"
    sounding = preprocess(sounding)
    sounding.attrs["validtime"] = validtime
    return sounding


@app.get("/", include_in_schema=False)
async def basic_view():
    return RedirectResponse("/docs")


@app.get("/cimetta")
async def predict_cimetta(time: str = "latest", leadtime_days: Optional[int] = None):

    # get inputs
    sounding = pipeline("Cameri", time, leadtime_days)
    validtime = sounding.attrs["validtime"]

    # fly prob
    model = tf.keras.models.load_model("models/fly_prob_1.h5")
    moments = xr.load_dataset("models/fly_prob_moments.nc")
    inputs = standardize(sounding, moments).values[None, ...]
    fly_prob = float(model.predict(inputs)[0][0])

    # max altitude
    model = tf.keras.models.load_model("models/fly_max_alt_1.h5")
    moments = xr.load_dataset("models/fly_max_moments.nc")
    inputs = standardize(sounding, moments).values[None, ...]
    # max_alt = int(model.predict(inputs)[0].argmax() * ALT_BIN + CIMETTA_ELEVATION)
    pred = np.array(model.predict(inputs)[0])
    max_alt = (
        np.sum(pred * (np.arange(pred.size) * ALT_BIN)) // 100 * 100 + CIMETTA_ELEVATION
    )

    # max distance
    model = tf.keras.models.load_model("models/fly_max_dist_1.h5")
    # max_dist = int(model.predict(inputs)[0].argmax() * DIST_BIN)
    pred = np.array(model.predict(inputs)[0])
    max_dist = np.sum(pred * (np.arange(pred.size) * DIST_BIN)) // 10 * 10

    return {
        "site": "Cimetta",
        "validtime": f"{validtime:%Y-%m-%d}",
        "flying_probability": fly_prob,
        "max_altitude_masl": max_alt,
        "max_distance_km": max_dist,
    }


@app.get("/cimetta_plot")
async def explain_cimetta(time: str = "latest", leadtime_days: Optional[int] = None):

    # get inputs
    sounding = pipeline("Cameri", time, leadtime_days)

    # fly prob
    model = tf.keras.models.load_model("models/fly_prob_1.h5")
    moments = xr.load_dataset("models/fly_prob_moments.nc")
    background = np.load("models/fly_prob_1_background.npy")
    inputs = standardize(sounding, moments).values[None, ...]
    fly_prob = float(model.predict(inputs)[0][0])
    shap_values = compute_shap(background, model, inputs)[0]

    # plot  shap
    fig = explainable_plot(sounding, shap_values, fly_prob)
    image_file = BytesIO()
    plt.savefig(image_file)
    plt.close(fig)
    image_file.seek(0)

    return StreamingResponse(image_file, media_type="image/png")
