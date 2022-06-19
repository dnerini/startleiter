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
ALT_BINS = [0, 500, 1000, 1500, 2000]
DIST_BINS = [0, 50, 100, 150]

CIMETTA_ELEVATION = 1600
ALT_BINS = [(x + CIMETTA_ELEVATION) // 500 * 500 for x in ALT_BINS]

STATIONS = CFG["stations"]

MODEL_FLYABILITY = tf.keras.models.load_model("models/flyability_1.h5")
MODEL_MAX_ALT = tf.keras.models.load_model("models/fly_max_alt_1.h5")
MODEL_MAX_DIST = tf.keras.models.load_model("models/fly_max_dist_1.h5")

MOMENTS_FLYABILITY = xr.load_dataset("models/flyability_moments.nc")
MOMENTS_MAX = xr.load_dataset("models/fly_max_moments.nc")

BACKGROUND = np.load("models/flyability_1_background.npy")


@try_wait()
def get_last_sounding(station, time):
    data = list(uwyo.scrape(station, time).items())[0]
    return data[0], data[1]["data"]


@try_wait()
def get_last_sounding_forecast(station, time, leadtime_hrs):
    leadtime = timedelta(hours=leadtime_hrs)
    data = list(rucsoundings.scrape(station, time, leadtime).items())[0]
    return data[0], data[1]["data"]


def preprocess(ds):
    ds = to_wind_components(ds)
    # dew point temperature depression
    ds["DWPD"] = ds["TEMP"] - ds["DWPT"]
    ds["WOY"] = ds.attrs["validtime"].isocalendar().week
    (ds,) = xr.broadcast(ds)
    return (
        ds[["TEMP", "DWPD", "U", "V", "WOY"]]
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


@lru_cache(maxsize=6)
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
    sounding.attrs["validtime"] = validtime
    sounding = preprocess(sounding)
    sounding = sounding.sel(level=slice(1000, 400))
    return sounding


@app.get("/", include_in_schema=False)
async def basic_view():
    return RedirectResponse("/docs")


@app.get("/cimetta")
async def predict_cimetta(time: str = "latest", leadtime_days: Optional[int] = None):

    # get inputs
    sounding = pipeline("Cameri", time, leadtime_days)
    validtime = sounding.attrs["validtime"]

    # flyability
    inputs = standardize(sounding, MOMENTS_FLYABILITY).values[None, ...]
    fly_prob = float(MODEL_FLYABILITY.predict(inputs)[0][0])

    # max altitude and distance
    inputs = standardize(sounding, MOMENTS_MAX).values[None, ...]
    max_alt = ALT_BINS[int(MODEL_MAX_ALT.predict(inputs)[0].argmax())]
    max_dist = DIST_BINS[int(MODEL_MAX_DIST.predict(inputs)[0].argmax())]

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

    # flyability
    inputs = standardize(sounding, MOMENTS_FLYABILITY).values[None, ...]
    fly_prob = float(MODEL_FLYABILITY.predict(inputs)[0][0])
    shap_values = compute_shap(BACKGROUND, MODEL_FLYABILITY, inputs)[0]

    # plot  shap
    fig = explainable_plot(sounding, shap_values, fly_prob)
    image_file = BytesIO()
    plt.savefig(image_file)
    plt.close(fig)
    image_file.seek(0)

    return StreamingResponse(image_file, media_type="image/png")
