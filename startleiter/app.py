from datetime import datetime
from io import BytesIO

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import xarray as xr
from fastapi import FastAPI
from starlette.responses import StreamingResponse, RedirectResponse

from startleiter.decorators import try_wait
from startleiter.explainer import compute_shap, explainable_plot
from startleiter.utils import to_wind_components
from startleiter.uwyo import scrape

app = FastAPI()


# TODO: do not hardcode
CIMETTA_ELEVATION = 1600
ALT_BIN = 300
DIST_BIN = 15


@try_wait()
def get_last_sounding(station):
    time = datetime.utcnow().replace(hour=0, minute=0, second=0, microsecond=0)
    data = list(scrape(station, time).items())[0]
    return data[0], data[1]["data"]


def preprocess(ds):
    ds = to_wind_components(ds)
    # dew point temperature depression
    ds["DWPD"] = ds["TEMP"] - ds["DWPT"]
    return (ds[["TEMP", "DWPD", "U", "V"]]
            .rename({"PRES": "level"})
            .bfill(dim="level", limit=3)
            .to_array().transpose("level", "variable")
            .astype("float32")
            )


def standardize(da, moments, inverse=False):
    """Standardize the input data with training mean and standard deviation."""
    if not inverse:
        return (da - moments.mu) / moments.sigma
    else:
        return da * moments.sigma + moments.mu


def pipeline(station):
    """Clean and preprocess the input data"""
    validtime, sounding = get_last_sounding(station)
    sounding = preprocess(sounding)
    sounding.attrs["validtime"] = validtime
    return sounding


@app.get('/', include_in_schema=False)
async def basic_view():
    return RedirectResponse('/docs')


@app.get('/cimetta')
async def predict():

    # get inputs
    sounding = pipeline(16080)
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
    max_alt = float(model.predict(inputs)[0].argmax() * ALT_BIN + CIMETTA_ELEVATION)

    # max distance
    model = tf.keras.models.load_model("models/fly_max_dist_1.h5")
    max_dist = float(model.predict(inputs)[0].argmax() * DIST_BIN)

    return {
        "site": "Cimetta",
        "validtime": f"{validtime:%Y-%m-%d}",
        "flying_probability": fly_prob,
        "max_altitude_masl": max_alt,
        "max_distance_km": max_dist,
    }


@app.get('/cimetta_plot')
async def explain():

    # get inputs
    sounding = pipeline(16080)

    # fly prob
    model = tf.keras.models.load_model("models/fly_prob_1.h5")
    moments = xr.load_dataset("models/fly_prob_moments.nc")
    background = np.load("models/fly_prob_1_background.npy")
    inputs = standardize(sounding, moments).values[None, ...]
    fly_prob = float(model.predict(inputs)[0][0])
    shap_values = compute_shap(background, model, inputs)[0]

    # plot  shap
    fig = explainable_plot(sounding, shap_values, fly_prob)
    fig.tight_layout()
    image_file = BytesIO()
    plt.savefig(image_file)
    plt.close(fig)
    image_file.seek(0)

    return StreamingResponse(image_file, media_type="image/png")
