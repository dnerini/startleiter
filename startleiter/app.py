from datetime import datetime

import tensorflow as tf
import xarray as xr
from fastapi import FastAPI

from startleiter.utils import to_wind_components
from startleiter.uwyo import scrape

app = FastAPI()


def get_last_radiosounding(station):
    time = datetime.utcnow().replace(hour=0, minute=0, second=0, microsecond=0)
    data = list(scrape(station, time).items())[0]
    return data[0], data[1]["data"]


def preprocess(ds):
    ds = to_wind_components(ds)
    return (ds[["TEMP", "DWPT", "U", "V"]]
            .rename({"PRES": "level"})
            .bfill(dim="level", limit=3)
            .to_array().transpose("level", "variable")
            .astype("float32")
            )


def standardize(da):
    """Standardize the input data with training mean and standard deviation."""
    moments = xr.load_dataset("models/moments.nc")
    return (da - moments.mu) / moments.sigma


def pipeline(station):
    """Clean and preprocess the input data"""
    date, data = get_last_radiosounding(station)
    data = preprocess(data)
    data = standardize(data)
    return date, data


@app.get('/cimetta')
def predict():
    date, data = pipeline(16080)
    model = tf.keras.models.load_model("models/conv1d.h5")
    prediction = model.predict(data.values[None, ...])[0]
    flying_prob = float(prediction[1])
    return {
        "site": "Cimetta",
        "validtime": f"{date:%Y-%m-%d}",
        "flying_probability": flying_prob,
    }