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
    ds["DWPT"] = ds["TEMP"] - ds["DWPT"]
    return (ds[["TEMP", "DWPT", "U", "V"]]
            .rename({"PRES": "level"})
            .bfill(dim="level", limit=3)
            .to_array().transpose("level", "variable")
            .astype("float32")
            )


def standardize(da):
    """Standardize the input data with training mean and standard deviation."""
    moments = xr.load_dataset("models/fly_prob_moments.nc")
    return (da - moments.mu) / moments.sigma


def pipeline(station):
    """Clean and preprocess the input data"""
    date, data = get_last_radiosounding(station)
    data = preprocess(data)
    data = standardize(data)
    return date, data.values[None, ...]


@app.get('/cimetta')
def predict():
    validtime, inputs = pipeline(16080)
    model_fly_prob = tf.keras.models.load_model("models/fly_prob_1.h5")
    model_max_alt = tf.keras.models.load_model("models/fly_max_alt_1.h5")
    return {
        "site": "Cimetta",
        "validtime": f"{validtime:%Y-%m-%d}",
        "flying_probability": float(model_fly_prob.predict(inputs)[0][0]),
        "max_altitude": float(model_max_alt.predict(inputs)[0].argmax() * 300 + 1600),
    }


if __name__ == "__main__":
    print(predict())