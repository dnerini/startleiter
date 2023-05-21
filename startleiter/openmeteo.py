import logging

import requests
import pandas as pd

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
