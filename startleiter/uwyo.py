import logging
import time
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import requests
import xarray as xr
from bs4 import BeautifulSoup

import startleiter.scraping as scr
from startleiter.database import Database
from startleiter.utils import to_wind_components
from startleiter import config as CFG


logger = logging.getLogger(__name__)

COUNTER = 0
BASE_URL = "http://weather.uwyo.edu"
SEARCH_URL = BASE_URL + "/cgi-bin/sounding"
DEFAULT_QUERY = {
    "region": "europe",
    "TYPE": "TEXT%3ASKEWT",
    "YEAR": "2021",
    "MONTH": "01",
    "FROM": "0100",
    "TO": "0100",
    "STNM": "16080"
}
STATION_NAMES = {
    "LIML": 16080,  # Milano-Linate
    "LSMP": 6610,  # Payerne
}


# http://weather.uwyo.edu/cgi-bin/sounding?region=europe&TYPE=TEXT%3ASKEWT&YEAR=2009&MONTH=05&FROM=1512&TO=1512&STNM=16080

def year_month_from_to(from_dtime, to_dtime=None):
    """
    Convert datetime object to a dictinary with
    YEAR: %Y, MONTH: %M, FROM: %d%H, TO: %d%H

    Parameters
    ----------
    from_dtime: datetime.datetime
    to_dtime: datetime.datetime, optional

    Returns
    -------
    dict
    """
    if not to_dtime:
        to_dtime = from_dtime
    assert to_dtime.month == from_dtime.month, "must use the same month"
    assert from_dtime <= to_dtime, "invalid end time"
    return {
        "YEAR": f"{from_dtime:%Y}",
        "MONTH": f"{from_dtime:%m}",
        "FROM": f"{from_dtime:%d%H}",
        "TO": f"{to_dtime:%d%H}",
    }


def interp_sounding(dataset, ref_pres):
    dataset = to_wind_components(dataset)
    dataset = dataset.interp(PRES_hPa=ref_pres)
    dataset = to_wind_components(dataset, inverse=True)
    return dataset


def sounding_data(body):
    lines = body.text.split("\n")
    var_names = lines[2].split()
    var_units = lines[3].split()
    var_names = ["_".join((name, unit)) for name, unit in zip(var_names, var_units)]
    n_cols = len(var_names)
    data = []
    for row in lines[5:-1]:
        values = [row[(i * 7):(i + 1) * 7] for i in range(n_cols)]
        data.append([float(x) if x.strip() else np.nan for x in values])
    data_frame = pd.DataFrame(data, columns=var_names).set_index("PRES_hPa").drop("HGHT_m", 1)
    dataset = xr.Dataset.from_dataframe(data_frame)

    rename_dict = {}
    for var in dataset.data_vars:
        new_name, unit = dataset[var].name.split("_")
        rename_dict[dataset[var].name] = new_name
        dataset[var].attrs["units"] = unit
        dataset[var] = dataset[var].astype("float32")
    for coord in dataset.coords:
        new_name, unit = dataset[coord].name.split("_")
        rename_dict[dataset[coord].name] = new_name
        dataset[coord].attrs["units"] = unit
        dataset[coord] = dataset[coord].astype("float32")
    dataset = dataset.rename(rename_dict)

    ref_pres = np.logspace(np.log10(200), 3, 64, base=10)[::-1] // 1
    return interp_sounding(dataset, ref_pres)


def sounding_indices(body):
    lines = body.text.split("\n")
    indices = {line.split(":")[0].strip(): line.split(":")[1].strip() for line in lines if line.strip()}
    return indices


def sounding(soup):
    headings = soup.find_all("h2")
    body = soup.find_all("pre")
    assert len(headings) * 2 == len(body)
    soundings = {}
    for n,  heading in enumerate(headings):
        validtime = heading.text.split(" at ")[1]
        validtime = datetime.strptime(validtime, "%HZ %d %b %Y")
        data = sounding_data(body[n * 2])
        indices = sounding_indices(body[n * 2 + 1])
        soundings[validtime] = {
            "data": data,
            "indices": indices
        }
    return soundings


def scrape(station_name, from_validtime, to_validtime=None):
    """
    Parameters
    ----------
    station_name: str or int
        The station shortname or its identifier
    from_validtime: datetime.datetime
    to_validtime: datetime.datetime, optional

    Returns
    -------

    """
    if not isinstance(station_name, int):
        station_id = STATION_NAMES.get(station_name)
    else:
        station_id = station_name
    this_query = {"STNM": station_id}
    this_query.update(year_month_from_to(from_validtime, to_validtime))
    query_url = scr.build_query(SEARCH_URL, DEFAULT_QUERY, this_query)
    logger.info(query_url)

    page = requests.get(query_url)
    soup = BeautifulSoup(page.content, "html.parser")

    return sounding(soup)


if __name__ == "__main__":
    # Setup logging
    logging.basicConfig(
        # format="%(asctime)s,%(msecs)d %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s",
        format="%(levelname)-4s [%(filename)s:%(lineno)d] %(message)s",
        datefmt='%Y-%m-%d:%H:%M:%S',
        level=logging.INFO
    )

    # Define source
    source = CFG["sources"]["uwyo"]
    station = CFG["stations"]["Milano"]

    # Connect to database
    db = Database(source, station=station)

    t0 = time.monotonic()
    total_sleep = 0
    date_start = db.query_last_sounding() + timedelta(days=1)
    end_dates = pd.date_range(date_start, datetime.utcnow(), freq="M")
    for n, date in enumerate(end_dates):
        logger.info(f"Retrieving sounding data for {date:%b %Y}")
        soundings = scrape(station["name"], date.replace(day=1), date)
        db.insert_soundings(soundings)
        total_sleep = scr.pacing(n, len(end_dates), t0, 200, total_sleep)
