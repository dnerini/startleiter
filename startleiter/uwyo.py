import logging
import time
from datetime import datetime

import numpy as np
import pandas as pd
import requests
from bs4 import BeautifulSoup

import startleiter.scraping as scr
from startleiter.database import Database


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
    "LIML": 16080,
}


# http://weather.uwyo.edu/cgi-bin/sounding?region=europe&TYPE=TEXT%3ASKEWT&YEAR=2009&MONTH=05&FROM=1512&TO=1512&STNM=16080

def year_month_from_to(dtime):
    """
    Convert datetime object to a dictinary with
    YEAR: %Y, MONTH: %M, FROM: %d%H, TO: %d%H

    Parameters
    ----------
    dtime: datetime.datetime

    Returns
    -------
    dict
    """
    return {
        "YEAR": f"{dtime:%Y}",
        "MONTH": f"{dtime:%m}",
        "FROM": f"{dtime:%d%H}",
        "TO": f"{dtime:%d%H}",
    }


def sounding_data(soup):
    text = soup.find_all("pre")[0].text
    lines = text.split("\n")
    header = lines[2].split()
    units = lines[3].split()
    columns = ["_".join((name, unit)) for name, unit in zip(header, units)]
    n_cols = len(columns)
    data = []
    for row in lines[5:-1]:
        values = [row[(i * 7):(i + 1) * 7] for i in range(n_cols)]
        data.append([float(x) if x.strip() else np.nan for x in values])
    data_frame = pd.DataFrame(data, columns=columns)
    logger.info(data_frame)
    return data_frame


def sounding_indices(soup):
    text = soup.find_all("pre")[1].text
    lines = text.split("\n")
    indices = {line.split(":")[0].strip(): line.split(":")[1].strip() for line in lines if line.strip()}
    logger.info(indices)
    return indices


def scrape(station_name, validtime):
    """

    Parameters
    ----------
    station_name: str or int
        The station shortname or its identifier
    validtime: datetime.datetime

    Returns
    -------

    """
    if not isinstance(station_name, int):
        station_id = STATION_NAMES.get(station_name)
    else:
        station_id = station_name
    this_query = {"STNM": station_id}
    this_query.update(year_month_from_to(validtime))
    query_url = scr.build_query(SEARCH_URL, DEFAULT_QUERY, this_query)
    logger.info(query_url)

    page = requests.get(query_url)
    soup = BeautifulSoup(page.content, "html.parser")

    sdata = sounding_data(soup)
    sindices = sounding_indices(soup)

    return sdata, sindices


def scrape_multi(station_name, timestamps, pace):
    t0 = time.monotonic()
    total_sleep = 0
    all_sdata = []
    all_sindices = []
    for n, timestamp in enumerate(timestamps):
        sdata, sindices = scrape(station_name, timestamp)
        all_sdata.append(sdata)
        all_sindices.append(sindices)
        total_sleep = scr.pacing(n, len(timestamps), t0, pace, total_sleep)
    return all_sdata, all_sindices


if __name__ == "__main__":
    # Setup logging
    logging.basicConfig(
        # format="%(asctime)s,%(msecs)d %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s",
        format="%(levelname)-4s [%(filename)s:%(lineno)d] %(message)s",
        datefmt='%Y-%m-%d:%H:%M:%S',
        level=logging.INFO
    )

    # Define source
    source = {
        "name": "uwyo",
        "base_url": BASE_URL,
    }

    station = {
        "name": "LIML",
        "long_name": "Milano",
        "stid": 16080,
        "country": "Italy",
        "latitude": 45.43,
        "longitude": 9.28,
        "elevation": 103.0
    }

    # Connect to database
    db = Database(source, station=station)

    validtime = datetime(2021, 1, 1, 0)
    sdata, sindices = scrape(station["name"], validtime)
    db.insert_sounding(validtime, sindices)
