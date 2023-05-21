import logging
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd
import xarray as xr

import startleiter.scraping as scr
from startleiter.uwyo import interp_sounding
from startleiter import config as CFG


logger = logging.getLogger(__name__)

COUNTER = 0
BASE_URL = "https://rucsoundings.noaa.gov"
SEARCH_URL = BASE_URL + "/get_soundings.cgi"
DEFAULT_QUERY = {
    "data_source": "GFS",
    "startSecs": 1652097600,  # unix time
    "fcst_len": 24,
    "airport": "LIML",
}

# https://rucsoundings.noaa.gov/get_soundings.cgi?data_source=GFS&startSecs=1652097600&fcst_len=24&airport=LIML


def sounding_data(body):
    lines = body.text.split("\n")
    var_names = lines[2].split()
    var_units = lines[3].split()
    var_names = ["_".join((name, unit)) for name, unit in zip(var_names, var_units)]
    n_cols = len(var_names)
    data = []
    for row in lines[5:-1]:
        values = [row[(i * 7) : (i + 1) * 7] for i in range(n_cols)]
        data.append([float(x) if x.strip() else np.nan for x in values])
    data_frame = (
        pd.DataFrame(data, columns=var_names).set_index("PRES_hPa").drop("HGHT_m", 1)
    )
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


def sounding(soup):
    headings = soup.find_all("table")
    body = soup.find_all("pre")
    assert len(headings) * 2 == len(body)
    soundings = {}
    for n, heading in enumerate(headings):
        validtime = heading.text.split(" at ")[1]
        validtime = datetime.strptime(validtime, "%HZ %d %b %Y")
        data = sounding_data(body[n * 2])
        soundings[validtime] = {
            "data": data,
        }
    return soundings


def scrape(station_name, date_start, leadtime):
    """
    Parameters
    ----------
    station: str
        The station shortname or its identifier
    date_start: datetime
    leadtime: timedelta

    Returns
    -------

    """
    validtime = date_start + leadtime
    this_query = {
        "airport": station_name,
        "startSecs": int(date_start.replace(tzinfo=timezone.utc).timestamp()),
        "endSecs": int(
            (date_start + timedelta(hours=3)).replace(tzinfo=timezone.utc).timestamp()
        ),
        "fcst_len": int(leadtime.total_seconds() / 3600),
    }
    query_url = scr.build_query(SEARCH_URL, DEFAULT_QUERY, this_query)
    logger.info(query_url)
    col_names = (
        "TYPE",
        "PRES_hPa",
        "HGHT_m",
        "TEMP_C",
        "DWPT_C",
        "DRCT_deg",
        "SKNT_knot",
    )
    table = pd.read_fwf(
        query_url, skiprows=6, header=None, names=col_names, na_values=99999
    )
    table["PRES_hPa"] *= 0.1
    table["TEMP_C"] *= 0.1
    table["DWPT_C"] *= 0.1
    table = table.set_index("PRES_hPa").drop("TYPE", axis=1)
    dataset = xr.Dataset.from_dataframe(table)
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
    dataset = interp_sounding(dataset, ref_pres)
    return {validtime: {"data": dataset}}


if __name__ == "__main__":
    source = CFG["sources"]["rucsoundings"]
    station = CFG["stations"]["Milano"]
    leadtime = timedelta(hours=120)
    date_start = datetime.utcnow()
    date_start = date_start.replace(hour=date_start.hour // 24 * 24)
    date_start = date_start.replace(minute=0, second=0, microsecond=0)
    sounding = scrape(station["name"], date_start, leadtime)
    print(sounding)
