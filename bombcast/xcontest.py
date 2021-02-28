import getpass
import logging
import time
from pprint import pprint

import pandas as pd
import psutil
import requests
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.common.exceptions import TimeoutException, StaleElementReferenceException
from selenium.webdriver.common.by import By
from selenium.webdriver.firefox.options import Options
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import WebDriverWait

from bombcast.database import Database

logger = logging.getLogger(__name__)

COUNTER = 0

BASE_URL = "https://www.xcontest.org"
SEARCH_URL = BASE_URL + "/world/en/flights-search"
DEFAULT_QUERY = {
    "list[sort]": "time_start",
    "list[dir]": "up",
    "list[start]": 0,
    "filter[point]": "8.78796%2046.1996",
    "filter[radius]": 1000,
    "filter[mode]": "START",
    "filter[date_mode]": "dmy",
    "filter[date]": "",
    "filter[value_mode]": "dst",
    "filter[min_value_dst]": "",
    "filter[catg]": "",
    "filter[route_types]": "",
    "filter[avg]": "",
    "filter[pilot]": "",
}


def launch_browser():
    options = Options()
    options.headless = True
    # options.log.level = "trace"  # log output is stored in geckodriver.log (current dir)
    browser = webdriver.Firefox(
        options=options,
        executable_path="/home/ned/.local/bin/geckodriver"
    )
    browser.set_page_load_timeout(5)
    browser.set_script_timeout(5)
    return browser


def query_flights(parameters=None):
    if parameters is None:
        parameters = {}
    query = DEFAULT_QUERY.copy()
    query.update(parameters)
    query_str = "&".join([f"{key}={value}" for key, value in query.items()])
    return SEARCH_URL + "/?" + query_str


def parse_table(soup, class_name):
    table = soup.find("table", attrs={"class": class_name})
    table_body = table.find("tbody")
    rows = table_body.find_all("tr")
    content = []
    for row in rows:
        cols = row.find_all("td")
        cols = [ele.text.strip() for ele in cols]
        content.append([ele for ele in cols if ele])  # Get rid of empty values
    return content


def flight_details_href(row):
    for div in row.find_all("div"):
        details = div.find_all("a", {"class": "detail"})
        if details:
            return details[0].get("href")
    return None


def wait_till_loaded(browser, url, maxattempts=3):
    try:
        pageid = url.split("/detail:")[1]
    except IndexError:
        return 0
    attempt = 0
    while pageid not in browser.current_url:
        time.sleep(2 ** attempt)
        attempt += 1
        if attempt > maxattempts:
            return 0
    return 1


def flight_details(flight):
    href = flight_details_href(flight)
    if not href:
        return None
    url = BASE_URL + href
    logger.info(url)

    try:
        browser.delete_all_cookies()
        browser.get(url + "#fd=flight")
    except Exception as e:
        logger.error(e)
        return None
    loaded = wait_till_loaded(browser, url)
    if not loaded:
        logger.error("Failed to load Flight Details")
        return None
    try:
        element = WebDriverWait(browser, 20, ignored_exceptions=StaleElementReferenceException).until(
            EC.element_to_be_clickable((By.LINK_TEXT, "Flight"))
        )
    except TimeoutException:
        logger.exception("Timeout while looking for Flight Details")
        raise
    element.click()
    soup = BeautifulSoup(browser.page_source, "html.parser")
    subsoup = soup.find("div", attrs={"class": "XCmoreInfo"})
    try:
        details = parse_table(subsoup, "XCinfo")
    except Exception as ex:
        logger.error(ex)
        return None
    details = [detail[0] if isinstance(detail, list) else None for detail in details]
    logger.info(details)
    return details


def as_dataframe(data):
    column_names = ["flid", "no.", "start time", "pilot", "launch", "route", "length", "points", "glider_class", "glider"]
    column_names += ["airtime", "max. altitude", "max. alt. gain", "max. climb"]
    column_names += ["max. sink", "tracklog length", "free distance"]
    df = pd.DataFrame(data, columns=column_names)
    df.set_index("No.", inplace=True)
    return df

def parse_flight(row):
    cols = row.find_all("td")  # 10 columns
    text = [ele.text.strip() for ele in cols][:-2]
    flid = cols[0].get("title").split(":")[1]
    glider = cols[7].find_all("div")[0].get("title")
    route = cols[4].find_all("div")[0].get("title")
    return [flid, *text, glider, route]


def parse_flights(query_url, pace):
    global COUNTER
    page = requests.get(query_url)
    soup = BeautifulSoup(page.content, "html.parser")
    table = soup.find("table", attrs={"class": "flights"})
    table_body = table.find("tbody")
    rows = table_body.find_all("tr")
    flights = []
    t0 = time.monotonic()
    total_sleep = 0
    for n, row in enumerate(rows):
        try:
            flight = parse_flight(row)
        except IndexError:
            logger.error(f"Failed to parse flight {n}")
            continue
        logger.info(flight)
        details = flight_details(row)
        if details:
            flight += details
        else:
            flight += [None, ] * 7
        flights.append(flight)
        # pacing and spacing
        time_lapsed = time.monotonic() - t0
        requests_done = n + 1
        requests_togo = len(rows) - requests_done
        logger.info(f"Averaging {requests_done / time_lapsed * 3600:.0f} requests per hour")
        if requests_togo > 0:
            time_togo = (requests_done + requests_togo) / pace * 3600 - time_lapsed
            spacing = max(0, time_togo / requests_togo - (time_lapsed - total_sleep) / requests_done)
            if spacing > 0:
                logger.info(f"Slowing down... wait {spacing:.1f} seconds...")
                time.sleep(spacing)
                total_sleep += spacing

        COUNTER += 1
        if COUNTER == stop_after:
            break
    return flights


def cleanup():
    for proc in psutil.process_iter():
        if (proc.name() in ["geckodriver", "Web Content", "firefox-bin"]
                and proc.username() == getpass.getuser()):
            try:
                proc.kill()
            except psutil.NoSuchProcess:
                pass


if __name__ == "__main__":

    logging.basicConfig(
        # format="%(asctime)s,%(msecs)d %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s",
        format="%(levelname)-4s [%(filename)s:%(lineno)d] %(message)s",
        datefmt='%Y-%m-%d:%H:%M:%S',
        level=logging.INFO
    )

    # Define source
    source = {
        "name": "xcontest",
        "base_url": BASE_URL,
    }

    # Define launching site
    site = {
        "name": "Cimetta",
        "country": "CH",
        "longitude": 8.78796,
        "latitude": 46.1996,
        "radius": 1000,
    }

    # Connect to database
    db = Database(source, site)

    # set pacing in number of requests per hour
    pace = 1000
    stop_after = 1e5

    browser = launch_browser()
    try:
        time_start = time.monotonic()
        id_start = db.query_last_flight()
        while id_start < stop_after:
            logger.info(f"XContest Flight Chunk {id_start} to {id_start + min(stop_after - 1, 50)}")
            flight_chunk = parse_flights(query_flights({"list[start]": id_start}), pace)
            if flight_chunk:
                db.insert_flights(flight_chunk)
                id_start += 50
            else:
                break
    except TimeoutException:
        time_lapsed = time.monotonic() - time_start
        logger.error(f"Timeout after {COUNTER} queries and {int(time_lapsed / 60)} minutes.")

    finally:
        browser.quit()
        # cleanup()

        df = db.to_pandas()
        print(df.head())
        print(df.describe())
