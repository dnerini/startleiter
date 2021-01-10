import getpass
import logging
import time
from pprint import pprint

import pandas as pd
import psutil
import requests
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.common.exceptions import TimeoutException
from selenium.webdriver.common.by import By
from selenium.webdriver.firefox.options import Options
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import WebDriverWait

logging.basicConfig(
    # format="%(asctime)s,%(msecs)d %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s",
    format="%(levelname)-4s [%(filename)s:%(lineno)d] %(message)s",
    datefmt='%Y-%m-%d:%H:%M:%S',
    level=logging.INFO
)

logger = logging.getLogger(__name__)

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
    options.log.level = "trace"  # log output is stored in geckodriver.log (current dir)
    browser = webdriver.Firefox(
        options=options,
        executable_path="/users/ned/.local/bin/geckodriver"
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
        content.append([ele for ele in cols if ele]) # Get rid of empty values
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

def flight_details(flight, maxattemps=3):
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
        element = WebDriverWait(browser, 20).until(
            EC.element_to_be_clickable((By.LINK_TEXT , "Flight"))
        )
    except TimeoutException:
        logger.error("Timeout while looking for Flight Details")
        return None
    element.click()
    soup = BeautifulSoup(browser.page_source, "html.parser")
    subsoup = soup.find("div", attrs={"class": "XCmoreInfo"})
    try:
        details = parse_table(subsoup, "XCinfo")
    except Exception as ex:
        logger.error(ex)
        return None
    details = [detail[0] for detail in details if isinstance(detail, list)]
    logger.info(details)
    return details

def as_dataframe(data):
    column_names = ["No.", "start time", "pilot", "launch", "route", "length", "points", "glider"]
    column_names += ["airtime", "max. altitude", "max. alt. gain", "max. climb"]
    column_names += ["max. sink", "tracklog length", "free distance"]
    df = pd.DataFrame(data, columns=column_names)
    df.set_index("No.", inplace=True)
    return df

def parse_flights(query_url, pace):
    page = requests.get(query_url)
    soup = BeautifulSoup(page.content, "html.parser")
    table = soup.find("table", attrs={"class": "flights"})
    table_body = table.find("tbody")
    rows = table_body.find_all("tr")
    flights = []
    t0 = time.monotonic()
    spacing = 0
    for n, row in enumerate(rows):
        cols = row.find_all("td")
        cols = [ele.text.strip() for ele in cols]
        flight = [ele for ele in cols if ele]
        flight.pop(-1)
        logger.info(
            f"({flight[0]}) {flight[2]} on {flight[1]} from {flight[3]}"
        )
        details = flight_details(row)
        if details:
            flight += details
        else:
            flight += [None,] * 7
        flights.append(flight)
        # pacing and spacing
        time_lapsed = time.monotonic() - t0
        requests_done = n + 1
        requests_togo = len(rows) - requests_done
        logger.info(f"Averaging {requests_done / time_lapsed * 3600:.0f} requests per hour")
        if requests_togo > 0:
            time_togo = (requests_done + requests_togo) / pace * 3600 - time_lapsed
            spacing = max(0, time_togo / requests_togo - time_lapsed / requests_done)
            if spacing > 0:
                logger.info(f"Slowing down... wait {spacing:.1f} seconds...")
                time.sleep(spacing)
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

    # set pacing in number of requests per hour
    pace = 1e9 # 224 requests before blocked (no pacing)
    pace = 500

    browser = launch_browser()
    try:
        flights = []
        id_start = 0
        while id_start < 5000:
            logger.info(f"XContest Flight Chunk {id_start} to {id_start + 50}")
            flight_chunk = parse_flights(query_flights({"list[start]": id_start}), pace)
            if flight_chunk:
                flights += flight_chunk
                id_start += 50
            else:
                break
    finally:
        browser.quit()
        cleanup()

    flights = as_dataframe(flights)
    pprint(flights)
    flights.to_csv("xcontest_cimetta.csv")
