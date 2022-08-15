import logging
import os
import random
import time
from datetime import datetime, timedelta


from bs4 import BeautifulSoup
from selenium.common.exceptions import (
    TimeoutException,
    StaleElementReferenceException,
    WebDriverException,
)
from selenium.webdriver.common.by import By
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import WebDriverWait

import startleiter.scraping as scr
from startleiter.database import Database
from startleiter import config as CFG

LOGGER = logging.getLogger(__name__)
TIME_START = time.monotonic()
NUM_FLIGHTS_ON_PAGE = 50
BUFFER_DAYS = 14
COUNTER = 0
STOP_AFTER = 50
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


def login_xcontest(browser):
    username = os.environ.get("XCONTEST_USERNAME")
    password = os.environ.get("XCONTEST_PASSWORD")
    browser.get("https://www.xcontest.org")
    browser.find_element(By.ID, "login-username").send_keys(username)
    browser.find_element(By.ID, "login-password").send_keys(password)
    browser.find_element(By.CLASS_NAME, "submit").submit()
    time.sleep(3)
    return browser


def flight_summary(table_row):
    """Parse flight summary data from a given query.

    Parameters
    ----------
    table_row

    Returns
    -------
    summary: list
        flid, no., start time, pilot, launch, route, length, points, glider_class, glider, route
    """
    cols = table_row.find_all("td")  # 10 columns
    text = [ele.text.strip() for ele in cols][:-2]
    flid = cols[0].get("title").split(":")[1]
    glider = cols[7].find_all("div")[0].get("title")
    route = cols[4].find_all("div")[0].get("title")
    summary = [flid, *text, glider, route]
    LOGGER.debug(summary)
    return summary


def _flight_details_href(row):
    for div in row.find_all("div"):
        details = div.find_all("a", {"class": "detail"})
        if details:
            return details[0].get("href")
    return None


def _parse_table(soup, class_name):
    table = soup.find("table", attrs={"class": class_name})
    table_body = table.find("tbody")
    rows = table_body.find_all("tr")
    content = []
    for row in rows:
        cols = row.find_all("td")
        cols = [ele.text.strip() for ele in cols]
        content.append([ele for ele in cols if ele])  # Get rid of empty values
    return content


def flight_details(flight, browser):
    """Parse flight detail data from a given query.

    Parameters
    ----------
    flight
    browser

    Returns
    -------
    details: list
        airtime, max. altitude, max. alt. gain, max. climb, max. sink, tracklog length, free distance
    """
    href = _flight_details_href(flight)
    if not href:
        return None
    url = BASE_URL + href
    LOGGER.debug(url)

    # browser.delete_all_cookies()  # after this, will need to login again!
    browser.get(url + "#fd=flight")
    loaded = scr.wait_till_loaded(browser, url)
    if not loaded:
        raise WebDriverException
    authorized = "You are not authorized to see the flight" not in browser.page_source
    if not authorized:
        raise WebDriverException("Not authorized to see the flight")
    element = WebDriverWait(
        browser, 20, ignored_exceptions=StaleElementReferenceException
    ).until(EC.element_to_be_clickable((By.LINK_TEXT, "Flight")))
    element.click()
    soup = BeautifulSoup(browser.page_source, "html.parser")
    subsoup = soup.find("div", attrs={"class": "XCmoreInfo"})
    details = _parse_table(subsoup, "XCinfo")
    details = [detail[0] if isinstance(detail, list) else None for detail in details]
    LOGGER.debug(details)
    return details


def parse_flights(browser, pace):
    """Loop all flights for a given query on xcontest.org, extract
    flight summary and details.

    Parameters
    ----------
    browser: selenium.WebDriver
        A selenium-webdriver client.
    pace: int
        The desired pacing in number of requests per hour.

    Returns
    -------

    """
    global COUNTER

    soup = BeautifulSoup(browser.page_source, "html.parser")
    table = soup.find("table", attrs={"class": "flights"})
    table_body = table.find("tbody")
    flights = table_body.find_all("tr")
    if len(flights) < NUM_FLIGHTS_ON_PAGE:
        LOGGER.warning(
            f"Not enough new flights available ({len(flights)} < {NUM_FLIGHTS_ON_PAGE}), skipping."
        )
        return None

    t0 = time.monotonic()
    total_sleep = 0
    consecutive_timeouts = 0
    flights_data = []
    for n, flight in enumerate(flights):

        flight_data = []

        # Parse flight summary
        try:
            summary = flight_summary(flight)
        except (AttributeError, IndexError):
            LOGGER.error(f"Failed to parse flight {n}")
            continue

        flight_datetime_str = (
            f"{summary[2]}+00:00" if summary[2][-3:] == "UTC" else summary[2]
        )
        flight_datetime = datetime.strptime(
            flight_datetime_str, "%d.%m.%y %H:%MUTC%z"
        ).replace(tzinfo=None)
        if datetime.utcnow() - flight_datetime < timedelta(days=BUFFER_DAYS):
            LOGGER.warning(
                f"Available flights are less than {BUFFER_DAYS} day old ({flight_datetime.isoformat()}), skipping."
            )
            return None

        flight_data += summary

        # Parse flight details
        try:
            details = flight_details(flight, browser)
        except (AttributeError, TimeoutException, WebDriverException) as e:
            details = [None] * 7
            if e.__class__.__name__ == "TimeoutException":
                consecutive_timeouts += 1
                print("T", end="", flush=True)
            else:
                print("F", end="", flush=True)
        else:
            print(".", end="", flush=True)
            consecutive_timeouts = 0
        finally:
            flight_data += details

        if consecutive_timeouts > 3:
            raise TimeoutException

        # TODO: Parse flight track

        flights_data.append(flight_data)
        total_sleep = scr.pacing(n, len(flights), t0, pace, total_sleep)
        COUNTER += 1

        if COUNTER == STOP_AFTER:
            break

    print("")
    return flights_data


def main(site, pace):
    """Main scraping routine for xcontest.org"""
    # Connect to database
    db = Database(CFG["sources"]["xcontest"], site)

    browser = scr.launch_browser()
    browser = login_xcontest(browser)

    try:
        id_start = db.query_last_flight()
        while COUNTER < STOP_AFTER:
            LOGGER.debug(
                f"XContest Flight Chunk {id_start} to {id_start + min(STOP_AFTER - 1, NUM_FLIGHTS_ON_PAGE)}"
            )
            this_query = {
                "list[start]": id_start,
                "filter[point]": f"{site['longitude']}%20{site['latitude']}",
                "filter[radius]": site["radius"],
            }
            query_url = scr.build_query(SEARCH_URL, DEFAULT_QUERY, this_query)
            LOGGER.info(query_url)
            browser.get(query_url)
            flight_chunk = parse_flights(browser, pace)
            if flight_chunk:
                db.insert_flights(flight_chunk)
                id_start += NUM_FLIGHTS_ON_PAGE
            else:
                break

    except TimeoutException as err:
        time_lapsed = time.monotonic() - TIME_START
        raise RuntimeError(
            f"Timeout after {COUNTER} queries and {int(time_lapsed / 60)} minutes."
        ) from err

    finally:
        browser.quit()
        # cleanup()


if __name__ == "__main__":

    logging.basicConfig(
        # format="%(asctime)s,%(msecs)d %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s",
        format="%(levelname)-4s [%(filename)s:%(lineno)d] %(message)s",
        datefmt="%Y-%m-%d:%H:%M:%S",
        level=logging.INFO,
    )
    sites = list(CFG["sites"].items())
    random.shuffle(sites)
    for site_name, site in sites:
        LOGGER.info(f"Site: {site_name}")
        main(site, pace=400)
        if COUNTER >= STOP_AFTER:
            break

    time_lapsed = time.monotonic() - TIME_START
    LOGGER.info(
        f"Successfully retrieved {COUNTER} flight records in {time_lapsed / 60:.1f} minutes."
    )
