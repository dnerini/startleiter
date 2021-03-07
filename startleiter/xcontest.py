import logging
import time

import requests
from bs4 import BeautifulSoup

from selenium.common.exceptions import TimeoutException, StaleElementReferenceException
from selenium.webdriver.common.by import By
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import WebDriverWait

from startleiter.database import Database
from startleiter.scraping import launch_browser, pacing, wait_till_loaded

logger = logging.getLogger(__name__)

COUNTER = 0
STOP_AFTER = 1e6
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


def build_query(parameters=None):
    """Build search url for xcontest.org.
    
    Parameters
    ----------
    parameters: dict
        Optional parameters to change wrt the default query.

    Returns
    -------
    query_url: str
    """
    if parameters is None:
        parameters = {}
    query = DEFAULT_QUERY.copy()
    query.update(parameters)
    query_str = "&".join([f"{key}={value}" for key, value in query.items()])
    return SEARCH_URL + "/?" + query_str


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
    logger.info(summary)
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
    logger.info(url)

    browser.delete_all_cookies()
    try:
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
        details = _parse_table(subsoup, "XCinfo")
    except Exception as ex:
        logger.error(ex)
        return None
    details = [detail[0] if isinstance(detail, list) else None for detail in details]
    logger.info(details)
    return details


def parse_flights(query_url, browser, pace=120):
    """Loop all flights for a given query on xcontest.org, extract
    flight summary and details.

    Parameters
    ----------
    query_url: str
        A search query url for xcontest.org.
    browser: selenium.WebDriver
        A selenium-webdriver client.
    pace: int
        The desired pacing in number of requests per hour.

    Returns
    -------

    """
    global COUNTER

    page = requests.get(query_url)
    soup = BeautifulSoup(page.content, "html.parser")
    table = soup.find("table", attrs={"class": "flights"})
    table_body = table.find("tbody")
    flights = table_body.find_all("tr")

    t0 = time.monotonic()
    total_sleep = 0
    consecutive_timeouts = 0
    flights_data = []
    for n, flight in enumerate(flights):

        flight_data = []

        # Parse flight summary
        try:
            summary = flight_summary(flight)
        except IndexError:
            logger.error(f"Failed to parse flight {n}")
            continue
        flight_data += summary

        # Parse flight details
        try:
            details = flight_details(flight, browser)
        except TimeoutException:
            consecutive_timeouts += 1
            details = [None, ] * 7
        else:
            consecutive_timeouts = 0
        flight_data += details

        if consecutive_timeouts > 3:
            raise TimeoutException

        # TODO: Parse flight track

        flights_data.append(flight_data)

        total_sleep = pacing(n, len(flights), t0, pace, total_sleep)

        COUNTER += 1
        if COUNTER == STOP_AFTER:
            break

    return flights_data


def scrape(source, site, pace):
    """Main scraping routine for xcontest.org
    """
    # Connect to database
    db = Database(source, site)

    browser = launch_browser()
    time_start = time.monotonic()

    try:
        id_start = db.query_last_flight()
        while id_start < STOP_AFTER:
            logger.info(f"XContest Flight Chunk {id_start} to {id_start + min(STOP_AFTER - 1, 50)}")
            query_url = build_query({"list[start]": id_start})
            flight_chunk = parse_flights(query_url, browser, pace)
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
        print(df)
        print(df.describe())


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

    # Set pace and start main routine
    pace = 120  # requests / hour
    scrape(source, site, pace)