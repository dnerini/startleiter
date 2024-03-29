import getpass
import logging
import time

import psutil
from selenium import webdriver
from selenium.webdriver import FirefoxOptions

LOGGER = logging.getLogger(__name__)


def launch_browser():
    options = FirefoxOptions()
    options.add_argument("--headless")
    browser = webdriver.Firefox(options=options)
    browser.set_page_load_timeout(60)
    browser.set_script_timeout(30)
    return browser


def build_query(search_url, default_query, parameters=None):
    """Build search url.

    Parameters
    ----------
    search_url: str
    default_query: dict
    parameters: dict, optional
        Optional parameters to change wrt the default query.

    Returns
    -------
    query_url: str
    """
    if parameters is None:
        parameters = {}
    query = default_query.copy()
    query.update(parameters)
    query_str = "&".join([f"{key}={value}" for key, value in query.items()])
    return search_url + "/?" + query_str


def wait_till_loaded(browser, url, maxattempts=3):
    try:
        pageid = url.split("/detail:")[1]
    except IndexError:
        return 0
    attempt = 0
    while pageid not in browser.current_url:
        time.sleep(2**attempt)
        attempt += 1
        if attempt > maxattempts:
            return 0
    return 1


def pacing(n, ntot, t0, pace, total_sleep):
    """Control execution time to not exceed a given pace.

    Parameters
    ----------
    n: int
        Current iteration (0-based).
    ntot: int
        Total number of iterations.
    t0: float
        Start of the computation as returned by a monotonic clock
        (ie. output of time.monotonic()) in fractional seconds.
    pace: int
        Desired pacing in number of iterations per hour.
    total_sleep: float
        Total sleep time (in fractional seconds) since t0.

    Returns
    -------
    total_sleep: float
        Updated total_sleep in fractional seconds.
    """
    time_lapsed = time.monotonic() - t0
    done = n + 1
    togo = ntot - done
    LOGGER.debug(f"Averaging {done / time_lapsed * 3600:.0f} requests per hour")
    if togo > 0:
        time_togo = (done + togo) / pace * 3600 - time_lapsed
        spacing = max(0, time_togo / togo - (time_lapsed - total_sleep) / done)
        if spacing > 0:
            LOGGER.debug(f"Slowing down... wait {spacing:.1f} seconds...")
            time.sleep(spacing)
            total_sleep += spacing
    return total_sleep


def cleanup():
    for proc in psutil.process_iter():
        if (
            proc.name() in ["geckodriver", "Web Content", "firefox-bin"]
            and proc.username() == getpass.getuser()
        ):
            try:
                proc.kill()
            except psutil.NoSuchProcess:
                pass
