import logging
import sys
import time
from functools import wraps

logger = logging.getLogger(__name__)


def try_wait(maxattempts=6):
    """Try and wait decorator."""

    def _try_wait(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            for attempt in range(maxattempts):
                try:
                    logger.debug(
                        f"@try_wait: {func.__name__} Trying ... ({attempt + 1})"
                    )
                    result = func(*args, **kwargs)
                except Exception as err:
                    logger.error(f"@try_wait: {func.__name__} Failed: {err}")
                    if attempt == (maxattempts - 1) or "pytest" in sys.modules:
                        logger.critical(
                            f"@try_wait: {func.__name__} Failed after "
                            f"{attempt + 1} attempts. Stopping."
                        )
                        raise err
                    wait = 2 ** attempt
                    logger.info(
                        f"@try_wait: {func.__name__} Waiting {wait} seconds ..."
                    )
                    time.sleep(wait)
                else:
                    logger.debug(f"@try_wait: {func.__name__} Success!")
                    break
            return result

        return wrapper

    return _try_wait