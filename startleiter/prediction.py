import logging
import requests
from datetime import datetime

from startleiter import config as CFG
from startleiter.database import Site, Source, Prediction
from startleiter.database import Database

LOGGER = logging.getLogger(__name__)


def preprocess_prediction(prediction):
    prediction.update(
        {
            "validtime": datetime.strptime(prediction["validtime"], "%Y-%m-%d").date(),
        }
    )
    prediction.pop("site")
    return prediction


def main(sites):
    """"""
    db = Database()

    source = CFG["sources"]["startleiter"]
    source_id = db.add(Source, source)

    reftime = datetime.today().date()

    n = 0
    for day in range(5):
        predictions = []
        LOGGER.info(f"Day: {day}")
        for site_name, site in sites:
            LOGGER.info(f"Site: {site_name}")
            site.update({"source_id": source_id})
            site_id = db.add(Site, site)
            response = requests.get(
                f"https://startleiter.herokuapp.com/site?site={site_name}&time={reftime:%Y-%m-%d}&leadtime_days={day}"
            )
            try:
                response.raise_for_status()
            except requests.exceptions.HTTPError as e:
                LOGGER.error(f"Request failed: {e}")
                continue
            else:
                prediction = response.json()
                prediction.update(
                    {
                        "source_id": source_id,
                        "site_id": site_id,
                        "reftime": reftime,
                        "leadtime_days": day,
                    }
                )
                predictions.append(prediction)
        if predictions:
            db.add_all(
                Prediction,
                predictions,
                preprocess_fn=preprocess_prediction,
            )
            n += len(predictions)
    LOGGER.info(f"Successfully added {n} predictions.")


if __name__ == "__main__":
    logging.basicConfig(
        # format="%(asctime)s,%(msecs)d %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s",
        format="%(levelname)-4s [%(filename)s:%(lineno)d] %(message)s",
        datefmt="%Y-%m-%d:%H:%M:%S",
        level=logging.INFO,
    )
    sites = list(CFG["sites"].items())
    main(sites)
