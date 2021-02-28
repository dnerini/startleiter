import logging
from datetime import datetime, timedelta

import pandas as pd

from sqlalchemy import create_engine
from sqlalchemy import Column, DateTime, Float, ForeignKey, Integer, Interval, String
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship, sessionmaker

from startleiter import config

logger = logging.getLogger(__name__)

Base = declarative_base()


class Source(Base):
    __tablename__ = "source"
    id = Column(Integer, primary_key=True)
    name = Column(String(30))
    base_url = Column(String(30))
    sites = relationship("Site", backref="source", lazy=True)
    flights = relationship("Flight", backref="source", lazy=True)


class Site(Base):
    __tablename__ = "site"
    id = Column(Integer, primary_key=True)
    source_id = Column(Integer, ForeignKey("source.id"), nullable=False)
    name = Column(String(30))
    country = Column(String(30))
    longitude = Column(Float)
    latitude = Column(Float)
    radius = Column(Integer)
    flights = relationship("Flight", backref="site", lazy=True)


class Flight(Base):
    __tablename__ = "flight"
    id = Column(Integer, primary_key=True)
    source_id = Column(Integer, ForeignKey("source.id"), nullable=False)
    site_id = Column(Integer, ForeignKey("site.id"), nullable=False)
    flid = Column(Integer)
    flno = Column(Integer)
    datetime = Column(DateTime(timezone=True))
    pilot = Column(String(30))
    route = Column(String(30))
    length_km = Column(Float)
    points = Column(Float)
    glider = Column(String(30))
    glider_cat = Column(String(30))
    airtime = Column(Interval)
    max_altitude_m = Column(Integer)
    max_alt_gain_m = Column(Integer)
    max_climb_ms = Column(Float)
    max_sink_ms = Column(Float)
    tracklog_length_km = Column(Float)
    free_distance_1_km = Column(Float)
    free_distance_2_km = Column(Float)

class Database:

    def __init__(self, source, site):
        db_uri = config["postgresql"]["uri"]
        self.engine = create_engine(db_uri, echo=False)

        Session = sessionmaker(self.engine)
        self.session = Session()

        Base.metadata.create_all(self.engine)

        self.source_id = self.insert_source(source)
        self.site_id = self.insert_site(site)

    def insert_source(self, source):
        obj = self.session.query(Source).filter_by(name=source["name"]).first()
        if obj is None:
            obj = Source(**source)
            self.session.add(obj)
            self.session.commit()
        else:
            logger.info(f"Source {source['name']} already exists.")
        return obj.id

    def insert_site(self, site):
        obj = self.session.query(Site).filter_by(name=site["name"]).first()
        if obj is None:
            site.update({"source_id": self.source_id})
            obj = Site(**site)
            self.session.add(obj)
            self.session.commit()
        else:
            logger.info(f"Site {site['name']} already exists.")
        return obj.id

    def print_sites(self):
        print(self.session.query(Site.__table__).all())

    def insert_flights(self, flights):
        flights = parse_flights(flights, self.source_id, self.site_id)
        self.session.add_all([Flight(**flight) for flight in flights])
        self.session.commit()

    def query_last_flight(self):
        obj = self.session.query(Flight).filter_by(site_id=self.site_id).order_by(Flight.flno.desc()).first()
        flight_no = 0 if obj is None else obj.flno
        if flight_no > 0:
            logger.info(f"Starting querying from flight no. {flight_no}.")
        return flight_no

    def to_pandas(self):
        return pd.read_sql("flight", self.engine)


def parse_flights(flights, source_id, site_id):
    out = []
    for flight in flights:
        try:
            datetime_str = f"{flight[2]}+00:00" if flight[2][-3:] == "UTC" else flight[2]
            airtime = flight[11] if flight[11] is None else flight[11].split(":")[:2]
            altitude = flight[12] if flight[12] is None else flight[12].replace(" m", "")
            alt_gain = flight[13] if flight[13] is None else flight[13].replace(" m", "")
            max_climb = flight[14] if flight[14] is None else flight[14].replace(" m/s", "")
            max_sink = flight[15] if flight[15] is None else flight[15].replace(" m/s", "")
            tracklog_length = flight[16] if flight[16] is None else flight[16].replace(" km", "")
            free_distance = flight[17] if flight[17] is None else flight[17].split("/")
            out.append({
                "source_id": source_id,
                "site_id": site_id,
                "flid": flight[0],
                "flno": flight[1],
                "datetime": datetime.strptime(datetime_str, "%d.%m.%y %H:%MUTC%z"),
                "pilot": flight[3][2:],
                "route": flight[10],
                "length_km": float(flight[6].replace(" km", "")),
                "points": float(flight[7].replace(" p.", "")),
                "glider": flight[9] if flight[9] else None,
                "glider_cat": flight[8],
                "airtime": airtime if airtime is None else timedelta(hours=int(airtime[0]), minutes=int(airtime[1])),
                "max_altitude_m": altitude if altitude is None else int(altitude),
                "max_alt_gain_m": alt_gain if alt_gain is None else int(alt_gain),
                "max_climb_ms": max_climb if max_climb is None else float(max_climb),
                "max_sink_ms": max_sink if max_sink is None else float(max_sink),
                "tracklog_length_km": tracklog_length if tracklog_length is None else float(tracklog_length),
                "free_distance_1_km": free_distance if free_distance is None else float(free_distance[0].replace(" km", "")),
                "free_distance_2_km": free_distance if free_distance is None else float(free_distance[1].replace(" km", "")),
                })
        except ValueError:
            logger.error(f"Could not parse {flight}")
    return out