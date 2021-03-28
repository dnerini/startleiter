import logging
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
from sqlalchemy import create_engine
from sqlalchemy import Column, DateTime, Float, ForeignKey, Integer, Interval, String
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship, sessionmaker

from startleiter import config as CFG

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


class Station(Base):
    __tablename__ = "station"
    id = Column(Integer, primary_key=True)
    source_id = Column(Integer, ForeignKey("source.id"), nullable=False)
    name = Column(String(30))
    long_name = Column(String(30))
    stid = Column(Integer)
    country = Column(String(30))
    longitude = Column(Float)
    latitude = Column(Float)
    elevation = Column(Float)
    soundings = relationship("Sounding", backref="site", lazy=True)


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


class Sounding(Base):
    __tablename__ = "sounding"
    id = Column(Integer, primary_key=True)
    source_id = Column(Integer, ForeignKey("source.id"), nullable=False)
    station_id = Column(Integer, ForeignKey("station.id"), nullable=False)
    datetime = Column(DateTime(timezone=False))
    data = Column(String(30))
    show = Column(Float)
    lift = Column(Float)
    lftv = Column(Float)
    swet = Column(Float)
    kinx = Column(Float)
    ctot = Column(Float)
    vtot = Column(Float)
    ttot = Column(Float)
    cape = Column(Float)
    capv = Column(Float)
    cins = Column(Float)
    cinv = Column(Float)
    eqlv = Column(Float)
    eqtv = Column(Float)
    lfct = Column(Float)
    lfcv = Column(Float)
    brch = Column(Float)
    brcv = Column(Float)
    lclt = Column(Float)
    lclp = Column(Float)
    mlth = Column(Float)
    mlmr = Column(Float)
    thtk = Column(Float)
    pwat = Column(Float)


class Database:

    def __init__(self, source, site=None, station=None):
        db_uri = CFG["postgresql"]["uri"]
        self.engine = create_engine(db_uri, echo=False)

        Session = sessionmaker(self.engine)
        self.session = Session()

        Base.metadata.create_all(self.engine)

        self.source_id = self.insert_source(source)
        if site is not None:
            self.site_id = self.insert_site(site)
        if station is not None:
            self.station_id = self.insert_station(station)

    def insert_source(self, source):
        obj = self.session.query(Source).filter_by(name=source["name"]).first()
        if obj is None:
            obj = Source(**source)
            self.session.add(obj)
            self.session.commit()
        else:
            logger.debug(f"Source {source['name']} already exists.")
        return obj.id

    def insert_site(self, site):
        obj = self.session.query(Site).filter_by(name=site["name"]).first()
        if obj is None:
            site.update({"source_id": self.source_id})
            obj = Site(**site)
            self.session.add(obj)
            self.session.commit()
        else:
            logger.debug(f"Site {site['name']} already exists.")
        return obj.id

    def insert_station(self, station):
        obj = self.session.query(Station).filter_by(name=station["name"]).first()
        if obj is None:
            station.update({"source_id": self.source_id})
            obj = Station(**station)
            self.session.add(obj)
            self.session.commit()
        else:
            logger.debug(f"Station {station['name']} already exists.")
        return obj.id

    def insert_sounding(self, validtime, sounding, data_fn=None, overwrite=False):
        indices = reformat_uwyo(validtime, sounding, self.source_id, self.station_id)
        if data_fn:
            indices["data"] = data_fn.name
        obj = self.session.query(Sounding).filter_by(station_id=self.station_id)
        obj = obj.filter_by(datetime=indices["datetime"]).first()
        if obj is None or overwrite:
            sounding.update({"station_id": self.station_id})
            obj = Sounding(**indices)
            self.session.add(obj)
            self.session.commit()
        else:
            logger.debug(f"Sounding {indices['datetime']} already exists.")
        return obj.id

    def insert_soundings(self, soundings):
        data_fn = self.save_sounding_data(soundings)
        for validtime, sounding in soundings.items():
            self.insert_sounding(validtime, sounding["indices"], data_fn)

    def save_sounding_data(self, soundings):
        data = [sound["data"] for sound in soundings.values()]
        validtimes = list(soundings.keys())
        data = xr.concat(data, "validtime")
        data = data.assign_coords(validtime=validtimes)
        outdir = Path(CFG["netcdf"]["repo"])
        outdir.mkdir(exist_ok=True)
        fn = f"sounding-{self.source_id}-{self.station_id}-{validtimes[0]:%Y%m}.nc"
        fn = outdir / fn
        data.to_netcdf(fn)
        logger.info(f"Saved: {fn}")
        return fn

    def insert_flights(self, flights):
        flights = reformat_xcontest(flights, self.source_id, self.site_id)
        self.session.add_all([Flight(**flight) for flight in flights])
        self.session.commit()

    def query_last_flight(self):
        obj = self.session.query(Flight).filter_by(site_id=self.site_id).order_by(Flight.flno.desc()).first()
        flight_no = 0 if obj is None else obj.flno
        if flight_no > 0:
            logger.info(f"Starting querying from flight no. {flight_no}.")
        return flight_no

    def query_last_sounding(self, default_start=datetime(2005, 12, 31, 0)):
        obj = self.session.query(Sounding).filter_by(station_id=self.station_id).order_by(Sounding.datetime.desc()).first()
        start_date = default_start if obj is None else obj.datetime
        if start_date > default_start:
            logger.info(f"Starting querying from {start_date}.")
        return start_date

    def to_pandas(self):
        return pd.read_sql("flight", self.engine)


def reformat_uwyo(validtime, sounding, source_id, station_id):
    return {
        "source_id": source_id,
        "station_id": station_id,
        "datetime": validtime,
        "show": sounding.get("Showalter index", np.nan),
        "lift": sounding.get("Lifted index", np.nan),
        "lftv": sounding.get("LIFT computed using virtual temperature", np.nan),
        "swet": sounding.get("SWEAT index", np.nan),
        "kinx": sounding.get("K index", np.nan),
        "ctot": sounding.get("Cross totals index", np.nan),
        "vtot": sounding.get("Vertical totals index", np.nan),
        "ttot": sounding.get("Totals totals index", np.nan),
        "cape": sounding.get("Convective Available Potential Energy", np.nan),
        "capv": sounding.get("CAPE using virtual temperature", np.nan),
        "cins": sounding.get("Convective Inhibition", np.nan),
        "cinv": sounding.get("CINS using virtual temperature", np.nan),
        "eqlv": sounding.get("Equilibrum Level", np.nan),
        "eqtv": sounding.get("Equilibrum Level using virtual temperature", np.nan),
        "lfct": sounding.get("Level of Free Convection", np.nan),
        "lfcv": sounding.get("LFCT using virtual temperature", np.nan),
        "brch": sounding.get("Bulk Richardson Number", np.nan),
        "brcv": sounding.get("Bulk Richardson Number using CAPV", np.nan),
        "lclt": sounding.get("Temp [K] of the Lifted Condensation Level", np.nan),
        "lclp": sounding.get("Pres [hPa] of the Lifted Condensation Level", np.nan),
        "mlth": sounding.get("Mean mixed layer potential temperature", np.nan),
        "mlmr": sounding.get("Mean mixed layer mixing ratio", np.nan),
        "thtk": sounding.get("1000 hPa to 500 hPa thickness", np.nan),
        "pwat": sounding.get("Precipitable water [mm] for entire sounding", np.nan),
    }


def reformat_xcontest(flights, source_id, site_id):
    """Reformat raw xcontest data before appending to the database"""
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
