import logging
import os

import pandas as pd
from sqlalchemy import create_engine
from sqlalchemy import (
    Column,
    Date,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    Interval,
    String,
)
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship, sessionmaker


LOGGER = logging.getLogger(__name__)

Base = declarative_base()


class Source(Base):
    __tablename__ = "source"
    id = Column(Integer, primary_key=True)
    name = Column(String(30))
    base_url = Column(String(50))
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


class Prediction(Base):
    __tablename__ = "prediction"
    id = Column(Integer, primary_key=True)
    source_id = Column(Integer, ForeignKey("source.id"), nullable=False)
    site_id = Column(Integer, ForeignKey("site.id"), nullable=False)
    reftime = Column(Date)
    validtime = Column(Date)
    leadtime_days = Column(Integer)
    flying_probability = Column(Float)
    max_altitude_masl = Column(Float)
    max_distance_km = Column(Float)


class Database:
    def __init__(self):
        db_url = os.environ.get("DATABASE_URL")
        db_url = db_url.replace("postgres://", "postgresql://")
        self.engine = create_engine(db_url, echo=False)
        Session = sessionmaker(self.engine)
        self.session = Session()
        Base.metadata.create_all(self.engine)

    def add(
        self, model: Base, entry: dict, preprocess_fn=None, preprocess_kwargs=None
    ) -> int:
        if preprocess_fn is not None:
            preprocess_kwargs = preprocess_kwargs or {}
            entry = preprocess_fn(entry, **preprocess_kwargs)
        obj = self.session.query(model).filter_by(name=entry["name"]).first()
        if obj is None:
            obj = model(**entry)
            self.session.add(obj)
            self.session.commit()
        else:
            LOGGER.debug(f"{model.__tablename__} {entry['name']} already exists.")
        return obj.id

    def add_all(
        self,
        model: Base,
        entries: list[dict],
        preprocess_fn=None,
        preprocess_kwargs=None,
    ) -> None:
        if preprocess_fn is not None:
            preprocess_kwargs = preprocess_kwargs or {}
            entries = [preprocess_fn(entry, **preprocess_kwargs) for entry in entries]
        self.session.add_all([model(**entry) for entry in entries if entry])
        self.session.commit()
