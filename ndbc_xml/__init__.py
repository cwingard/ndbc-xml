"""
ndbc_xml — Generate NDBC XML submission files from OOI buoy data.

Public API
----------
run_station(config)
    Full pipeline for a single station.
run_all(sites)
    Run all (or a subset of) configured stations.
STATIONS
    Dict of pre-configured :class:`StationConfig` instances keyed by
    site code (``'CE02'``, ``'CE04'``, ``'CE07'``, ``'CE09'``).
get_declination(latitude, longitude, date)
    Compute IGRF magnetic declination at a point and date.
"""

from .config import STATIONS, StationConfig, QC_BOUNDS
from .declination import get_declination
from .pipeline import run_station, run_all

__all__ = [
    "STATIONS",
    "StationConfig",
    "QC_BOUNDS",
    "get_declination",
    "run_station",
    "run_all",
]
