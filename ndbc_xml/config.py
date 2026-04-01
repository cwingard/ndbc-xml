#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Station configuration for NDBC XML generation.

Each StationConfig holds all site-specific parameters. QC_BOUNDS
defines the global range checks applied identically to all stations.
"""
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple

@dataclass
class StationConfig:
    """
    Per-station parameters for NDBC XML generation.

    Parameters
    ----------
    site : str
        OOI site code (e.g. ``'CE02'``).
    ndbc_id : str
        NDBC station identifier written into the XML (e.g. ``'46097'``).
    latitude : float
        Station latitude in decimal degrees north. Used for GSW
        salinity calculation and IGRF declination.
    longitude : float
        Station longitude in decimal degrees east (negative west).
        Used for IGRF magnetic declination calculation.
    metbk_dir : Path
        Directory containing ``*.json`` METBK data files.
    wavss_dir : Path
        Directory containing ``*.wavss.json`` WAVSS data files.
    xml_out_dir : Path
        Directory where output XML files are written.
    state_file : Path
        JSON file used to persist the last-processed bin timestamp
        across runs. Created automatically on first run.
    sensor_depth_m : float
        Nominal depth of the CTD/temperature sensor in meters
        (positive down). Used for pressure calculation. Default 1.25 m.
    """
    site: str
    ndbc_id: str
    latitude: float
    longitude: float
    metbk_dir: Path
    wavss_dir: Path
    xml_out_dir: Path
    state_file: Path
    sensor_depth_m: float = 1.25

# ---------------------------------------------------------------------------
# QC range bounds — (min_valid, max_valid); values outside → NaN → -9999
# ---------------------------------------------------------------------------
QC_BOUNDS: Dict[str, Tuple[float, float]] = {
    "wind_dir":    (0.0,   360.0),
    "wind_speed":  (0.0,    60.0),
    "baro":        (860.0, 1050.0),
    "air_temp":    (-10.0,  30.0),
    "rel_humidity":(0.0,   100.0),
    "shortwave":   (0.0,  2000.0),
    "longwave":    (0.0,  2000.0),
    "sst":         (0.0,    25.0),
    "salinity":    (10.0,   40.0),
    "sig_wave_hgt":(0.0,    30.0),
    "max_wave_hgt":(0.0,    30.0),
    "peak_period": (0.0,    30.0),
    "avg_period":  (0.0,    30.0),
    "wave_dir":    (0.0,   360.0),
}

# ---------------------------------------------------------------------------
# Site metadata: NDBC ID and mooring coordinates
# ---------------------------------------------------------------------------
SITES: Dict[str, dict] = {
    "CE02": {"mooring": "ce02shsm", "ndbc_id": "46097", "latitude": 44.639, "longitude": -124.304},
    "CE04": {"mooring": "ce04ossm", "ndbc_id": "46098", "latitude": 44.381, "longitude": -124.956},
    "CE07": {"mooring": "ce07shsm", "ndbc_id": "46099", "latitude": 46.986, "longitude": -124.566},
    "CE09": {"mooring": "ce09ossm", "ndbc_id": "46100", "latitude": 46.851, "longitude": -124.972},
}
