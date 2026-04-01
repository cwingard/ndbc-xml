"""
Station configuration for NDBC XML generation.

Each StationConfig holds all site-specific parameters. QC_BOUNDS
defines the global range checks applied identically to all stations.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Tuple


@dataclass
class StationConfig:
    """Per-station parameters for NDBC XML generation.

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
        Nominal depth of the CTD/temperature sensor in metres
        (positive down). Used for pressure calculation. Default 1.5 m.
    deployment_indicator_file : Optional[Path]
        Text file (single float) flagging whether this is a fresh
        deployment (0) or a continuation (1). When ``None`` the
        pipeline always uses state_file for continuity tracking.
    """

    site: str
    ndbc_id: str
    latitude: float
    longitude: float
    metbk_dir: Path
    wavss_dir: Path
    xml_out_dir: Path
    state_file: Path
    sensor_depth_m: float = 1.5
    deployment_indicator_file: Optional[Path] = None


# ---------------------------------------------------------------------------
# QC range bounds — (min_valid, max_valid); values outside → NaN → -9999
# ---------------------------------------------------------------------------

QC_BOUNDS: dict[str, Tuple[float, float]] = {
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
# Pre-configured station instances
# ---------------------------------------------------------------------------

_BASE = Path("/mnt/ooinas/data/cgsn/proc")
_SCRIPTS = Path("/home/craig/cgoms/scripts")
_XML_BASE = Path("/home/craig/cgoms/xml")

STATIONS: dict[str, StationConfig] = {
    "CE02": StationConfig(
        site="CE02",
        ndbc_id="46097",
        latitude=44.639,
        longitude=-124.095,
        metbk_dir=_BASE / "ce02shsm/D00020/buoy/metbk",
        wavss_dir=_BASE / "ce02shsm/D00020/buoy/wavss",
        xml_out_dir=_XML_BASE / "CE02",
        state_file=_SCRIPTS / "CE02_position.json",
        deployment_indicator_file=_SCRIPTS / "CE02_deployment_indicator.txt",
    ),
    "CE04": StationConfig(
        site="CE04",
        ndbc_id="46098",
        latitude=44.369,
        longitude=-124.954,
        metbk_dir=_BASE / "ce04ossm/D00019/buoy/metbk",
        wavss_dir=_BASE / "ce04ossm/D00019/buoy/wavss",
        xml_out_dir=_XML_BASE / "CE04",
        state_file=_SCRIPTS / "CE04_position.json",
        deployment_indicator_file=_SCRIPTS / "CE04_deployment_indicator.txt",
    ),
    "CE07": StationConfig(
        site="CE07",
        ndbc_id="46099",
        latitude=44.369,
        longitude=-124.555,
        metbk_dir=_BASE / "ce07shsm/D00020/buoy/metbk",
        wavss_dir=_BASE / "ce07shsm/D00020/buoy/wavss",
        xml_out_dir=_XML_BASE / "CE07",
        state_file=_SCRIPTS / "CE07_position.json",
        deployment_indicator_file=_SCRIPTS / "CE07_deployment_indicator.txt",
    ),
    "CE09": StationConfig(
        site="CE09",
        ndbc_id="46100",
        latitude=46.859,
        longitude=-124.973,
        metbk_dir=_BASE / "ce09ossm/D00020/buoy/metbk",
        wavss_dir=_BASE / "ce09ossm/D00020/buoy/wavss",
        xml_out_dir=_XML_BASE / "CE09",
        state_file=_SCRIPTS / "CE09_position.json",
        deployment_indicator_file=_SCRIPTS / "CE09_deployment_indicator.txt",
    ),
}
