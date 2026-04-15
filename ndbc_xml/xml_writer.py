#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Write NDBC-format XML files from processed, QC'd observation data.

The output format follows the NDBC BUFR/XML submission specification:
  - A WMO bulletin header line (``SXML99 KWBC DDHHMM``) is prepended
    before the XML declaration.  This is intentional and required.
  - Each 10-minute observation is wrapped in a ``<message>`` element
    inside a single file containing a bare sequence of messages (no
    root element wrapper).
  - Missing values are written as ``-9999``.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

MISSING = -9999
_FLOAT_FMT = "{:.6f}"

# Fixed FM-64 BUOY metadata tags — identical for all OOI stations
_FM64_TAGS = (
    "<fm64iii>830</fm64iii>",
    "<fm64xx>99</fm64xx>",
    "<fm64k1>7</fm64k1>",
    "<fm64k2>1</fm64k2>",
)


def _fmt(value: float) -> str:
    """Format a scalar observation, substituting -9999 for NaN/inf.

    Parameters
    ----------
    value : float
        Observation value.

    Returns
    -------
    str
        Formatted string, or ``'-9999'`` if value is not finite.
    """
    if np.isfinite(value):
        return _FLOAT_FMT.format(value)
    return str(MISSING)


def _tag(name: str, value: float) -> str:
    """Render a single ``<name>value</name>`` XML element.

    Parameters
    ----------
    name : str
        XML tag name.
    value : float
        Observation value (NaN/inf → -9999).

    Returns
    -------
    str
        Formatted XML element string.
    """
    return f"<{name}>{_fmt(value)}</{name}>"


def build_message(
    station_id: str,
    timestamp: pd.Timestamp,
    wind_dir: float,
    wind_speed: float,
    baro: float,
    air_temp: float,
    rel_humidity: float,
    shortwave: float,
    longwave: float,
    sig_wave_hgt: float,
    peak_period: float,
    wave_dir: float,
    sst: float,
    salinity: float,
    sensor_depth_m: float,
) -> str:
    """Build a single NDBC XML ``<message>`` block.

    Parameters
    ----------
    station_id : str
        NDBC station identifier (e.g. ``'46097'``).
    timestamp : pd.Timestamp
        Observation time (UTC), formatted as ``MM/DD/YYYY HH:MM:SS``.
    wind_dir : float
        Wind direction, degrees true (``<wdir1>``).
    wind_speed : float
        Wind speed, m/s (``<wspd1>``).
    baro : float
        Barometric pressure, hPa (``<baro1>``).
    air_temp : float
        Air temperature, °C (``<atmp1>``).
    rel_humidity : float
        Relative humidity, % (``<rrh>``).
    shortwave : float
        Shortwave irradiance, W/m² (``<srad1>``).
    longwave : float
        Longwave irradiance, W/m² (``<lwrad>``).
    sig_wave_hgt : float
        Significant wave height, m (``<wvhgt>``).
    peak_period : float
        Dominant wave period, s (``<dompd>``).
    wave_dir : float
        Mean wave direction, degrees (``<mwdir>``).
    sst : float
        Sea surface temperature, °C (``<tp001>``, ``<wtmp1>``).
    salinity : float
        Sea surface practical salinity, PSU (``<sp001>``).
    sensor_depth_m : float
        CTD sensor depth in meters (``<dp001>``).

    Returns
    -------
    str
        Complete ``<message>...</message>`` block as a string.
    """
    date_str = timestamp.strftime("%m/%d/%Y %H:%M:%S")
    lines = [
        "<message>",
        f"<station>{station_id}</station>",
        f"<date>{date_str}</date>",
        f"<missing>{MISSING}</missing>",
        "<roundtime>no</roundtime>",
        "<met>",
        _tag("wdir1",  wind_dir),
        _tag("wspd1",  wind_speed),
        _tag("baro1",  baro),
        _tag("atmp1",  air_temp),
        _tag("rrh",    rel_humidity),
        _tag("srad1",  shortwave),
        _tag("lwrad",  longwave),
        _tag("wvhgt",  sig_wave_hgt),
        _tag("dompd",  peak_period),
        _tag("mwdir",  wave_dir),
        _tag("tp001",  sst),
        *_FM64_TAGS,
        f"<dp001>{sensor_depth_m}</dp001>",
        _tag("wtmp1",  sst),
        _tag("sp001",  salinity),
        "</met>",
        "</message>",
    ]
    return "\n".join(lines)


def _bulletin_header(last_time: pd.Timestamp) -> str:
    """Generate the WMO bulletin header line.

    Format: ``SXML99 KWBC DDHHMM`` where DD/HH/MM are taken from
    the most recent observation timestamp.

    Parameters
    ----------
    last_time : pd.Timestamp
        Timestamp of the last observation in the file (UTC).

    Returns
    -------
    str
        Header line (no trailing newline).
    """
    return f"SXML99 KWBC {last_time.strftime('%d%H%M')}"


def write_xml(
    df: pd.DataFrame,
    station_id: str,
    output_path: Path,
    sensor_depth_m: float,
) -> Path:
    """Write a complete NDBC XML submission file.

    The file begins with a WMO bulletin header line followed by the
    XML declaration, then a sequence of ``<message>`` blocks — one per
    row in *df*.

    Parameters
    ----------
    df : pd.DataFrame
        QC'd, binned observations from
        :func:`~ndbc_xml.process.apply_qc`.  Required columns:
        ``time``, ``wind_dir``, ``wind_speed``, ``baro``,
        ``air_temp``, ``rel_humidity``, ``shortwave``, ``longwave``,
        ``sig_wave_hgt``, ``peak_period``, ``wave_dir``, ``sst``,
        ``salinity``.
    station_id : str
        NDBC station identifier written into each ``<station>`` tag.
    output_path : Path
        Destination file path. Parent directory must exist.
    sensor_depth_m : float
        CTD sensor depth in meters, written into each ``<dp001>`` tag.

    Returns
    -------
    Path
        The path to the written file (*output_path*).

    Raises
    ------
    ValueError
        If *df* is empty.
    """
    if df.empty:
        raise ValueError("Cannot write XML from empty DataFrame.")

    last_time = df["time"].iloc[-1]
    header = _bulletin_header(last_time)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="ISO-8859-1") as fh:
        fh.write(header)
        fh.write('\n<?xml version="1.0" encoding="ISO-8859-1"?>')

        for row in df.to_dict("records"):
            fh.write("\n")
            fh.write(
                build_message(
                    station_id=station_id,
                    timestamp=row["time"],
                    wind_dir=row["wind_dir"],
                    wind_speed=row["wind_speed"],
                    baro=row["baro"],
                    air_temp=row["air_temp"],
                    rel_humidity=row["rel_humidity"],
                    shortwave=row["shortwave"],
                    longwave=row["longwave"],
                    sig_wave_hgt=row["sig_wave_hgt"],
                    peak_period=row["peak_period"],
                    wave_dir=row["wave_dir"],
                    sst=row["sst"],
                    salinity=row["salinity"],
                    sensor_depth_m=sensor_depth_m,
                )
            )

    log.info("Wrote %d records to %s", len(df), output_path)
    return output_path


def xml_filename(station_id: str) -> str:
    """Generate the NDBC XML filename for the current UTC time.

    Format: ``HH-DD-Mon-YYYY-{station_id}.xml``
    (e.g. ``14-30-Mar-2026-46097.xml``).

    Parameters
    ----------
    station_id : str
        NDBC station identifier.

    Returns
    -------
    str
        Filename string.
    """
    now = pd.Timestamp.now("UTC")
    return now.strftime(f"%H-%d-%b-%Y-{station_id}.xml")


def daily_xml_filename(station_id: str, date: pd.Timestamp, hour: int) -> str:
    """Generate a date-stamped NDBC XML filename for daily archive files.

    Format: ``HH-DD-Mon-YYYY-{station_id}.xml``
    (e.g. ``23-01-Apr-2026-46097.xml``).

    Past days use hour ``23``; the current UTC day uses the actual run hour.

    Parameters
    ----------
    station_id : str
        NDBC station identifier.
    date : pd.Timestamp
        UTC calendar date of the observations in the file.
    hour : int
        Hour to embed in the filename (0–23).

    Returns
    -------
    str
        Filename string.
    """
    return date.strftime(f"{hour:02d}-%d-%b-%Y-{station_id}.xml")


def write_xml_daily(
    df: pd.DataFrame,
    station_id: str,
    output_dir: Path,
    sensor_depth_m: float,
) -> list[Path]:
    """Write one NDBC XML file per UTC calendar day.

    Used when reprocessing or running for the first time, where *df*
    may span many days. Each output file contains only the observations
    for that UTC day and is named with :func:`daily_xml_filename`.

    Past days get hour ``23`` in their filename; the current UTC day gets
    the actual UTC hour of the run.

    Parameters
    ----------
    df : pd.DataFrame
        QC'd, binned observations (same schema as :func:`write_xml`).
    station_id : str
        NDBC station identifier written into each ``<station>`` tag.
    output_dir : Path
        Directory where daily files are written. Created if absent.
    sensor_depth_m : float
        CTD sensor depth in meters, written into each ``<dp001>`` tag.

    Returns
    -------
    list[Path]
        Paths of all files written, one per UTC calendar day.

    Raises
    ------
    ValueError
        If *df* is empty.
    """
    if df.empty:
        raise ValueError("Cannot write XML from empty DataFrame.")

    output_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []

    now = pd.Timestamp.now("UTC")
    today = now.normalize()
    run_hour = now.hour

    for date, day_df in df.groupby(df["time"].dt.normalize()):
        hour = run_hour if date == today else 23
        out_path = output_dir / daily_xml_filename(station_id, date, hour)
        write_xml(day_df.reset_index(drop=True), station_id=station_id,
                  output_path=out_path, sensor_depth_m=sensor_depth_m)
        written.append(out_path)

    log.info("Wrote %d daily XML file(s) to %s", len(written), output_dir)
    return written
