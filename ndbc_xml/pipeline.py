#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
End-to-end pipeline: ingest → process → QC → write XML.

Entry points
------------
run_station(config)
    Run the full pipeline for a single :class:`~ndbc_xml.config.StationConfig`.

Typical usage::

    from ndbc_xml.pipeline import run_station
"""
from __future__ import annotations

import logging
from pathlib import Path

import gsw
import pandas as pd

from .config import StationConfig
from .declination import get_declination
from .ingest import load_metbk, load_wavss
from .process import bin_observations, make_bin_edges, apply_qc
from .state import bin_start_from_state, save_state
from .xml_writer import write_xml, write_xml_daily, xml_filename

log = logging.getLogger(__name__)


# Spans longer than this are written as one file per UTC day instead of a
# single file.  Reprocessing a full deployment produces months of data, so
# daily splitting keeps file sizes manageable and mirrors NDBC expectations.
_DAILY_SPLIT_THRESHOLD = pd.Timedelta(days=1)


def run_station(config: StationConfig) -> list[Path] | None:
    """Run the full NDBC XML pipeline for one station.

    Steps:
      1. Load METBK and WAVSS JSON files.
      2. Determine bin start (from state file or deployment indicator).
      3. Determine bin end (last raw timestamp minus margin).
      4. Compute IGRF magnetic declination at the mid-deployment date.
      5. Bin, process, and QC the observations.
      6. Write the XML file.
      7. Persist the new state.

    Parameters
    ----------
    config : StationConfig
        Station-specific configuration.

    Returns
    -------
    list[Path] or None
        Paths of the written XML file(s), or ``None`` if no new data
        were available to process.  Spans longer than one day produce
        one file per UTC calendar day; shorter spans produce a single
        file named with the current UTC time.

    Raises
    ------
    FileNotFoundError
        If the METBK or WAVSS data directories contain no matching
        files.
    """
    log.info("=== %s (NDBC %s) ===", config.site, config.ndbc_id)

    # --- 1. Ingest ---
    metbk = load_metbk(config.metbk_dir)
    wavss = load_wavss(config.wavss_dir)

    metbk_ts = pd.to_datetime(metbk["time"].values, unit="s", utc=True)
    earliest = metbk_ts.min()
    latest = metbk_ts.max()

    # --- 2. Bin start ---
    bin_start = bin_start_from_state(config.state_file, earliest)

    # --- 3. Bin end ---
    # Floor latest to the hour (matching the original MATLAB: subtract 1 h
    # then truncate to the hour, then set minutes to :55).  With the +5 min
    # METBK timestamp shift and bins aligned to :00 boundaries, flooring
    # to the hour gives a last xml timestamp at (latest_hour - 1):50.
    bin_end = latest.floor("h")

    if bin_end <= bin_start:
        log.info(
            "%s: no new bins to process (bin_start=%s, bin_end=%s)",
            config.site, bin_start, bin_end,
        )
        return None

    log.info("%s: processing %s → %s", config.site, bin_start, bin_end)

    # --- 4. Compute magnetic declination (IGRF) at mid-deployment date ---
    mid_date = (bin_start + (bin_end - bin_start) / 2).to_pydatetime()
    alpha_deg = get_declination(
        latitude=config.latitude,
        longitude=config.longitude,
        date=mid_date,
    )
    log.info("%s: IGRF declination = %.3f", config.site, alpha_deg)

    # --- 5. Bin, QC ---
    # Convert sensor depth to pressure: GSW uses z (negative-down), so
    # depth_m positive → z = -depth_m.
    pressure_dbar = float(gsw.p_from_z(-config.sensor_depth_m, config.latitude))
    edges = make_bin_edges(bin_start, bin_end)
    binned = bin_observations(
        metbk=metbk,
        wavss=wavss,
        bin_edges=edges,
        alpha_deg=alpha_deg,
        pressure_dbar=pressure_dbar,
    )
    qc_data = apply_qc(binned)

    if qc_data.empty:
        log.warning("%s: QC produced an empty DataFrame — skipping.", config.site)
        return None

    # --- 6. Write XML ---
    # Multi-day spans (reprocess / first run) → one file per UTC day.
    # Short incremental spans → single timestamped file.
    if (bin_end - bin_start) > _DAILY_SPLIT_THRESHOLD:
        written = write_xml_daily(
            qc_data,
            station_id=config.ndbc_id,
            output_dir=config.xml_out_dir,
            sensor_depth_m=config.sensor_depth_m,
        )
    else:
        fname = xml_filename(config.ndbc_id)
        out_path = config.xml_out_dir / fname
        written = [write_xml(qc_data, station_id=config.ndbc_id,
                             output_path=out_path,
                             sensor_depth_m=config.sensor_depth_m)]

    # --- 7. Persist state ---
    save_state(config.state_file, last_bin_end=bin_end, station=config.site)

    return written
