"""
End-to-end pipeline: ingest → process → QC → write XML.

Entry points
------------
run_station(config)
    Run the full pipeline for a single :class:`~ndbc_xml.config.StationConfig`.

run_all(sites)
    Run :func:`run_station` for every site in the supplied list (or all
    configured sites if *sites* is ``None``).

Typical usage::

    from ndbc_xml.pipeline import run_station
    from ndbc_xml.config import STATIONS

    run_station(STATIONS["CE02"])
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from .config import StationConfig, STATIONS
from .declination import get_declination
from .ingest import load_metbk, load_wavss
from .process import bin_observations, make_bin_edges, apply_qc
from .state import determine_bin_start, save_state
from .xml_writer import write_xml, xml_filename

log = logging.getLogger(__name__)

# Subtract ~1 h from the last raw timestamp to avoid partially-filled bins
# (matches the original MATLAB comment about data arriving at :05)
_BIN_END_MARGIN = pd.Timedelta(hours=1)


def run_station(config: StationConfig) -> Path | None:
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
    Path or None
        Path to the written XML file, or ``None`` if no new data were
        available to process.

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
    bin_start = determine_bin_start(
        state_file=config.state_file,
        earliest_data_time=earliest,
        deployment_indicator_file=config.deployment_indicator_file,
    )

    # --- 3. Bin end (last 10-min boundary safely within data) ---
    bin_end = (latest - _BIN_END_MARGIN).floor("10min")

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

    # --- 5. Bin edges, bin, QC ---
    edges = make_bin_edges(bin_start, bin_end)
    binned = bin_observations(
        metbk=metbk,
        wavss=wavss,
        bin_edges=edges,
        alpha_deg=alpha_deg,
        latitude=config.latitude,
        sensor_depth_m=config.sensor_depth_m,
    )
    qc_data = apply_qc(binned)

    if qc_data.empty:
        log.warning("%s: QC produced an empty DataFrame — skipping.", config.site)
        return None

    # --- 5. Write XML ---
    fname = xml_filename(config.ndbc_id)
    out_path = config.xml_out_dir / fname
    written = write_xml(qc_data, station_id=config.ndbc_id,
                        output_path=out_path)

    # --- 6. Persist state ---
    save_state(config.state_file, last_bin_end=bin_end, station=config.site)

    return written


def run_all(sites: list[str] | None = None) -> dict[str, Path | None]:
    """Run the pipeline for multiple stations.

    Parameters
    ----------
    sites : list of str or None
        Site codes to process (e.g. ``['CE02', 'CE04']``).
        Pass ``None`` to process all sites in
        :data:`~ndbc_xml.config.STATIONS`.

    Returns
    -------
    dict
        Mapping of site code → output XML path (or ``None`` if no
        new data were available for that site).
    """
    targets = sites if sites is not None else list(STATIONS.keys())
    results = {}
    for site in targets:
        if site not in STATIONS:
            log.error("Unknown site '%s' — skipping.", site)
            results[site] = None
            continue
        try:
            results[site] = run_station(STATIONS[site])
        except Exception:
            log.exception("Pipeline failed for %s", site)
            results[site] = None
    return results
