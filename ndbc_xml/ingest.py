#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Load raw METBK and WAVSS JSON files into pandas DataFrames.

Each JSON file produced by the OOI system contains arrays of values
keyed by variable name. Files are sorted by name before concatenation
so that time ordering is preserved.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

# Columns expected in METBK JSON files
_METBK_COLS = [
    "time",
    "air_temperature",
    "barometric_pressure",
    "eastward_wind_velocity",
    "northward_wind_velocity",
    "longwave_irradiance",
    "relative_humidity",
    "sea_surface_conductivity",
    "sea_surface_temperature",
    "shortwave_irradiance",
    "precipitation_level",
]

# Columns expected in WAVSS JSON files
_WAVSS_COLS = [
    "time",
    "peak_period",
    "average_wave_period",
    "significant_wave_height",
    "maximum_wave_height",
    "mean_wave_direction",
]


def _load_json_dir(
    directory: Path,
    glob: str,
    columns: list[str],
) -> pd.DataFrame:
    """Load all matching JSON files from a directory into a DataFrame.

    Parameters
    ----------
    directory : Path
        Directory to search for files.
    glob : str
        Glob pattern (e.g. ``'*.json'`` or ``'*.wavss.json'``).
    columns : list of str
        Variable names to extract. Files missing a column receive
        NaN for that column.

    Returns
    -------
    pd.DataFrame
        Concatenated data sorted by ``time`` (Unix epoch seconds),
        with duplicate timestamps removed (first occurrence kept).

    Raises
    ------
    FileNotFoundError
        If no files matching *glob* exist in *directory*.
    """
    files = sorted(directory.glob(glob))
    if not files:
        raise FileNotFoundError(
            f"No files matching '{glob}' in {directory}"
        )

    frames = []
    for path in files:
        try:
            with path.open() as fh:
                raw = json.load(fh)
        except (json.JSONDecodeError, OSError) as exc:
            log.warning("Skipping %s: %s", path.name, exc)
            continue

        row = {col: np.asarray(raw.get(col, []), dtype=float)
               for col in columns}

        # All arrays in a file should be the same length; drop the file
        # if lengths are inconsistent.
        lengths = {k: len(v) for k, v in row.items() if len(v) > 0}
        if not lengths:
            log.warning("Skipping empty file %s", path.name)
            continue

        n = max(lengths.values())
        for col in columns:
            if col not in lengths:
                row[col] = np.full(n, np.nan)
            elif lengths[col] != n:
                log.warning(
                    "%s: column '%s' length %d != expected %d — filling NaN",
                    path.name, col, lengths[col], n,
                )
                row[col] = np.full(n, np.nan)

        frames.append(pd.DataFrame(row))

    if not frames:
        raise ValueError(f"All files in {directory} failed to load.")

    df = pd.concat(frames, ignore_index=True)
    df.sort_values("time", inplace=True)
    df.drop_duplicates(subset="time", keep="first", inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df


def load_metbk(directory: Path) -> pd.DataFrame:
    """Load all METBK JSON files from *directory*.

    Parameters
    ----------
    directory : Path
        Directory containing ``*.json`` METBK files.

    Returns
    -------
    pd.DataFrame
        Columns: ``time`` (Unix epoch s) plus all METBK variables.
        Sorted by time, duplicates removed.
    """
    return _load_json_dir(directory, "*.json", _METBK_COLS)


def load_wavss(directory: Path) -> pd.DataFrame:
    """Load all WAVSS JSON files from *directory*.

    Parameters
    ----------
    directory : Path
        Directory containing ``*.wavss.json`` WAVSS files.

    Returns
    -------
    pd.DataFrame
        Columns: ``time`` (Unix epoch s) plus all WAVSS variables.
        Timestamps are shifted back by 10 minutes to center them in
        the middle of the 20-minute WAVSS measurement cycle (matching
        the original MATLAB behavior).
        Sorted by time, duplicates removed.
    """
    df = _load_json_dir(directory, "*.wavss.json", _WAVSS_COLS)
    # Shift timestamps to center of the 20-min cycle
    df["time"] = df["time"] - 600.0
    return df
