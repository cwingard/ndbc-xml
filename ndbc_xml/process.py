#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Scientific processing for NDBC XML generation.

Transforms raw METBK/WAVSS arrays into 10-minute binned, QC-checked
variables ready for XML output.

Processing steps:
  1. Rotate wind (u, v) components by buoy heading alpha.
  2. Compute wind speed and meteorological wind direction.
  3. Convert wave direction to unit vectors, rotate by alpha,
     then recover meteorological wave direction.
  4. Compute practical salinity from conductivity via GSW.
  5. Compute rain rate from cumulative precipitation level.
  6. Bin all variables into 10-minute averages using pandas resample.
  7. Compute binned wind and wave directions from averaged vectors.
  8. Apply global range QC — out-of-range values become NaN.
"""
import numpy as np
import pandas as pd
import gsw

from .config import QC_BOUNDS

# ---------------------------------------------------------------------------
# Wind processing
# ---------------------------------------------------------------------------
EPOCH = pd.Timestamp("1970-01-01", tz="UTC")


def epoch_to_datetime(time_s: np.ndarray) -> pd.DatetimeIndex:
    """Convert Unix epoch seconds to UTC DatetimeIndex.

    Parameters
    ----------
    time_s : np.ndarray
        Unix epoch timestamps in seconds.

    Returns
    -------
    pd.DatetimeIndex
        UTC timestamps.
    """
    return pd.to_datetime(time_s, unit="s", utc=True)


def rotate_wind(
    u: np.ndarray,
    v: np.ndarray,
    alpha_deg: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Rotate wind components by buoy heading angle.

    Applies a standard 2-D rotation to correct for the buoy's
    orientation offset from true north:

        u_true =  u * cos(α) + v * sin(α)
        v_true = -u * sin(α) + v * cos(α)

    Parameters
    ----------
    u : np.ndarray
        Eastward wind velocity (m/s).
    v : np.ndarray
        Northward wind velocity (m/s).
    alpha_deg : float
        Rotation angle in degrees (buoy heading correction).

    Returns
    -------
    u_true, v_true : np.ndarray
        Rotated eastward and northward components (m/s).
    """
    a = np.deg2rad(alpha_deg)
    u_true = u * np.cos(a) + v * np.sin(a)
    v_true = -u * np.sin(a) + v * np.cos(a)
    return u_true, v_true


def wind_speed(u: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Compute scalar wind speed from components.

    Parameters
    ----------
    u, v : np.ndarray
        Eastward and northward wind velocity components (m/s).

    Returns
    -------
    np.ndarray
        Wind speed (m/s).
    """
    return np.sqrt(u**2 + v**2)


def wind_direction(u: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Compute meteorological wind direction from components.

    Meteorological convention: direction *from* which the wind blows,
    measured clockwise from true north, 0–360°.

    Parameters
    ----------
    u, v : np.ndarray
        Eastward and northward wind velocity components (m/s).

    Returns
    -------
    np.ndarray
        Wind direction in degrees (0–360).
    """
    wdir = np.degrees(np.arctan2(-u, -v))
    wdir[wdir < 0] += 360.0
    return wdir


# ---------------------------------------------------------------------------
# Wave direction processing
# ---------------------------------------------------------------------------
def rotate_wave_dir(
    mean_wave_dir: np.ndarray,
    alpha_deg: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Convert wave direction to rotated unit vectors.

    Converts the WAVSS mean wave direction (oceanographic convention,
    degrees) to unit vector components and rotates them by *alpha_deg*
    to correct for buoy orientation.

    Steps (matching original MATLAB):
      1. Convert oceanographic → meteorological-intermediate:
         ``dir_a = 90 - dir + 180``, wrap to [-180, 180].
      2. Compute unit vector: ``(cos(dir_a), sin(dir_a))``.
      3. Rotate by alpha.

    Parameters
    ----------
    mean_wave_dir : np.ndarray
        WAVSS mean wave direction (degrees, oceanographic convention).
    alpha_deg : float
        Buoy heading correction angle in degrees.

    Returns
    -------
    u_wave_true, v_wave_true : np.ndarray
        Rotated unit-vector components. Pass these to
        :func:`wave_direction_from_vectors` after binning.
    """
    dir_a = 90.0 - mean_wave_dir + 180.0
    dir_a[dir_a > 180.0] -= 360.0
    dir_a[dir_a < -180.0] += 360.0

    u_wave = np.cos(np.deg2rad(dir_a))
    v_wave = np.sin(np.deg2rad(dir_a))

    a = np.deg2rad(alpha_deg)
    u_true = u_wave * np.cos(a) + v_wave * np.sin(a)
    v_true = -u_wave * np.sin(a) + v_wave * np.cos(a)
    return u_true, v_true


def wave_direction_from_vectors(
    u: np.ndarray,
    v: np.ndarray,
) -> np.ndarray:
    """Recover meteorological wave direction from binned unit vectors.

    Parameters
    ----------
    u, v : np.ndarray
        Mean-binned rotated wave unit-vector components.

    Returns
    -------
    np.ndarray
        Wave direction in degrees (0–360, meteorological convention).
    """
    wdir = np.degrees(np.arctan2(-u, -v))
    wdir[wdir < 0] += 360.0
    return wdir


# ---------------------------------------------------------------------------
# Oceanographic derived variables
# ---------------------------------------------------------------------------
def calc_salinity(
    conductivity: np.ndarray,
    temperature: np.ndarray,
) -> np.ndarray:
    """Compute practical salinity from conductivity using GSW.

    Parameters
    ----------
    conductivity : np.ndarray
        Sea surface conductivity in S/m (OOI native units).
    temperature : np.ndarray
        Sea surface temperature in °C.

    Returns
    -------
    np.ndarray
        Practical salinity (PSU).
    """
    # OOI conductivity is in S/m; gsw_SP_from_C expects mS/cm
    return gsw.SP_from_C(conductivity * 10, temperature, 1.0)


def calc_rain_rate(precipitation_level: np.ndarray) -> np.ndarray:
    """Compute rain rate from cumulative precipitation level.

    Takes the first difference of the tipping-bucket accumulation and
    masks physically implausible values (resets or spikes > 2.5 mm per
    sample and any negative values).

    Parameters
    ----------
    precipitation_level : np.ndarray
        Cumulative precipitation level (mm).

    Returns
    -------
    np.ndarray
        Rain increment per sample interval (mm), NaN where invalid.
    """
    diff = np.empty_like(precipitation_level)
    diff[0] = np.nan
    diff[1:] = np.diff(precipitation_level)
    diff[diff > 2.5] = np.nan   # strict: 2.5 mm/sample is still valid
    diff[diff < 0.2] = np.nan
    return diff


# ---------------------------------------------------------------------------
# 10-minute binning
# ---------------------------------------------------------------------------
def make_bin_edges(
    bin_start: pd.Timestamp,
    bin_end: pd.Timestamp,
    interval_min: int = 10,
) -> pd.DatetimeIndex:
    """Generate bin-edge timestamps at *interval_min* spacing.

    Parameters
    ----------
    bin_start : pd.Timestamp
        First bin left edge (UTC).
    bin_end : pd.Timestamp
        Last bin right edge (UTC).
    interval_min : int
        Bin width in minutes. Default 10.

    Returns
    -------
    pd.DatetimeIndex
        Monotonically increasing bin edges including both endpoints.
    """
    return pd.date_range(
        start=bin_start, end=bin_end, freq=f"{interval_min}min", tz="UTC"
    )


def bin_observations(
    metbk: pd.DataFrame,
    wavss: pd.DataFrame,
    bin_edges: pd.DatetimeIndex,
    alpha_deg: float
) -> pd.DataFrame:
    """Bin and process METBK and WAVSS data into 10-minute averages.

    Applies wind rotation, wave direction rotation, salinity
    calculation, and rain rate computation before binning.  Wind and
    wave directions are recovered from the averaged vector components
    *after* binning.

    Binning is performed with ``DataFrame.resample`` anchored to
    ``bin_edges[0]``, so bins align exactly with the state-derived
    window boundaries.  METBK and WAVSS are resampled independently
    (they have different sample rates and timestamps) then concatenated.
    Empty bins yield NaN.

    Parameters
    ----------
    metbk : pd.DataFrame
        Raw METBK data from :func:`~ndbc_xml.ingest.load_metbk`.
    wavss : pd.DataFrame
        Raw WAVSS data from :func:`~ndbc_xml.ingest.load_wavss`.
    bin_edges : pd.DatetimeIndex
        Bin boundaries from :func:`make_bin_edges` (length n+1).
        ``bin_edges[0]`` anchors the resample origin; ``bin_edges[:-1]``
        are the bin labels used to reindex the result.
    alpha_deg : float
        Buoy heading correction angle in degrees.

    Returns
    -------
    pd.DataFrame
        One row per bin (``len(bin_edges) - 1`` rows).  The ``time``
        column holds the bin-center timestamp (UTC).  All other columns
        are the processed, binned variable names used downstream by
        :func:`~ndbc_xml.process.apply_qc` and the XML writer.
    """
    metbk_ts = epoch_to_datetime(metbk["time"].values)
    wavss_ts = epoch_to_datetime(wavss["time"].values)

    # --- derived METBK arrays (pre-binning) ---
    u, v = rotate_wind(
        metbk["eastward_wind_velocity"].values,
        metbk["northward_wind_velocity"].values,
        alpha_deg,
    )
    wspd = wind_speed(u, v)
    salinity = calc_salinity(
        metbk["sea_surface_conductivity"].values,
        metbk["sea_surface_temperature"].values,
    )
    rain = calc_rain_rate(metbk["precipitation_level"].values)

    # --- derived WAVSS arrays (pre-binning) ---
    u_wave, v_wave = rotate_wave_dir(
        wavss["mean_wave_direction"].values,
        alpha_deg,
    )

    # --- Build indexed DataFrames for resample ---
    rs_kw = dict(closed="left", label="left", origin=bin_edges[0])
    bin_labels = bin_edges[:-1]

    metbk_proc = pd.DataFrame(
        {
            "wind_speed":   wspd,
            "wind_dir_u":   u,
            "wind_dir_v":   v,
            "baro":         metbk["barometric_pressure"].values,
            "air_temp":     metbk["air_temperature"].values,
            "rel_humidity": metbk["relative_humidity"].values,
            "shortwave":    metbk["shortwave_irradiance"].values,
            "longwave":     metbk["longwave_irradiance"].values,
            "sst":          metbk["sea_surface_temperature"].values,
            "salinity":     salinity,
            "rain_rate":    rain,
        },
        index=metbk_ts,
    )

    wavss_proc = pd.DataFrame(
        {
            "sig_wave_hgt": wavss["significant_wave_height"].values,
            "max_wave_hgt": wavss["maximum_wave_height"].values,
            "peak_period":  wavss["peak_period"].values,
            "avg_period":   wavss["average_wave_period"].values,
            "wave_dir_u":   u_wave,
            "wave_dir_v":   v_wave,
        },
        index=wavss_ts,
    )

    m_bin = metbk_proc.resample("10min", **rs_kw).mean().reindex(bin_labels)
    w_bin = wavss_proc.resample("10min", **rs_kw).mean().reindex(bin_labels)

    bin_centers = bin_labels + pd.Timedelta(minutes=5)
    df = pd.concat([m_bin, w_bin], axis=1)
    df.insert(0, "time", bin_centers)

    # Recover directions from averaged vectors
    df["wind_dir"] = wind_direction(
        df["wind_dir_u"].values, df["wind_dir_v"].values
    )
    df["wave_dir"] = wave_direction_from_vectors(
        df["wave_dir_u"].values, df["wave_dir_v"].values
    )
    df.drop(columns=["wind_dir_u", "wind_dir_v",
                     "wave_dir_u", "wave_dir_v"], inplace=True)
    return df


# ---------------------------------------------------------------------------
# QC checks
# ---------------------------------------------------------------------------
def apply_qc(df: pd.DataFrame) -> pd.DataFrame:
    """Replace out-of-range values with NaN using global QC bounds.

    Operates on a copy; the input DataFrame is not modified.

    Parameters
    ----------
    df : pd.DataFrame
        Binned observations from :func:`bin_observations`.

    Returns
    -------
    pd.DataFrame
        Copy of *df* with out-of-range values set to NaN.
    """
    out = df.copy()
    for col, (lo, hi) in QC_BOUNDS.items():
        if col not in out.columns:
            continue
        mask = (out[col] < lo) | (out[col] > hi)
        out.loc[mask, col] = np.nan
    return out
