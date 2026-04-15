#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Compute magnetic declination using the IGRF-14 model via ppigrf.

ppigrf is the official pure-Python IGRF implementation released by
the IAGA V-MOD Working Group. It ships its own IGRF-14 spherical
harmonic coefficient file (``IGRF14.shc``) and requires no Fortran
compiler or external data downloads.

Reference
---------
Laundal, K. M. and Richmond, A. D. (2017). Magnetic Coordinate
Systems. Space Sci. Rev., 206, 27–59.
https://doi.org/10.1007/s11214-016-0275-y

IGRF-14 coefficients: Alken et al. (2021), Earth Planets Space 73,
49. https://doi.org/10.1186/s40623-020-01288-x

Declination is the angle between true north and magnetic north,
positive east. It is used here as the buoy heading correction angle
(alpha) to rotate wind and wave direction components from the buoy's
magnetic frame to true geographic north.
"""

from datetime import datetime

import numpy as np
import ppigrf


def get_declination(
    latitude: float,
    longitude: float,
    date: datetime,
    altitude_km: float = 0.0,
) -> float:
    """Compute magnetic declination at a point using IGRF-14.

    Parameters
    ----------
    latitude : float, array_like
        Geodetic latitude in decimal degrees north.
    longitude : float, array_like
        Geodetic longitude in decimal degrees east (negative west).
    date : datetime, array_like
        Date for which to compute declination. A mid-deployment date
        is appropriate for a multi-month record; declination changes
        at roughly 0.1 deg/yr at Oregon coast locations.
    altitude_km : float, array_like
        Height above the WGS-84 ellipsoid in kilometers.
        Default 0.0 (sea level).

    Returns
    -------
    float
        Magnetic declination in degrees, positive east.

    Examples
    --------
    >>> from datetime import datetime
    >>> get_declination(44.639, -124.304, datetime(2026, 1, 1))
    14.602...
    """
    # ppigrf requires a timezone-naive datetime; strip tzinfo if present.
    if hasattr(date, "tzinfo") and date.tzinfo is not None:
        naive_date = date.replace(tzinfo=None)
    else:
        naive_date = date
    be, bn, _ = ppigrf.igrf(longitude, latitude, altitude_km, naive_date)  # type: ignore[arg-type]
    # arctan2(east, north) gives the signed angle of the magnetic
    # field vector from geographic north — i.e. the declination.
    return float(np.degrees(np.arctan2(be.squeeze(), bn.squeeze())))
