"""Observation-geometry math for FITS provenance.

Airmass, geocentric site vector, and sexagesimal formatting. Kept dependency-light
(astropy only) and free of any ``obs_utils`` imports so writers and readers can reuse
it without pulling in the device layer.
"""
from __future__ import annotations

import math
from typing import Optional, Tuple

__all__ = [
    "airmass_kasten_young",
    "zenith_distance_to_altitude",
    "obsgeo_xyz",
    "hours_to_hms",
    "deg_to_dms",
]


def airmass_kasten_young(altitude_deg: Optional[float]) -> Optional[float]:
    """Relative optical airmass from apparent altitude.

    Uses the Kasten & Young (1989) approximation formula:

        X = 1 / (sin h + 0.50572 (h + 6.07995)^-1.6364)

    with ``h`` the apparent altitude in degrees.

    Reference: Kasten, F. & Young, A. T. 1989, "Revised optical air mass tables
    and approximation formula", Applied Optics 28, 4735 (doi:10.1364/AO.28.004735).

    Returns ``None`` for a missing altitude or a target at/below the horizon.
    """
    if altitude_deg is None:
        return None
    h = float(altitude_deg)
    if h <= 0.0:
        return None
    denom = math.sin(math.radians(h)) + 0.50572 * (h + 6.07995) ** -1.6364
    if denom <= 0.0:
        return None
    return 1.0 / denom


def zenith_distance_to_altitude(zenith_distance_deg: Optional[float]) -> Optional[float]:
    """Convert zenith distance to conventional apparent altitude.

    POLITE's PWI4 display/API uses ``0 deg = zenith`` and ``90 deg = horizon``
    for the field it calls ``altitude_degs``.  FITS ``ALTITUDE`` and airmass,
    by contrast, use the conventional astronomical altitude above the horizon.
    Keep this conversion explicit at the device boundary rather than letting a
    PWI4 coordinate leak into physical metadata.
    """
    if zenith_distance_deg is None:
        return None
    z = float(zenith_distance_deg)
    if not 0.0 <= z <= 90.0:
        return None
    return 90.0 - z


def obsgeo_xyz(
    lat_deg: Optional[float],
    lon_deg: Optional[float],
    elev_m: Optional[float],
) -> Optional[Tuple[float, float, float]]:
    """Geocentric (ITRS) site position in metres → ``(OBSGEO-X, -Y, -Z)``.

    Longitude is +East. Returns ``None`` if any input is missing or astropy fails.
    """
    if lat_deg is None or lon_deg is None or elev_m is None:
        return None
    try:
        from astropy import units as u
        from astropy.coordinates import EarthLocation

        loc = EarthLocation.from_geodetic(
            lon=float(lon_deg) * u.deg,
            lat=float(lat_deg) * u.deg,
            height=float(elev_m) * u.m,
        )
        return (
            float(loc.x.to_value(u.m)),
            float(loc.y.to_value(u.m)),
            float(loc.z.to_value(u.m)),
        )
    except Exception:
        return None


def hours_to_hms(
    hours: Optional[float],
    *,
    signed: bool = False,
    wrap: bool = True,
) -> Optional[str]:
    """Format an angle in hours as sexagesimal ``HH:MM:SS.ss``.

    ``wrap`` folds into ``[0, 24)`` (use for RA/LST); ``signed`` keeps a leading sign
    (use for hour angle). Returns ``None`` on missing input or formatting failure.
    """
    if hours is None:
        return None
    try:
        from astropy import units as u
        from astropy.coordinates import Angle

        ang = Angle(float(hours) * u.hourangle)
        if wrap:
            ang = ang.wrap_at(24 * u.hourangle)
        return ang.to_string(
            unit=u.hour, sep=":", precision=2, pad=True, alwayssign=signed
        )
    except Exception:
        return None


def deg_to_dms(deg: Optional[float]) -> Optional[str]:
    """Format an angle in degrees as signed sexagesimal ``±DD:MM:SS.s`` (for Dec)."""
    if deg is None:
        return None
    try:
        from astropy import units as u
        from astropy.coordinates import Angle

        return Angle(float(deg) * u.deg).to_string(
            unit=u.deg, sep=":", precision=1, alwayssign=True, pad=True
        )
    except Exception:
        return None
