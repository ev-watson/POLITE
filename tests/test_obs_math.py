"""obs_utils.obs_math: airmass (Kasten-Young 1989), geocentric site vector, sexagesimal."""
from __future__ import annotations

import pytest

from obs_utils.obs_math import (
    airmass_kasten_young,
    deg_to_dms,
    hours_to_hms,
    obsgeo_xyz,
)


def test_airmass_kasten_young_reference_values():
    # Kasten, F. & Young, A.T. 1989, Appl. Opt. 28, 4735 (doi:10.1364/AO.28.004735).
    assert airmass_kasten_young(90.0) == pytest.approx(1.000, abs=1e-3)
    assert airmass_kasten_young(60.0) == pytest.approx(1.154, abs=3e-3)
    assert airmass_kasten_young(30.0) == pytest.approx(1.995, abs=5e-3)
    assert airmass_kasten_young(10.0) == pytest.approx(5.6, abs=0.1)


def test_airmass_horizon_and_missing_return_none():
    assert airmass_kasten_young(0.0) is None
    assert airmass_kasten_young(-5.0) is None
    assert airmass_kasten_young(None) is None


def test_airmass_increases_toward_horizon():
    xs = [airmass_kasten_young(a) for a in (90, 70, 50, 30, 20, 10, 5)]
    assert all(b > a for a, b in zip(xs, xs[1:]))


def test_obsgeo_matches_astropy_earthlocation():
    u = pytest.importorskip("astropy.units")
    from astropy.coordinates import EarthLocation

    lat, lon, elev = 33.0701, -116.6451, 1294.0
    loc = EarthLocation.from_geodetic(lon * u.deg, lat * u.deg, elev * u.m)
    x, y, z = obsgeo_xyz(lat, lon, elev)
    assert x == pytest.approx(loc.x.to_value(u.m), abs=1.0)
    assert y == pytest.approx(loc.y.to_value(u.m), abs=1.0)
    assert z == pytest.approx(loc.z.to_value(u.m), abs=1.0)


def test_obsgeo_missing_input_returns_none():
    assert obsgeo_xyz(None, -116.0, 1294.0) is None
    assert obsgeo_xyz(33.0, None, 1294.0) is None
    assert obsgeo_xyz(33.0, -116.0, None) is None


def test_sexagesimal_formatting():
    # RA (hours) -> HH:MM:SS.ss, wrapped into [0,24).
    assert hours_to_hms(23.2114722) == "23:12:41.30"
    # Hour angle keeps its sign and is not wrapped.
    assert hours_to_hms(-0.00489, signed=True, wrap=False) == "-00:00:17.60"
    # Dec (deg) -> signed DD:MM:SS.s.
    assert deg_to_dms(-5.349722).startswith("-05:20:")
    assert deg_to_dms(12.5).startswith("+12:30:")
    assert deg_to_dms(None) is None
    assert hours_to_hms(None) is None
