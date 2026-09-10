"""PWI4 horizontal-coordinate convention -- the single settled answer.

PWI4's ``mount.altitude_degs`` is **conventional apparent altitude**: 90 deg is
the zenith, 0 deg the horizon, higher is better. POLITE's usable window is
altitude **42--90 deg**; the shed blocks everything below 42 deg.

Owner decision 2026-09-07. These tests exist so the project cannot drift back
into treating that field as a zenith distance.
"""

from types import SimpleNamespace

import pytest

from obs_utils.config import SlewLimits, default_sky_regions
from obs_utils.mount import verify_pwi4_altitude
from obs_utils.night_session import TargetPlan, _auto_pointing_fields, _slew_to_target


def _pwi4(altitude_deg, azimuth_deg=180.0):
    return SimpleNamespace(
        status=lambda: SimpleNamespace(
            mount=SimpleNamespace(
                altitude_degs=altitude_deg,
                azimuth_degs=azimuth_deg,
            )
        )
    )


def test_default_pwi4_window_is_shed_safe():
    """The shed is not axisymmetric: az 0--90 needs alt >= 60, elsewhere >= 42."""
    limits = SimpleNamespace(enforce_regions=True, regions=default_sky_regions())

    # General sky (az 90--360): floor 42, zenith allowed.
    verify_pwi4_altitude(_pwi4(42.0, azimuth_deg=180.0), limits)
    verify_pwi4_altitude(_pwi4(90.0, azimuth_deg=180.0), limits)
    for bad in (41.9, 20.0):
        with pytest.raises(RuntimeError, match="outside the allowed"):
            verify_pwi4_altitude(_pwi4(bad, azimuth_deg=180.0), limits)

    # Northeast quadrant: shed wall, measured on sky 2026-09-07 (HD 204827).
    verify_pwi4_altitude(_pwi4(60.0, azimuth_deg=45.0), limits)
    verify_pwi4_altitude(_pwi4(64.0, azimuth_deg=5.0), limits)
    for bad_alt, bad_az in ((59.9, 45.0), (50.0, 45.0), (42.0, 89.0)):
        with pytest.raises(RuntimeError, match="outside the allowed"):
            verify_pwi4_altitude(_pwi4(bad_alt, azimuth_deg=bad_az), limits)


def test_fits_pointing_records_pwi4_altitude_verbatim():
    """No 90-minus conversion: ``altitude_degs`` already is FITS ALTITUDE."""
    pwi4 = SimpleNamespace(
        status=lambda: SimpleNamespace(
            mount=SimpleNamespace(
                altitude_degs=42.0,
                azimuth_degs=180.0,
                ra_j2000_hours=None,
                dec_j2000_degs=None,
            )
        )
    )

    fields = _auto_pointing_fields(pwi4)

    assert fields.alt_deg == 42.0            # not 48.0
    assert fields.airmass == pytest.approx(1.492, abs=0.003)


class _TrackingPwi4:
    def __init__(self, *, accepts_tracking=True):
        self.accepts_tracking = accepts_tracking
        self.goto = None
        self.tracking_requested = False

    def mount_goto_ra_dec_j2000(self, ra_hours, dec_deg):
        self.goto = (ra_hours, dec_deg)

    def mount_tracking_on(self):
        self.tracking_requested = True

    def status(self):
        return SimpleNamespace(
            mount=SimpleNamespace(
                altitude_degs=70.0,   # conventional altitude, inside 42--90
                azimuth_degs=180.0,
                is_slewing=False,
                is_tracking=self.accepts_tracking and self.tracking_requested,
            )
        )


def test_pointed_sequence_requires_tracking_after_the_target_slew():
    target = TargetPlan("HD 154892", ra_hours=17.12817, dec_deg=15.21056)

    healthy = _TrackingPwi4()
    _slew_to_target(healthy, target, SlewLimits())
    assert healthy.goto == (17.12817, 15.21056)
    assert healthy.tracking_requested

    with pytest.raises(RuntimeError, match="tracking enabled"):
        _slew_to_target(_TrackingPwi4(accepts_tracking=False), target, SlewLimits())
