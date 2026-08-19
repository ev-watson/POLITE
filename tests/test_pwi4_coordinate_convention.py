from types import SimpleNamespace

import pytest

from obs_utils.config import SlewLimits, default_sky_regions
from obs_utils.mount import verify_pwi4_zenith_distance
from obs_utils.night_session import TargetPlan, _auto_pointing_fields, _slew_to_target
from obs_utils.obs_math import zenith_distance_to_altitude


def _pwi4(zenith_distance_deg, azimuth_deg=180.0):
    return SimpleNamespace(
        status=lambda: SimpleNamespace(
            mount=SimpleNamespace(
                altitude_degs=zenith_distance_deg,
                azimuth_degs=azimuth_deg,
            )
        )
    )


def test_pwi4_zenith_distance_is_converted_before_airmass_use():
    assert zenith_distance_to_altitude(0.0) == 90.0
    assert zenith_distance_to_altitude(42.0) == 48.0
    assert zenith_distance_to_altitude(90.0) == 0.0
    assert zenith_distance_to_altitude(None) is None
    assert zenith_distance_to_altitude(91.0) is None


def test_default_pwi4_window_is_shed_safe():
    region, = default_sky_regions()
    assert (region.alt_min_deg, region.alt_max_deg) == (3.0, 42.0)

    limits = SimpleNamespace(enforce_regions=True, regions=[region])
    verify_pwi4_zenith_distance(_pwi4(3.0), limits)
    verify_pwi4_zenith_distance(_pwi4(42.0), limits)
    with pytest.raises(RuntimeError, match="outside the allowed"):
        verify_pwi4_zenith_distance(_pwi4(42.1), limits)


def test_fits_pointing_uses_conventional_altitude_not_pwi4_zenith_distance():
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

    assert fields.alt_deg == 48.0
    assert fields.airmass == pytest.approx(1.342, abs=0.003)


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
                altitude_degs=20.0,
                azimuth_degs=180.0,
                is_slewing=False,
                is_tracking=self.accepts_tracking and self.tracking_requested,
            )
        )


def test_pointed_sequence_requires_tracking_after_the_repaired_mount_slew():
    target = TargetPlan("HD 154892", ra_hours=17.12817, dec_deg=15.21056)

    healthy = _TrackingPwi4()
    _slew_to_target(healthy, target, SlewLimits())
    assert healthy.goto == (17.12817, 15.21056)
    assert healthy.tracking_requested

    with pytest.raises(RuntimeError, match="tracking enabled"):
        _slew_to_target(_TrackingPwi4(accepts_tracking=False), target, SlewLimits())
