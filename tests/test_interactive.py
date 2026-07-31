"""Bounded PWI4 notebook controls and fail-closed pair centering."""

from types import SimpleNamespace as NS

import numpy as np
import pytest

from obs_utils import interactive, startup
from obs_utils.waits import DeviceTimeout


def _status(*, focuser_connected=True, focuser_enabled=True,
            rotator_connected=True, rotator_enabled=True):
    return NS(
        focuser=NS(exists=True, is_connected=focuser_connected,
                   is_enabled=focuser_enabled, position=1000.0, is_moving=False),
        rotator=NS(exists=True, is_connected=rotator_connected,
                   is_enabled=rotator_enabled, mech_position_degs=10.0,
                   field_angle_degs=20.0, is_moving=False),
        mount=NS(is_connected=True, is_slewing=False, is_tracking=False,
                 ra_j2000_hours=1.0, dec_j2000_degs=2.0),
    )


class FakePWI:
    def __init__(self, status=None):
        self.st = status or _status()
        self.calls = []

    def status(self):
        return self.st

    def focuser_connect(self):
        self.calls.append("focuser_connect")
        self.st.focuser.is_connected = True

    def focuser_enable(self):
        self.calls.append("focuser_enable")
        self.st.focuser.is_enabled = True

    def focuser_goto(self, target):
        self.calls.append(("focuser_goto", target))
        self.st.focuser.position = target

    def focuser_stop(self):
        self.calls.append("focuser_stop")

    def rotator_connect(self):
        self.calls.append("rotator_connect")
        self.st.rotator.is_connected = True

    def rotator_enable(self):
        self.calls.append("rotator_enable")
        self.st.rotator.is_enabled = True

    def rotator_goto_field(self, target):
        self.calls.append(("rotator_goto_field", target))
        self.st.rotator.field_angle_degs = target

    def rotator_goto_mech(self, target):
        self.calls.append(("rotator_goto_mech", target))
        self.st.rotator.mech_position_degs = target

    def rotator_offset(self, delta):
        self.calls.append(("rotator_offset", delta))
        self.st.rotator.mech_position_degs += delta

    def rotator_stop(self):
        self.calls.append("rotator_stop")


def test_interactive_focuser_and_field_rotator_wait_for_the_reported_position():
    pwi4 = FakePWI()
    session = interactive.ObservatorySession(pwi4=pwi4)

    assert session.focus(1200.0, timeout_s=0.0) == 1200.0
    assert session.focus_relative(-50.0, timeout_s=0.0) == 1150.0
    assert session.field_rotator_goto_field(45.0, timeout_s=0.0) == 45.0
    assert session.field_rotator_goto_mech(90.0, timeout_s=0.0) == 90.0
    assert session.field_rotator_offset(-5.0, timeout_s=0.0) == 85.0
    session.focus_stop()
    session.field_rotator_stop()
    assert "focuser_stop" in pwi4.calls and "rotator_stop" in pwi4.calls


def test_startup_auxiliary_connects_enable_each_axis_and_are_bounded():
    pwi4 = FakePWI(_status(focuser_connected=False, focuser_enabled=False,
                           rotator_connected=False, rotator_enabled=False))
    assert startup.connect_focuser(pwi4, timeout_s=0.0)
    assert startup.connect_field_rotator(pwi4, timeout_s=0.0)
    assert {"focuser_connect", "focuser_enable", "rotator_connect", "rotator_enable"} <= set(pwi4.calls)

    stalled = FakePWI(_status(focuser_connected=False, focuser_enabled=False))
    stalled.focuser_connect = lambda: None
    with pytest.raises(DeviceTimeout, match="focuser connect"):
        startup.connect_focuser(stalled, timeout_s=0.0)


def test_center_on_pair_rejects_nominal_uncharacterized_geometry_before_exposure():
    session = interactive.ObservatorySession()
    cfg = NS(beam_geometry_characterized=False)
    with pytest.raises(RuntimeError, match="not characterized"):
        session.center_on_pair(frame=np.zeros((20, 20)), pol_config=cfg)

