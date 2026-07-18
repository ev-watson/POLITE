import subprocess
import sys
from pathlib import Path

import pytest

from obs_utils.session_context import session_context_from_yaml


def test_cooler_policy_parses_and_rejects_unknown_values():
    assert session_context_from_yaml({"cooler_policy": "at-or-below"}).cooler_policy == "at_or_below"
    assert session_context_from_yaml({"cooler_policy": "exact"}).cooler_policy == "exact"
    with pytest.raises(ValueError, match="cooler_policy"):
        session_context_from_yaml({"cooler_policy": "manual"})


def test_at_or_below_preserves_a_colder_detector_and_exact_does_not():
    code = '''
from types import SimpleNamespace
from obs_utils import night_session
from obs_utils.session_context import SessionCaptureContext

night_session.configure_camera = lambda *_args, **_kwargs: None

class Camera:
    CanSetCCDTemperature = True
    CCDTemperature = -18.0
    def __init__(self):
        self.commanded = None
    @property
    def SetCCDTemperature(self):
        return self.commanded
    @SetCCDTemperature.setter
    def SetCCDTemperature(self, value):
        self.commanded = value

camera = Camera()
session = SimpleNamespace(camera=camera)
ctx = SessionCaptureContext(cooler_setpoint_c=-15.0, cooler_policy="at_or_below")
effective = night_session._apply_session_camera(session, ctx)
assert camera.commanded == -18.0
assert effective.cooler_setpoint_c == -18.0

ctx = SessionCaptureContext(cooler_setpoint_c=-15.0, cooler_policy="exact")
effective = night_session._apply_session_camera(session, ctx)
assert camera.commanded == -15.0
assert effective.cooler_setpoint_c == -15.0

camera.CCDTemperature = -15.0
assert night_session.wait_for_cooler(
    camera, -15.0, tolerance_c=0.1, stable_s=0.0, timeout_s=1.0, poll_s=0.0,
) == -15.0
'''
    root = Path(__file__).resolve().parents[1]
    result = subprocess.run(
        [sys.executable, "-c", code], cwd=root, text=True, capture_output=True,
    )
    assert result.returncode == 0, result.stderr
