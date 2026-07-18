import subprocess
import sys
from pathlib import Path


def test_cooler_gate_only_observes_after_initial_setpoint():
    code = '''
from scripts.run_calibration_night import _wait_for_cooler

class Camera:
    CanSetCCDTemperature = True
    CCDTemperature = 0.1
    CoolerPower = 42.0

    def __init__(self):
        self.setpoint_writes = 0

    @property
    def SetCCDTemperature(self):
        return 0.0

    @SetCCDTemperature.setter
    def SetCCDTemperature(self, _value):
        self.setpoint_writes += 1

camera = Camera()
achieved = _wait_for_cooler(
    camera, 0.0, tol_c=0.5, stable_s=0.0,
    timeout_s=1.0, poll_s=0.0, assume_yes=True,
)
assert achieved == 0.1
assert camera.setpoint_writes == 0
'''
    root = Path(__file__).resolve().parents[1]
    result = subprocess.run(
        [sys.executable, "-c", code], cwd=root, text=True, capture_output=True,
    )

    assert result.returncode == 0, result.stderr
