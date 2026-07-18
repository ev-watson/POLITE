import subprocess
import sys
from pathlib import Path



def _run(code: str) -> None:
    root = Path(__file__).resolve().parents[1]
    result = subprocess.run(
        [sys.executable, "-c", code], cwd=root, text=True, capture_output=True,
    )
    assert result.returncode == 0, result.stderr


def test_connect_camera_waits_for_the_sdk_handshake():
    _run('''
from obs_utils import alpaca

class FakeCamera:
    def __init__(self, host, device_number):
        self.polls = 0
        self.started = False

    @property
    def Connected(self):
        self.polls += 1
        return self.started and self.polls >= 3

    @Connected.setter
    def Connected(self, value):
        assert value is True
        self.started = True

    @property
    def Connecting(self):
        return True

alpaca.CameraDevice = FakeCamera
alpaca.time.sleep = lambda _s: None
camera = alpaca.connect_camera("localhost:11111", 0, timeout_s=1, poll_s=0)
assert isinstance(camera, FakeCamera)
assert camera.polls == 3
''')


def test_connect_camera_reports_a_failed_handshake():
    _run('''
from obs_utils import alpaca

class FakeCamera:
    def __init__(self, host, device_number):
        pass

    @property
    def Connected(self):
        return False

    @Connected.setter
    def Connected(self, value):
        assert value is True

    @property
    def Connecting(self):
        return False

alpaca.CameraDevice = FakeCamera
try:
    alpaca.connect_camera("localhost:11111", 0)
except RuntimeError as exc:
    assert "did not connect" in str(exc)
else:
    raise AssertionError("expected failed handshake")
''')
