import pytest

from obs_utils.waits import DeviceTimeout, wait_for_value, wait_until


def test_wait_until_returns_immediately_when_ready():
    assert wait_until(lambda: True, timeout_s=1.0, poll_s=0.0, what="camera") >= 0.0


def test_wait_until_names_a_timed_out_device():
    with pytest.raises(DeviceTimeout, match="focuser enable"):
        wait_until(lambda: False, timeout_s=0.0, poll_s=0.0, what="focuser enable")


def test_wait_for_value_reports_the_last_reading():
    with pytest.raises(DeviceTimeout, match="last position 3"):
        wait_for_value(lambda: 3, 10, tol=0.1, timeout_s=0.0, poll_s=0.0, what="focus")
