"""Offline protocol tests for the native Optec Pyxis 2" Gen3 serial driver.

No hardware / no pyserial: a scripted fake serial port is injected as the
driver's ``_ser`` so ``transact`` round-trips against the manual's example
responses (double-END GETDNN, GETSTA fields, error frames, CR-only line
endings, move/home polling). Protocol reference:
``reference-sheets/hardware/pyxis-command-processing-rev2.md``.
"""

import pytest

from obs_utils.pyxis_gen3 import PyxisError, PyxisGen3, PyxisTimeout

STATUS_HOMED = (
    "!{tid}\n"
    "Current Step = 0\n"
    "Target Step = 0\n"
    "Current PA = 180000\n"
    "Target PA = 180000\n"
    "Is Moving = 0\n"
    "Is Homing = 0\n"
    "Is Homed = 1\n"
    "Is Sleeping = 0\n"
    "END\n"
)


def _parse_frame(frame: str) -> dict:
    inner = frame[1:-1]  # strip < >
    return {
        "dev": inner[0],
        "did": inner[1],
        "tid": inner[2:4],
        "cmd": inner[4:10],
        "payload": inner[10:],
    }


class FakeSerial:
    """Scripted serial port: each ``write`` enqueues ``responder(frame)``."""

    def __init__(self, responder, line_sep="\n"):
        self.responder = responder
        self.line_sep = line_sep
        self._buf = b""
        self.is_open = True
        self.timeout = 1.0

    def reset_input_buffer(self):
        self._buf = b""

    def reset_output_buffer(self):
        pass

    def write(self, data):
        resp = self.responder(_parse_frame(data.decode()))
        if self.line_sep != "\n":
            resp = resp.replace("\n", self.line_sep)
        self._buf += resp.encode()
        return len(data)

    def flush(self):
        pass

    def read(self, n):
        chunk, self._buf = self._buf[:n], self._buf[n:]
        return chunk

    def close(self):
        self.is_open = False


def make_dev(responder, line_sep="\n") -> PyxisGen3:
    dev = PyxisGen3("fake", read_timeout_s=0.5)
    dev._ser = FakeSerial(responder, line_sep=line_sep)
    return dev


# --------------------------------------------------------------------------- #
# PA <-> deg conversion
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "deg,pa",
    [
        (90, 90000),
        (0, 0),
        (22.5, 22500),
        (180, 180000),
        (359.999, 359999),
        (360, 0),
        (-15, 345000),
        (405, 45000),
        (720, 0),
    ],
)
def test_deg_to_pa(deg, pa):
    assert PyxisGen3._deg_to_pa(deg) == pa


def test_pa_to_deg():
    assert PyxisGen3._pa_to_deg("180000") == 180.0
    assert PyxisGen3._pa_to_deg(90000) == 90.0


def test_deg_to_pa_out_of_range_after_wrap_is_impossible():
    # every real deg wraps into [0,360); conversion must always be in range
    for deg in (-1000.0, 1e6, 359.9999, -0.0001):
        assert 0 <= PyxisGen3._deg_to_pa(deg) <= 359999


# --------------------------------------------------------------------------- #
# Command round-trips
# --------------------------------------------------------------------------- #
def test_ping_handles_doubled_end():
    def responder(f):
        if f["cmd"] == "GETDNN":
            return f"!{f['tid']}\nNickname = Pollux\nEND\nEND\n"  # doubled END
        return STATUS_HOMED.format(tid=f["tid"])

    dev = make_dev(responder)
    assert dev.ping() == "Pollux"
    # the leftover second END must not corrupt the next command
    assert dev.get_status()["Is Homed"] == "1"


def test_get_status_and_accessors():
    dev = make_dev(lambda f: STATUS_HOMED.format(tid=f["tid"]))
    st = dev.get_status()
    assert st["Current PA"] == "180000"
    assert dev.is_homed is True
    assert dev.is_moving is False
    assert dev.position_deg == 180.0


def test_cr_only_line_endings():
    dev = make_dev(lambda f: STATUS_HOMED.format(tid=f["tid"]), line_sep="\r")
    assert dev.get_status()["Is Homed"] == "1"


def test_error_frame_raises_pyxis_error():
    def responder(f):
        return (
            "ERROR ID = 11\n"
            "ERROR TEXT = The command failed because the rotator is not homed\n"
            "END\n"
        )

    dev = make_dev(responder)
    with pytest.raises(PyxisError) as exc:
        dev.move_absolute(90)
    assert exc.value.code == 11
    assert "not homed" in exc.value.text.lower()


def test_frame_construction_and_transaction_ids():
    seen = []

    def responder(f):
        seen.append(f)
        return f"!{f['tid']}\nEND\n"

    dev = make_dev(responder)
    for _ in range(3):
        dev.transact("DOHALT")
    assert [s["tid"] for s in seen] == ["00", "01", "02"]
    assert seen[0]["dev"] == "R" and seen[0]["did"] == "1"
    assert seen[0]["cmd"] == "DOHALT"


# --------------------------------------------------------------------------- #
# Move / home polling
# --------------------------------------------------------------------------- #
def test_move_absolute_polls_until_settled():
    class Mover:
        def __init__(self, n_moving=2):
            self.calls = 0
            self.n_moving = n_moving
            self.moved_to = None

        def __call__(self, f):
            if f["cmd"] == "MOVEPA":
                self.moved_to = int(f["payload"])
                return f"!{f['tid']}\nEND\n"
            self.calls += 1
            moving = 1 if self.calls <= self.n_moving else 0
            pa = self.moved_to or 0
            return (
                f"!{f['tid']}\nCurrent PA = {pa}\nTarget PA = {pa}\n"
                f"Is Moving = {moving}\nIs Homing = 0\nIs Homed = 1\nEND\n"
            )

    mv = Mover()
    dev = make_dev(mv)
    final = dev.move_absolute(90, poll_s=0.001, pre_poll_s=0.0)
    assert mv.moved_to == 90000
    assert final == 90.0
    assert mv.calls >= 3  # polled through the moving phase


def test_move_absolute_times_out_when_never_settles():
    dev = make_dev(
        lambda f: (
            f"!{f['tid']}\nCurrent PA = 0\nIs Moving = 1\n"
            f"Is Homing = 0\nIs Homed = 1\nEND\n"
        )
    )
    with pytest.raises(PyxisTimeout):
        dev.move_absolute(45, poll_s=0.001, timeout_s=0.02, pre_poll_s=0.0)


def test_home_polls_until_homed():
    class Homer:
        def __init__(self, n_homing=2):
            self.calls = 0
            self.n_homing = n_homing
            self.did_home = False

        def __call__(self, f):
            if f["cmd"] == "DOHOME":
                self.did_home = True
                return f"!{f['tid']}\nEND\n"
            self.calls += 1
            homing = 1 if self.calls <= self.n_homing else 0
            homed = 0 if homing else 1
            return (
                f"!{f['tid']}\nCurrent PA = 0\nIs Moving = {homing}\n"
                f"Is Homing = {homing}\nIs Homed = {homed}\nEND\n"
            )

    hm = Homer()
    dev = make_dev(hm)
    final = dev.home(poll_s=0.001, timeout_s=5.0)
    assert hm.did_home is True
    assert final == 0.0
