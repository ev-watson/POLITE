"""Fail-closed behaviour of the shared night-runner safety gates.

These gates are the only thing standing between an operator and a night of
unreducible frames, so each test asserts the *abort*, not just the log line.

``obs_utils.night_safety`` is pure: it never imports the camera/Alpaca stack at
module scope.  To keep that property testable (see
``tests/test_optional_imports.py``) the plan-driven and preflight-move cases run
in subprocesses -- importing ``obs_utils.night_plan`` or ``obs_utils.imaging``
in-process would poison ``sys.modules`` for every test that follows.
"""
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

from obs_utils.night_safety import (
    INSTALLED_EFW_NAMES,
    plan_hwp_angles,
    plan_pointed_targets,
    plan_requires_hwp,
    plan_requires_mount,
    verify_filter_wheel,
    verify_hwp,
    verify_mount,
)

ROOT = Path(__file__).resolve().parents[1]
CAL_PLAN = ROOT / "night_plans" / "20260717_darkcal.yaml"
POL_PLAN = ROOT / "night_plans" / "20260709.yaml"


def _run(code: str) -> None:
    """Execute ``code`` in a clean interpreter rooted at the repo."""
    result = subprocess.run(
        [sys.executable, "-c", code], cwd=ROOT, text=True, capture_output=True,
    )
    assert result.returncode == 0, result.stderr


# --------------------------------------------------------------------------- #
# Cooler gate
# --------------------------------------------------------------------------- #
def test_cooler_gate_only_observes_after_initial_setpoint():
    """The QHY SDK owns the cooler PID; the gate must never write the setpoint."""
    _run('''
from obs_utils.night_safety import cooler_gate

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
achieved = cooler_gate(
    camera, 0.0, tol_c=0.5, stable_s=0.0,
    timeout_s=1.0, poll_s=0.0, assume_yes=True,
)
assert achieved == 0.1
assert camera.setpoint_writes == 0
''')


# --------------------------------------------------------------------------- #
# EFW name -> slot resolution
# --------------------------------------------------------------------------- #
class _Wheel:
    def __init__(self, names):
        self.Names = names


class _Imaging:
    def __init__(self, *, wheel_names=None, filter_names=None, rotator=None):
        self.filter_wheel = _Wheel(wheel_names) if wheel_names is not None else None
        self.filter_names = filter_names
        self.rotator = rotator
        self.camera = None


def test_filter_wheel_names_matching_carousel_pass():
    verify_filter_wheel(_Imaging(wheel_names=list(INSTALLED_EFW_NAMES)))


def test_filter_wheel_name_mismatch_aborts():
    """A reordered carousel sends every 'Dark' request to the wrong slot."""
    scrambled = list(reversed(INSTALLED_EFW_NAMES))
    with pytest.raises(SystemExit):
        verify_filter_wheel(_Imaging(wheel_names=scrambled))


def test_filter_wheel_mismatch_is_overridable():
    scrambled = list(reversed(INSTALLED_EFW_NAMES))
    verify_filter_wheel(_Imaging(wheel_names=scrambled), skip=True)


# --------------------------------------------------------------------------- #
# HWP angle detection from a plan
# --------------------------------------------------------------------------- #
def _fake_config(*, before=(), targets=(), after=(), stage="before"):
    """Duck-typed NightSessionConfig -- night_safety only reads these fields."""
    return SimpleNamespace(
        calibration_before=list(before),
        calibration_after=list(after),
        calibration_stage=stage,
        targets=[SimpleNamespace(frames=list(frames)) for frames in targets],
    )


def _frame(angle=None):
    return SimpleNamespace(hwp_angle_deg=angle)


def test_cal_frames_command_no_hwp_angles():
    """A dark ladder must not drag the rotator into the run."""
    config = _fake_config(before=[_frame(), _frame()])
    assert plan_hwp_angles(config) == []
    assert plan_requires_hwp(config) is False


def test_plan_angles_are_deduplicated_and_sorted():
    config = _fake_config(targets=[[_frame(45.0), _frame(0.0), _frame(45.0)]])
    assert plan_hwp_angles(config) == [0.0, 45.0]
    assert plan_requires_hwp(config) is True


def test_skipped_calibration_stage_is_not_scanned():
    """``calibration_stage: after`` means the before-block never executes."""
    config = _fake_config(
        before=[_frame(0.0)], after=[_frame(90.0)], stage="after",
    )
    assert plan_hwp_angles(config) == [90.0]


def test_real_plans_report_their_hwp_angles():
    """The duck-typed fakes above must match the shipped plan schema."""
    _run(f'''
from pathlib import Path
from obs_utils.night_plan import load_night_plan
from obs_utils.night_safety import plan_hwp_angles, plan_requires_hwp

cal = load_night_plan(Path({str(CAL_PLAN)!r}))
assert plan_hwp_angles(cal) == []
assert plan_requires_hwp(cal) is False

pol = load_night_plan(Path({str(POL_PLAN)!r}))
angles = plan_hwp_angles(pol)
assert plan_requires_hwp(pol) is True
assert angles == sorted(set(angles)), "angles must be unique and sorted"
# Standard sampling is 22.5 deg steps from 0 (night_plan.hwp_angles).
assert angles[0] == 0.0
assert all(abs((a / 22.5) - round(a / 22.5)) < 1e-9 for a in angles)
''')


# --------------------------------------------------------------------------- #
# HWP gate
# --------------------------------------------------------------------------- #
def test_hwp_gate_aborts_when_rotator_missing():
    """Capturing a modulation sequence with no rotator yields wrong HWPANG."""
    with pytest.raises(SystemExit):
        verify_hwp(_Imaging(rotator=None), [0.0, 22.5, 45.0, 67.5])


def test_hwp_gate_missing_rotator_is_overridable():
    assert verify_hwp(_Imaging(rotator=None), [0.0], skip=True) is None


class _Rotator:
    """Stand-in for the open-loop Pyxis Gen3 seen through Alpaca."""

    def __init__(self):
        self.Position = 0.0
        self.MechanicalPosition = 0.0
        self.IsMoving = False
        self.StepSize = 0.012


def test_hwp_preflight_can_be_skipped():
    """--no-hwp-preflight leaves the stage where it is (no physical motion)."""
    rot = _Rotator()
    assert verify_hwp(_Imaging(rotator=rot), [22.5], preflight=False) is None
    assert rot.Position == 0.0


_PREFLIGHT_HARNESS = '''
import obs_utils.imaging as im
from obs_utils.night_safety import verify_hwp

class Rotator:
    def __init__(self):
        self.Position = 0.0
        self.MechanicalPosition = 0.0
        self.IsMoving = False
        self.StepSize = 0.012

class Imaging:
    def __init__(self, rotator):
        self.filter_wheel = None
        self.filter_names = None
        self.rotator = rotator
        self.camera = None

def patch_move(error_deg):
    """Route select_hwp_angle to the stub without touching alpaca transport."""
    def _fake(session, angle_deg, poll_s=0.5, timeout_s=120.0):
        if session.rotator is None:
            raise RuntimeError("No HWP rotator connected")
        session.rotator.Position = float(angle_deg) + error_deg
        return session.rotator.Position
    im.select_hwp_angle = _fake
'''


def test_hwp_preflight_passes_within_step_quantization():
    """~0.012 deg is the open-loop step floor, not a fault."""
    _run(_PREFLIGHT_HARNESS + '''
patch_move(0.012)
achieved = verify_hwp(Imaging(Rotator()), [22.5, 45.0])
assert abs(achieved - 22.512) < 1e-9, achieved
''')


def test_hwp_preflight_aborts_when_stage_does_not_reach_angle():
    """A stalled or unhomed stage must abort before the first frame."""
    _run(_PREFLIGHT_HARNESS + '''
patch_move(5.0)
try:
    verify_hwp(Imaging(Rotator()), [22.5])
except SystemExit:
    pass
else:
    raise AssertionError("a 5 deg pointing error must abort the run")
''')


# --------------------------------------------------------------------------- #
# Runner wiring
# --------------------------------------------------------------------------- #
def test_execute_night_refuses_pol_plan_with_hwp_off():
    """--hwp off on a plan that steps the HWP aborts before touching disk."""
    _run(f'''
from pathlib import Path
from obs_utils.night_plan import load_night_plan
from scripts.execute_night import _resolve_hwp

config = load_night_plan(Path({str(POL_PLAN)!r}))
angles, connect = _resolve_hwp(config, "auto")
assert angles and connect, "pol plan should auto-connect the rotator"

cal = load_night_plan(Path({str(CAL_PLAN)!r}))
assert _resolve_hwp(cal, "auto") == ([], False)
assert _resolve_hwp(cal, "on")[1] is True

try:
    _resolve_hwp(config, "off")
except SystemExit:
    pass
else:
    raise AssertionError("--hwp off must abort a plan that steps the HWP")
''')


# --------------------------------------------------------------------------- #
# Mount gate
# --------------------------------------------------------------------------- #
def _target(name, *, ra=None, dec=None, alt=None, az=None):
    """Duck-typed TargetPlan -- the mount gate only reads these five fields."""
    return SimpleNamespace(name=name, ra_hours=ra, dec_deg=dec,
                           alt_deg=alt, az_deg=az, frames=[])


def _pointed_config(*targets):
    return SimpleNamespace(
        calibration_before=[], calibration_after=[], calibration_stage="before",
        targets=list(targets),
    )


class _FakeMount:
    """Minimal PWI4 stand-in: records commands, reports the state it was given."""

    def __init__(self, *, connected=True, axes=(True, True), enables_ok=True):
        self.connected = connected
        self.axes = list(axes)
        self.enables_ok = enables_ok
        self.commands = []

    def status(self):
        # PWI4's Status exposes the axes both by name and as a list of the same
        # two objects (pwi4_client.Status: mount.axis = [axis0, axis1]).
        axis0, axis1 = (
            SimpleNamespace(is_enabled=a, position_degs=0.0) for a in self.axes
        )
        mount = SimpleNamespace(
            is_connected=self.connected,
            is_tracking=False,
            is_slewing=False,
            ra_j2000_hours=0.0,
            dec_j2000_degs=0.0,
            altitude_degs=45.0,
            azimuth_degs=180.0,
            axis0=axis0,
            axis1=axis1,
            axis=[axis0, axis1],
        )
        return SimpleNamespace(mount=mount)

    def mount_connect(self):
        self.commands.append("connect")
        self.connected = True

    def mount_enable(self, axis):
        self.commands.append(f"enable{axis}")
        if self.enables_ok:
            self.axes[axis] = True

    def mount_find_home(self):
        self.commands.append("home")


def test_plan_pointed_targets_reports_only_targets_with_coordinates():
    config = _pointed_config(
        _target("bias-only"),
        _target("HD14069", ra=2.283, dec=39.663),
        _target("TwilightFlat", alt=45.0, az=118.0),
    )
    assert plan_pointed_targets(config) == ["HD14069", "TwilightFlat"]
    assert plan_requires_mount(config) is True


def test_cal_only_plan_needs_no_mount():
    config = _pointed_config(_target("darks"), _target("bias"))
    assert plan_pointed_targets(config) == []
    assert plan_requires_mount(config) is False


def test_verify_mount_connects_enables_and_homes():
    pwi4 = _FakeMount(connected=False, axes=(False, False))
    verify_mount(pwi4, ["M42"])
    assert pwi4.commands == ["connect", "enable0", "enable1", "home"]


def test_verify_mount_skips_homing_when_asked():
    pwi4 = _FakeMount()
    verify_mount(pwi4, ["M42"], home=False)
    assert "home" not in pwi4.commands


def test_dead_dec_axis_aborts_and_is_named():
    """The known POLITE fault: axis1 never energizes. Must abort, not hang."""
    pwi4 = _FakeMount(axes=(True, False), enables_ok=False)
    with pytest.raises(SystemExit) as excinfo:
        verify_mount(pwi4, ["M42"], enable_timeout_s=0.0)
    msg = str(excinfo.value)
    assert "axis1" in msg and "DEC" in msg
    assert "--mount off --unpointed" in msg, "abort must name the salvage path"
    assert "home" not in pwi4.commands, "must not home a mount with a dead axis"


def test_dead_axis_abort_is_bounded_in_time():
    """obs_utils.mount.enable_motors polls forever; this gate must not."""
    pwi4 = _FakeMount(axes=(True, False), enables_ok=False)
    start = time.monotonic()
    with pytest.raises(SystemExit):
        verify_mount(pwi4, ["M42"], enable_timeout_s=0.5)
    assert time.monotonic() - start < 10.0


def test_skip_override_downgrades_dead_axis_to_warning():
    pwi4 = _FakeMount(axes=(True, False), enables_ok=False)
    verify_mount(pwi4, ["M42"], enable_timeout_s=0.0, skip=True)  # must not raise
    assert "home" not in pwi4.commands


def test_unreachable_pwi4_aborts_with_a_fix():
    class _Dead:
        def status(self):
            raise ConnectionRefusedError("PWI4 not running")

    with pytest.raises(SystemExit) as excinfo:
        verify_mount(_Dead(), ["M42"])
    assert "PWI4" in str(excinfo.value)


def test_resolve_mount_matches_the_shipped_plans():
    """--mount off must refuse a pointed plan unless --unpointed says so."""
    _run(f'''
from pathlib import Path
from obs_utils.night_plan import load_night_plan
from scripts.execute_night import _resolve_mount

pointed = load_night_plan(Path({str(POL_PLAN)!r}))
targets, connect = _resolve_mount(pointed, "auto", False)
assert targets and connect, "a plan with coordinates should auto-connect the mount"

cal = load_night_plan(Path({str(CAL_PLAN)!r}))
assert _resolve_mount(cal, "auto", False) == ([], False)
assert _resolve_mount(cal, "on", False)[1] is True

# Salvage mode: explicitly acknowledged, so it runs -- but without the mount.
targets, connect = _resolve_mount(pointed, "off", True)
assert targets and not connect

try:
    _resolve_mount(pointed, "off", False)
except SystemExit:
    pass
else:
    raise AssertionError("--mount off must abort a plan that names coordinates")
''')
