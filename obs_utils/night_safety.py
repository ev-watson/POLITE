"""Fail-closed preflight gates shared by the night runners.

These are the safety checks that must run *before* a single frame is written,
and they must behave identically no matter which entry point drives the night
(``scripts/execute_night.py`` today, a control notebook cell tomorrow). They
live here so the two paths cannot diverge.

Every gate follows the same contract:

* it **reads state back from the hardware** rather than trusting the value that
  was commanded;
* a disagreement **aborts** by raising :class:`SystemExit` with an operator-
  readable fix, unless the caller passes an explicit ``skip=`` override;
* it never mutates a control loop it is only observing (in particular the
  cooler gate is strictly read-only -- the QHY SDK owns the cooler PID once the
  setpoint has been issued).

Gates provided here:

``verify_filter_wheel``  EFW name->slot resolution matches the installed carousel.
``report_settings``      camera/EFW/HWP settings banner + commanded-vs-read-back verify.
``cooler_gate``          cooler stabilization gate (read-only polling).
``plan_hwp_angles``      which HWP angles a loaded plan will command (may be empty).
``verify_hwp``           rotator present, and (optionally) a commanded move lands.
``plan_pointed_targets`` which targets a loaded plan will slew to (may be empty).
``verify_mount``         mount connected, both axes enabled, homed -- bounded in time.

Note the deliberate split on cooler policy. ``cooler_gate`` here is the
*operator-gated* variant: on timeout it offers continue-at-achieved-temperature,
because per-frame ``DET-TEMP`` is what reduction actually uses and a cal ladder
at -13 C is still good data. ``obs_utils.night_session.wait_for_cooler`` is the
*strict* variant used by the pointed-science path -- it raises ``TimeoutError``,
because a science frame at an unknown temperature is not worth keeping. Two
policies, two names, on purpose.

``verify_mount`` is written for a mount that is currently BROKEN. POLITE's DEC
drive (axis 1) does not engage, and ``obs_utils.mount.enable_motors`` waits for
both axes in an unbounded ``while True`` -- on this hardware that never returns.
The gate therefore does its own bounded enable and names the axis that failed,
so a dead drive produces an operator-readable abort in under a minute instead of
a hang. Nothing here is special-cased to the fault: when the drive is repaired
the same gate passes and the pointed path runs unchanged.
"""
from __future__ import annotations

import logging
import time
from typing import Any, Iterable, List, Optional, Sequence

logger = logging.getLogger(__name__)

# Physical EFW carousel as installed 2026-07-10 (0-indexed ASCOM Position):
#   pos 0 = Clear, pos 1 = B, pos 2 = V, pos 3 = R, pos 4 = Dark.
# The wheel's OWN Names (ASCOM Remote EFW driver) are authoritative for
# name->position resolution in obs_utils.alpaca.set_filter_position, so they
# MUST match this order or every "Dark" request lands on the wrong slot.
INSTALLED_EFW_NAMES = ["Clear", "Photometric B", "Photometric V", "Photometric R", "Dark"]

# Optec Pyxis Gen3 is open-loop: there is no encoder, so commanded-vs-reported
# angle disagreement is dominated by step quantization (~0.012 deg). A quarter
# of a degree is far above that floor while still catching a stalled, unhomed,
# or wrongly-geared rotator.
HWP_DEFAULT_TOL_DEG = 0.25

# Bounded replacement for obs_utils.mount.enable_motors' unbounded poll. The
# PlaneWave axes energize in a few seconds when healthy; a minute is generous
# and still fails fast on the dead DEC drive.
MOUNT_ENABLE_TIMEOUT_S = 60.0
MOUNT_CONNECT_TIMEOUT_S = 30.0

# PWI4 axis numbering: 0 = azimuth/RA, 1 = altitude/DEC.
_AXIS_LABELS = {0: "axis0 (RA/azimuth)", 1: "axis1 (DEC/altitude)"}


def _read(obj: Any, attr: str) -> Any:
    """Read one driver property, returning ``None`` if the driver refuses."""
    try:
        return getattr(obj, attr)
    except Exception:
        return None


# --------------------------------------------------------------------------- #
# EFW fail-closed check
# --------------------------------------------------------------------------- #
def verify_filter_wheel(imaging, *, skip: bool = False) -> None:
    """Fail-closed check that name->slot resolution matches installed hardware."""
    wheel = imaging.filter_wheel
    try:
        live = [str(n) for n in list(wheel.Names)]
    except Exception:
        live = []

    logger.info("[efw] installed truth   : %s", INSTALLED_EFW_NAMES)
    logger.info("[efw] driver Names      : %s", live or "(empty -> falls back to config)")
    logger.info("[efw] config filter_names: %s", list(imaging.filter_names or []))

    if not live:
        if list(imaging.filter_names or []) != INSTALLED_EFW_NAMES:
            raise SystemExit(
                "config filter_names do not match installed EFW; fix "
                "obs_utils/user_config.py before running."
            )
        logger.info("[efw] OK: resolving via config filter_names (driver Names empty).")
        return

    if live == INSTALLED_EFW_NAMES:
        logger.info("[efw] OK: driver Names match installed carousel.")
        return

    msg = (
        "EFW driver Names do NOT match the installed carousel.\n"
        f"    driver reports : {live}\n"
        f"    installed truth: {INSTALLED_EFW_NAMES}\n"
        "Name-based filter selection would land on the WRONG slot.\n"
        "FIX: set the filter names in ASCOM Remote's ZWO EFW driver to the\n"
        "installed order, restart the ASCOM Remote server, and rerun."
    )
    if skip:
        logger.warning("[efw] %s", msg)
        logger.warning("[efw] skip override set: continuing DESPITE mismatch.")
        return
    raise SystemExit(msg)


# --------------------------------------------------------------------------- #
# HWP (Optec Pyxis via Alpaca rotator)
# --------------------------------------------------------------------------- #
def _iter_plan_frames(config) -> Iterable[Any]:
    """Every FramePlan the run will actually execute, cal stage respected."""
    stage = config.calibration_stage
    if config.calibration_before and stage in ("before", "both"):
        yield from config.calibration_before
    for target in config.targets:
        yield from target.frames
    if config.calibration_after and stage in ("after", "both"):
        yield from config.calibration_after


def plan_hwp_angles(config) -> List[float]:
    """Sorted unique HWP angles [deg] the loaded plan will command.

    Empty when no brick steps the half-wave plate -- that is the signal a runner
    uses to decide whether the rotator needs connecting at all. Cal-only nights
    (bias/dark ladders, PTC sweeps) return ``[]`` and never touch the HWP.
    """
    angles = {
        float(f.hwp_angle_deg)
        for f in _iter_plan_frames(config)
        if getattr(f, "hwp_angle_deg", None) is not None
    }
    return sorted(angles)


def plan_requires_hwp(config) -> bool:
    """True when the plan steps the half-wave plate at least once."""
    return bool(plan_hwp_angles(config))


def verify_hwp(
    imaging,
    angles: Sequence[float],
    *,
    preflight: bool = True,
    tol_deg: float = HWP_DEFAULT_TOL_DEG,
    skip: bool = False,
) -> Optional[float]:
    """Fail-closed check that the HWP rotator is live and actually moves.

    ``angles`` are the plan's HWP angles (see :func:`plan_hwp_angles`). With
    ``preflight`` the rotator is commanded to the plan's *first* angle and the
    reported position is compared to it: this is physical motion, and it is the
    point -- an unhomed, stalled, or disconnected Pyxis is caught now rather
    than 40 frames into a modulation sequence, where every frame would carry a
    wrong ``HWPANG`` and the whole sequence would be unreducible.

    Returns the achieved angle [deg], or ``None`` when no preflight ran.
    """
    rotator = getattr(imaging, "rotator", None)
    if rotator is None:
        msg = (
            "Plan steps the HWP but no rotator is connected.\n"
            f"    plan angles: {[f'{a:g}' for a in angles]}\n"
            "Frames would be captured at an UNKNOWN half-wave plate angle and\n"
            "the modulation sequence would be unreducible.\n"
            "FIX: check that the Optec Pyxis is powered and answering on the\n"
            "ASCOM Remote server, and that AlpacaConfig.rotator_index is set."
        )
        if skip:
            logger.warning("[hwp] %s", msg)
            logger.warning("[hwp] skip override set: continuing WITHOUT a rotator.")
            return None
        raise SystemExit(msg)

    logger.info("[hwp] rotator position  : %s deg", _read(rotator, "Position"))
    logger.info("[hwp] rotator moving    : %s", _read(rotator, "IsMoving"))
    logger.info("[hwp] plan angles       : %s", [f"{a:g}" for a in angles] or "(none)")

    if not preflight or not angles:
        return None

    from .imaging import select_hwp_angle

    target = float(angles[0])
    logger.info("[hwp] preflight move -> %.3f deg (physical motion)", target)
    try:
        achieved = float(select_hwp_angle(imaging, target))
    except Exception as exc:
        msg = (
            f"HWP preflight move to {target:.3f} deg FAILED: {exc}\n"
            "The Pyxis Gen3 rejects moves until it has been homed once per\n"
            "power cycle. FIX: home the rotator (ASCOM driver or the lab\n"
            "notebook's s.home_hwp()), then rerun."
        )
        if skip:
            logger.warning("[hwp] %s", msg)
            logger.warning("[hwp] skip override set: continuing DESPITE failure.")
            return None
        raise SystemExit(msg) from exc

    err = abs(achieved - target)
    logger.info("[hwp] commanded %.3f deg -> reported %.3f deg (|err| = %.4f deg)",
                target, achieved, err)
    if err > tol_deg:
        msg = (
            f"HWP commanded {target:.3f} deg but reports {achieved:.3f} deg "
            f"(|err| = {err:.3f} deg > tol {tol_deg:g} deg).\n"
            "The Pyxis Gen3 is OPEN-LOOP -- a disagreement this large is not\n"
            "step quantization (~0.012 deg); it means the stage did not reach\n"
            "the commanded angle. Every HWPANG in the headers would be wrong."
        )
        if skip:
            logger.warning("[hwp] %s", msg)
            logger.warning("[hwp] skip override set: continuing DESPITE mismatch.")
            return achieved
        raise SystemExit(msg + "\nAborting (override with --skip-hwp-check).")

    logger.info("[hwp] OK: rotator reaches commanded angle within %g deg.", tol_deg)
    return achieved


# --------------------------------------------------------------------------- #
# Mount (PlaneWave via PWI4)
# --------------------------------------------------------------------------- #
def plan_pointed_targets(config) -> List[str]:
    """Names of the plan's targets that specify a sky position to slew to.

    Empty for a cal-only plan (bias/dark ladders, PTC sweeps, dome flats) --
    that is the signal a runner uses to decide whether the mount is needed at
    all. A target counts as pointed when it gives either an equatorial pair
    (``ra``/``dec``) or a horizontal pair (``alt``/``az``); the half-specified
    case is rejected downstream by ``night_session._slew_to_target``.
    """
    names = []
    for target in config.targets:
        equatorial = target.ra_hours is not None and target.dec_deg is not None
        horizontal = target.alt_deg is not None and target.az_deg is not None
        if equatorial or horizontal:
            names.append(target.name)
    return names


def plan_requires_mount(config) -> bool:
    """True when the plan names at least one sky position to slew to."""
    return bool(plan_pointed_targets(config))


def _axis_enabled(status, axis: int) -> Optional[bool]:
    try:
        return bool(status.mount.axis[axis].is_enabled)
    except Exception:
        return None


def _enable_axes(pwi4, *, timeout_s: float, poll_s: float = 1.0) -> List[int]:
    """Enable both mount axes, bounded in time. Returns the axes still down.

    ``obs_utils.mount.enable_motors`` polls forever, which on POLITE's dead DEC
    drive means the runner hangs with no message. This does the same work with
    a deadline and reports *which* axis never came up.
    """
    for axis in (0, 1):
        if _axis_enabled(pwi4.status(), axis) is not True:
            logger.info("[mount] enabling %s ...", _AXIS_LABELS[axis])
            try:
                pwi4.mount_enable(axis)
            except Exception as exc:
                logger.warning("[mount] %s enable request failed: %r",
                               _AXIS_LABELS[axis], exc)

    deadline = time.monotonic() + timeout_s
    while True:
        status = pwi4.status()
        down = [a for a in (0, 1) if _axis_enabled(status, a) is not True]
        if not down:
            return []
        if time.monotonic() >= deadline:
            return down
        time.sleep(poll_s)


def verify_mount(
    pwi4,
    targets: Sequence[str] = (),
    *,
    home: bool = True,
    connect_timeout_s: float = MOUNT_CONNECT_TIMEOUT_S,
    enable_timeout_s: float = MOUNT_ENABLE_TIMEOUT_S,
    skip: bool = False,
) -> None:
    """Fail-closed check that the mount is connected, energized, and homed.

    ``targets`` are the plan's pointed target names (see
    :func:`plan_pointed_targets`), used only to make the abort message say what
    the run was about to do. Every step is bounded in time: on a mount whose
    axis never energizes this returns an abort within
    ``connect_timeout_s + enable_timeout_s`` rather than blocking the night.

    Homing is physical motion across the full range of both axes. It is what
    makes the pointing model meaningful, so it is on by default; pass
    ``home=False`` when the mount was homed earlier in the same power cycle.
    """
    from .mount import home_mount

    what = f" for {', '.join(targets)}" if targets else ""

    # --- connect ---------------------------------------------------------- #
    try:
        connected = bool(pwi4.status().mount.is_connected)
    except Exception as exc:
        msg = (
            f"Mount status unreachable{what}: {exc!r}\n"
            "PWI4 is not answering. FIX: start PlaneWave PWI4 on the "
            "observatory PC, confirm the mount is powered, and check\n"
            "Pwi4Config.host/port in obs_utils/user_config.py."
        )
        if skip:
            logger.warning("[mount] %s", msg)
            logger.warning("[mount] skip override set: continuing WITHOUT the mount.")
            return
        raise SystemExit(msg) from exc

    if not connected:
        logger.info("[mount] connecting ...")
        pwi4.mount_connect()
        deadline = time.monotonic() + connect_timeout_s
        while not bool(pwi4.status().mount.is_connected):
            if time.monotonic() >= deadline:
                msg = (
                    f"Mount did not connect within {connect_timeout_s:.0f} s{what}.\n"
                    "FIX: check the mount power and the PWI4 'Connect' state, "
                    "then rerun."
                )
                if skip:
                    logger.warning("[mount] %s", msg)
                    logger.warning("[mount] skip override set: continuing WITHOUT the mount.")
                    return
                raise SystemExit(msg)
            time.sleep(1.0)

    # --- energize both axes (bounded) -------------------------------------- #
    down = _enable_axes(pwi4, timeout_s=enable_timeout_s)
    if down:
        names = ", ".join(_AXIS_LABELS[a] for a in down)
        msg = (
            f"Mount {names} did not energize within {enable_timeout_s:.0f} s{what}.\n"
            "The run would slew with a dead axis, so every frame would be "
            "pointed somewhere other than\nthe requested coordinates -- or the "
            "mount would drop its connection mid-sequence.\n"
            "FIX: clear the drive fault in PWI4 and confirm both axes enable "
            "there before rerunning."
        )
        if 1 in down:
            msg += (
                "\nNOTE: axis1 is POLITE's known-dead DEC drive. If it has not "
                "been repaired yet, run\nthis plan without pointing "
                "(--mount off --unpointed) and accept that the frames carry no "
                "sky position."
            )
        if skip:
            logger.warning("[mount] %s", msg)
            logger.warning("[mount] skip override set: continuing DESPITE dead axis.")
        else:
            raise SystemExit(msg + "\nAborting (override with --skip-mount-check).")

    # --- home --------------------------------------------------------------- #
    if home and not down:
        logger.info("[mount] homing (physical motion on both axes) ...")
        home_mount(pwi4)

    # --- read back ----------------------------------------------------------- #
    status = pwi4.status()
    mount = status.mount
    logger.info("[mount] connected        : %s", _read(mount, "is_connected"))
    logger.info("[mount] axis0 enabled    : %s", _axis_enabled(status, 0))
    logger.info("[mount] axis1 enabled    : %s", _axis_enabled(status, 1))
    logger.info("[mount] tracking         : %s", _read(mount, "is_tracking"))
    logger.info("[mount] slewing          : %s", _read(mount, "is_slewing"))
    logger.info("[mount] RA/Dec J2000     : %s h, %s deg",
                _read(mount, "ra_j2000_hours"), _read(mount, "dec_j2000_degs"))
    logger.info("[mount] alt/az           : %s deg, %s deg",
                _read(mount, "altitude_degs"), _read(mount, "azimuth_degs"))
    logger.info("[mount] plan targets     : %s", list(targets) or "(none)")
    if not down:
        logger.info("[mount] OK: connected, both axes enabled%s.",
                    ", homed" if home else "")


# --------------------------------------------------------------------------- #
# Settings banner: read every setting BACK from the hardware and log it
# --------------------------------------------------------------------------- #
def report_settings(imaging, ctx, *, skip_check: bool = False) -> None:
    """Print + log camera/EFW/HWP settings read back from hardware; verify vs plan.

    Every line goes through ``logger`` so it lands on the console AND in the
    session log file. Gain/offset/readout-mode read-back that disagrees with
    the plan's ``camera:`` block aborts unless ``skip_check``.
    """
    cam = imaging.camera
    mode = _read(cam, "ReadoutMode")
    modes = _read(cam, "ReadoutModes")
    mode_name = None
    if modes is not None and mode is not None:
        try:
            mode_name = str(list(modes)[int(mode)])
        except Exception:
            mode_name = None

    rows = [
        ("camera",            _read(cam, "Name")),
        ("sensor",            _read(cam, "SensorName")),
        ("sensor size",       f"{_read(cam, 'CameraXSize')} x {_read(cam, 'CameraYSize')}"),
        ("binning",           f"{_read(cam, 'BinX')} x {_read(cam, 'BinY')}"),
        ("ROI start",         f"({_read(cam, 'StartX')}, {_read(cam, 'StartY')})"),
        ("ROI size",          f"{_read(cam, 'NumX')} x {_read(cam, 'NumY')}"),
        ("gain [read-back]",  _read(cam, "Gain")),
        ("offset [read-back]", _read(cam, "Offset")),
        ("readout [read-back]", f"{mode}" + (f" ({mode_name})" if mode_name else "")),
        ("cooler setpoint",   _read(cam, "SetCCDTemperature")),
        ("CCD temperature",   _read(cam, "CCDTemperature")),
        ("cooler power [%]",  _read(cam, "CoolerPower")),
        ("cooler on",         _read(cam, "CoolerOn")),
    ]
    wheel = imaging.filter_wheel
    if wheel is not None:
        rows.append(("EFW position", _read(wheel, "Position")))
        try:
            rows.append(("EFW names", ", ".join(str(n) for n in wheel.Names)))
        except Exception:
            rows.append(("EFW names", None))

    rotator = getattr(imaging, "rotator", None)
    if rotator is not None:
        rows.append(("HWP position [deg]", _read(rotator, "Position")))
        rows.append(("HWP mech. pos [deg]", _read(rotator, "MechanicalPosition")))
        rows.append(("HWP moving", _read(rotator, "IsMoving")))
        rows.append(("HWP step size [deg]", _read(rotator, "StepSize")))

    logger.info("=" * 66)
    logger.info("CAMERA / EFW / HWP SETTINGS (read back from hardware)")
    for label, value in rows:
        logger.info("  %-22s %s", label, "n/a" if value is None else value)
    logger.info("  %-22s gain=%s offset=%s readout=%s (%s) cooler=%+.1f C",
                "plan commanded:", ctx.gain_setting, ctx.offset_setting,
                ctx.readout_mode, ctx.readout_mode_name, ctx.cooler_setpoint_c)
    logger.info("=" * 66)

    mismatches = []
    for label, commanded, actual in (
        ("gain", ctx.gain_setting, _read(cam, "Gain")),
        ("offset", ctx.offset_setting, _read(cam, "Offset")),
        ("readout_mode", ctx.readout_mode, mode),
    ):
        if actual is not None and int(actual) != int(commanded):
            mismatches.append(f"{label}: commanded {commanded}, camera reports {actual}")

    if mismatches:
        msg = "Commanded vs read-back settings MISMATCH:\n    " + "\n    ".join(mismatches)
        if skip_check:
            logger.warning("%s", msg)
            logger.warning("skip override set: continuing DESPITE mismatch.")
        else:
            raise SystemExit(msg + "\nAborting (override with --skip-settings-check).")
    else:
        logger.info("Settings verified: read-back matches plan (gain/offset/readout).")


# --------------------------------------------------------------------------- #
# Cooler stabilization gate
# --------------------------------------------------------------------------- #
def cooler_gate(cam, setpoint_c: float, *, tol_c: float, stable_s: float,
                timeout_s: float, poll_s: float, assume_yes: bool) -> float:
    """Block until CCD temperature holds ``setpoint_c`` +/- ``tol_c``.

    The setpoint was applied once by ``_apply_session_camera``.  The QHY SDK
    regulates cooler power internally; this function is read-only so polling
    cannot reset or perturb that controller. Returns the achieved temperature
    (also on operator-approved timeout).

    This is the operator-gated policy -- see the module docstring for why it
    differs from ``obs_utils.night_session.wait_for_cooler``.
    """
    if not _read(cam, "CanSetCCDTemperature"):
        logger.warning("[cooler] camera reports CanSetCCDTemperature=False; skipping wait.")
        return float("nan")

    logger.info("[cooler] waiting for %+.1f C (tol %.2f C, hold %.0f s, timeout %.0f s)",
                setpoint_c, tol_c, stable_s, timeout_s)
    start = time.monotonic()
    stable_since = None
    while True:
        t = _read(cam, "CCDTemperature")
        p = _read(cam, "CoolerPower")
        t = float(t) if t is not None else float("nan")
        elapsed = time.monotonic() - start
        logger.info("[cooler] T=%+6.2f C  target=%+.1f C  power=%s%%  elapsed=%4.0f s",
                    t, setpoint_c, "n/a" if p is None else f"{float(p):.0f}", elapsed)

        if abs(t - setpoint_c) <= tol_c:
            if stable_since is None:
                stable_since = time.monotonic()
                if stable_s <= 0:
                    logger.info("[cooler] IN TOLERANCE at %+.2f C", t)
                    return t
            elif time.monotonic() - stable_since >= stable_s:
                logger.info("[cooler] STABLE at %+.2f C (power %s%%) after %.0f s",
                            t, "n/a" if p is None else f"{float(p):.0f}", elapsed)
                return t
        else:
            stable_since = None

        if elapsed > timeout_s:
            logger.warning(
                "[cooler] TIMEOUT: achieved %+.2f C (target %+.1f C, power %s%%). "
                "Per-frame DET-TEMP is recorded, so data taken at the achieved "
                "temperature is still usable for the dark-vs-T fit.",
                t, setpoint_c, "n/a" if p is None else f"{float(p):.0f}")
            if assume_yes:
                logger.warning("[cooler] assume-yes set: continuing at achieved temperature.")
                return t
            ans = input("Continue at achieved temperature? [y/N] ").strip().lower()
            if ans in ("y", "yes"):
                return t
            raise SystemExit("Aborted: cooler did not reach setpoint.")
        time.sleep(poll_s)
