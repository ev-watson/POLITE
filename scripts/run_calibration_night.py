#!/usr/bin/env python3
"""Standalone NO-MOUNT calibration runner for detector-characterization nights.

Drives ONLY the QHY268M camera + ZWO EFW filter wheel via the fault-isolated
per-component connect helpers in ``obs_utils.interactive``. It never enables
motors, homes, or slews the mount (dead DEC: engaging DEC auto-disconnects the
mount), and it never touches the HWP rotator (calibration frames need none).
Everything downstream -- exposure, filter selection, FITS filenames, headers,
PolConfig sidecar -- is reused verbatim from the tested
``obs_utils.night_session`` path (``_run_frames``), with ``pwi4=None``.

On top of the salvage-runner base this adds the three things a *calibration*
night needs:

1. **Settings banner with hardware read-back** -- before any frame, the camera
   is configured from the plan's ``camera:`` block and then every setting is
   read BACK from the hardware (gain, offset, readout mode, binning, ROI,
   cooler setpoint, CCD temperature, cooler power, EFW names/position) and
   printed to the screen AND to a session log file. A commanded-vs-actual
   mismatch on gain/offset/readout aborts (override: ``--skip-settings-check``).
2. **Cooler stabilization gate** (``--setpoint`` override, ``--no-cooler-wait``
   to skip) -- polls CCD temperature + cooler power until the setpoint holds
   within ``--cooler-tol`` for ``--cooler-stable-s``, logging the approach
   curve. The setpoint is RE-ISSUED on every poll: some QHYCCD SDK builds only
   regulate while CONTROL_COOLER is refreshed periodically (the INDI/N.I.N.A
   drivers do the same), and July-9 never reached its setpoint. On timeout the
   operator chooses continue-at-achieved-T or abort; per-frame CCD-TEMP in the
   headers is authoritative for reduction either way.
3. **Per-invocation output subdirectory** -- FITS names for cal frames encode
   only date/type/exposure/index, so re-running a plan (e.g. the same dark
   ladder at three temperatures) would silently overwrite. Each run therefore
   writes into ``FITSDATA/<session>/<subdir>/``; the default subdir is the plan
   stem plus a ``T<setpoint>C`` tag, override with ``--subdir``.

    python scripts/run_calibration_night.py night_plans/20260717_darkcal.yaml            # dry-run
    python scripts/run_calibration_night.py night_plans/20260717_darkcal.yaml --run --setpoint -10

The dry-run touches no hardware: it prints the expanded plan and a duration
estimate so you can eyeball it before committing.
"""
from __future__ import annotations

import argparse
import logging
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from obs_utils.night_plan import NightPlanError, describe, load_night_plan
from obs_utils.night_session import (
    _apply_session_camera,
    _flush_pol_config_sidecar,
    _run_cal_frames,
    _run_frames,
    plan_total_frames,
)
from obs_utils.night_display import NightReporter, total_frame_count
from obs_utils import interactive

logger = logging.getLogger("calnight")

# Physical EFW carousel as installed 2026-07-10 (0-indexed ASCOM Position):
#   pos 0 = Clear, pos 1 = B, pos 2 = V, pos 3 = R, pos 4 = Dark.
# The wheel's OWN Names (ASCOM Remote EFW driver) are authoritative for
# name->position resolution in obs_utils.alpaca.set_filter_position, so they
# MUST match this order or every "Dark" request lands on the wrong slot.
INSTALLED_EFW_NAMES = ["Clear", "Photometric B", "Photometric V", "Photometric R", "Dark"]

# Rough per-frame non-exposure overhead (readout + download + save) used only
# for the dry-run duration estimate.
_FRAME_OVERHEAD_S = 2.5


# --------------------------------------------------------------------------- #
# EFW fail-closed check (carried over from the salvage runner)
# --------------------------------------------------------------------------- #
def _verify_filter_wheel(imaging, *, skip: bool) -> None:
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
        logger.warning("[efw] --skip-filter-check set: continuing DESPITE mismatch.")
        return
    raise SystemExit(msg)


# --------------------------------------------------------------------------- #
# Settings banner: read every setting BACK from the hardware and log it
# --------------------------------------------------------------------------- #
def _read(obj, attr):
    try:
        return getattr(obj, attr)
    except Exception:
        return None


def _report_settings(imaging, ctx, *, skip_check: bool) -> None:
    """Print + log camera/EFW settings read back from hardware; verify vs plan.

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

    logger.info("=" * 66)
    logger.info("CAMERA / EFW SETTINGS (read back from hardware)")
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
            logger.warning("--skip-settings-check set: continuing DESPITE mismatch.")
        else:
            raise SystemExit(msg + "\nAborting (override with --skip-settings-check).")
    else:
        logger.info("Settings verified: read-back matches plan (gain/offset/readout).")


# --------------------------------------------------------------------------- #
# Cooler stabilization gate
# --------------------------------------------------------------------------- #
def _wait_for_cooler(cam, setpoint_c: float, *, tol_c: float, stable_s: float,
                     timeout_s: float, poll_s: float, assume_yes: bool) -> float:
    """Block until CCD temperature holds ``setpoint_c`` +/- ``tol_c``.

    Re-issues the setpoint every poll: some QHYCCD SDK builds only regulate
    while CONTROL_COOLER is refreshed periodically (INDI/N.I.N.A do the same).
    Returns the achieved temperature (also on operator-approved timeout).
    """
    if not _read(cam, "CanSetCCDTemperature"):
        logger.warning("[cooler] camera reports CanSetCCDTemperature=False; skipping wait.")
        return float("nan")

    logger.info("[cooler] waiting for %+.1f C (tol %.2f C, hold %.0f s, timeout %.0f s)",
                setpoint_c, tol_c, stable_s, timeout_s)
    start = time.monotonic()
    stable_since = None
    while True:
        try:
            cam.SetCCDTemperature = float(setpoint_c)
        except Exception:
            logger.warning("[cooler] re-issuing setpoint failed", exc_info=True)
        t = _read(cam, "CCDTemperature")
        p = _read(cam, "CoolerPower")
        t = float(t) if t is not None else float("nan")
        elapsed = time.monotonic() - start
        logger.info("[cooler] T=%+6.2f C  target=%+.1f C  power=%s%%  elapsed=%4.0f s",
                    t, setpoint_c, "n/a" if p is None else f"{float(p):.0f}", elapsed)

        if abs(t - setpoint_c) <= tol_c:
            if stable_since is None:
                stable_since = time.monotonic()
            elif time.monotonic() - stable_since >= stable_s:
                logger.info("[cooler] STABLE at %+.2f C (power %s%%) after %.0f s",
                            t, "n/a" if p is None else f"{float(p):.0f}", elapsed)
                return t
        else:
            stable_since = None

        if elapsed > timeout_s:
            logger.warning(
                "[cooler] TIMEOUT: achieved %+.2f C (target %+.1f C, power %s%%). "
                "Per-frame CCD-TEMP is recorded, so data taken at the achieved "
                "temperature is still usable for the dark-vs-T fit.",
                t, setpoint_c, "n/a" if p is None else f"{float(p):.0f}")
            if assume_yes:
                logger.warning("[cooler] --yes set: continuing at achieved temperature.")
                return t
            ans = input("Continue at achieved temperature? [y/N] ").strip().lower()
            if ans in ("y", "yes"):
                return t
            raise SystemExit("Aborted: cooler did not reach setpoint.")
        time.sleep(poll_s)


# --------------------------------------------------------------------------- #
# Session paths / logging
# --------------------------------------------------------------------------- #
def _session_paths(config, subdir: str) -> tuple[Path, Path, str]:
    """Session base dir, per-invocation output dir, and the polite date string."""
    now = datetime.now(timezone.utc) if config.use_utc else datetime.now()
    if config.session_name and len(str(config.session_name)) == 8:
        ymd = str(config.session_name)
    else:
        ymd = now.strftime("%Y%m%d")
    base = Path(config.base_data_dir).expanduser().resolve() / ymd
    out_dir = base / subdir if subdir else base
    out_dir.mkdir(parents=True, exist_ok=True)
    polite_date = f"{ymd[:4]}-{ymd[4:6]}-{ymd[6:8]}"
    return base, out_dir, polite_date


def _default_subdir(plan_path: Path, setpoint: float | None) -> str:
    # HHMM stamp so re-running the same plan (same or different setpoint)
    # can never overwrite an earlier invocation's frames.
    stem = re.sub(r"^\d{8}_", "", plan_path.stem)
    stamp = datetime.now().strftime("%H%M")
    if setpoint is not None:
        return f"{stem}_T{setpoint:+g}C_{stamp}"
    return f"{stem}_{stamp}"


def _attach_log_file(base_dir: Path, subdir: str) -> Path:
    logs = base_dir / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%dT%H%M%S")
    log_path = logs / f"calnight_{stamp}_{subdir or 'root'}.log"
    fh = logging.FileHandler(log_path, encoding="utf-8")
    fh.setLevel(logging.INFO)
    fh.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s %(message)s"))
    logging.getLogger().addHandler(fh)
    logger.info("Session log file: %s", log_path)
    return log_path


# --------------------------------------------------------------------------- #
# Dry-run preview
# --------------------------------------------------------------------------- #
def _estimate_s(config) -> float:
    frames = list(config.calibration_before) + list(config.calibration_after)
    for t in config.targets:
        frames.extend(t.frames)
    return sum(f.count * (f.exposure_s + _FRAME_OVERHEAD_S) for f in frames)


def _describe(config, plan_path: Path, setpoint: float | None) -> None:
    print(describe(config))
    est = _estimate_s(config)
    sp = setpoint if setpoint is not None else config.capture_context.cooler_setpoint_c
    print(f"\ncooler setpoint for --run : {sp:+.1f} C"
          f"{'  (from --setpoint)' if setpoint is not None else '  (from plan camera: block)'}")
    print(f"output subdir             : {_default_subdir(plan_path, setpoint)}  (override: --subdir)")
    print(f"estimated capture time    : {est/60:.0f} min "
          f"(+ cooler stabilization, excluded)")
    print("\n(dry-run; pass --run to execute against the camera/EFW -- NO mount, NO HWP)")


# --------------------------------------------------------------------------- #
# Execute
# --------------------------------------------------------------------------- #
def run(config, plan_path: Path, args) -> int:
    from dataclasses import replace as dc_replace

    if args.setpoint is not None:
        config = dc_replace(
            config,
            capture_context=dc_replace(config.capture_context,
                                       cooler_setpoint_c=float(args.setpoint)),
        )
    ctx = config.capture_context

    subdir = args.subdir if args.subdir is not None else _default_subdir(plan_path, args.setpoint)
    base_dir, out_dir, polite_date = _session_paths(config, subdir)
    _attach_log_file(base_dir, subdir)
    logger.info("Plan: %s", plan_path)
    logger.info("Output directory: %s", out_dir)
    for line in describe(config).splitlines():
        logger.info("[plan] %s", line)

    # --- connect ONLY camera + wheel; never the mount, never the HWP ---------
    logger.info("[connect] camera ...")
    session = interactive.connect_camera(alpaca_config=config.startup.alpaca)
    logger.info("[connect] filter wheel ...")
    session = interactive.connect_filter_wheel(alpaca_config=config.startup.alpaca)
    imaging = session.imaging

    _verify_filter_wheel(imaging, skip=args.skip_filter_check)

    # --- apply plan settings, read them back, verify, and log ----------------
    _apply_session_camera(imaging, ctx)
    _report_settings(imaging, ctx, skip_check=args.skip_settings_check)

    # --- cooler stabilization gate -------------------------------------------
    if args.no_cooler_wait:
        logger.info("[cooler] --no-cooler-wait: capturing while cooler settles "
                    "(per-frame CCD-TEMP recorded in headers).")
    else:
        achieved = _wait_for_cooler(
            imaging.camera, ctx.cooler_setpoint_c,
            tol_c=args.cooler_tol, stable_s=args.cooler_stable_s,
            timeout_s=args.cooler_timeout, poll_s=args.cooler_poll,
            assume_yes=args.yes,
        )
        logger.info("[cooler] proceeding at %+.2f C", achieved)
        _report_settings(imaging, ctx, skip_check=True)  # re-log post-stabilization state

    # --- capture via the tested night_session path ---------------------------
    state = SimpleNamespace(imaging=imaging, pwi4=None, ntp_status=None)
    block_log: list = []
    session_id = config.session_name or base_dir.name

    def _flush() -> None:
        if config.capture_context is not None:
            _flush_pol_config_sidecar(
                out_dir, config.capture_context, session_id=session_id, block_log=block_log
            )

    stage = config.calibration_stage
    reporter = NightReporter(plan_total_frames(config), title=f"CALNIGHT {session_id}/{subdir}")
    with reporter:
        reporter.banner([
            ("session", session_id),
            ("output dir", out_dir),
            ("mode", "NO-MOUNT calibration (camera+EFW only)"),
            ("gain/offset/readout", f"{ctx.gain_setting}/{ctx.offset_setting}/{ctx.readout_mode}"),
            ("cooler setpoint", f"{ctx.cooler_setpoint_c:+.1f} C"),
            ("total frames", plan_total_frames(config)),
        ])
        if config.calibration_before and stage in ("before", "both"):
            _run_cal_frames(
                state, config, config.calibration_before, out_dir, polite_date,
                None, block_log, "before", reporter=reporter,
            )
            _flush()

        for target in config.targets:
            logger.info("Target: %s", target.name)
            reporter.start_block(target.name, subtitle=f"{total_frame_count(target.frames)} frames")
            _run_frames(
                imaging, None, target.frames, out_dir, target, config,
                ntp_status=None, date_str=polite_date,
                block_log=block_log, block_id=target.name, reporter=reporter,
            )
            _flush()

        if config.calibration_after and stage in ("after", "both"):
            _run_cal_frames(
                state, config, config.calibration_after, out_dir, polite_date,
                None, block_log, "after", reporter=reporter,
            )
            _flush()

        if config.capture_context is not None:
            _flush_pol_config_sidecar(
                out_dir, config.capture_context, session_id=session_id,
                block_log=block_log, final=True,
            )
        reporter.note("✓ Calibration block complete", style="moss")
    logger.info("Calibration block complete. Data in %s", out_dir)
    return 0


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("plan", help="Path to the calibration night-plan YAML")
    p.add_argument("--run", action="store_true", help="Execute (default is dry-run)")
    p.add_argument("--setpoint", type=float, default=None,
                   help="Override the plan's cooler_setpoint_c [C] for this invocation")
    p.add_argument("--subdir", default=None,
                   help="Output subdirectory under FITSDATA/<session>/ "
                        "(default: plan stem + T<setpoint>C tag; reruns MUST differ "
                        "or files overwrite)")
    p.add_argument("--no-cooler-wait", action="store_true",
                   help="Skip the cooler stabilization gate (twilight blocks: sky "
                        "is time-critical; CCD-TEMP is recorded per frame)")
    p.add_argument("--cooler-tol", type=float, default=0.5,
                   help="Stabilization tolerance [C] (default 0.5)")
    p.add_argument("--cooler-stable-s", type=float, default=120.0,
                   help="Required continuous in-tolerance time [s] (default 120)")
    p.add_argument("--cooler-timeout", type=float, default=1500.0,
                   help="Stabilization timeout [s] before prompting (default 1500)")
    p.add_argument("--cooler-poll", type=float, default=15.0,
                   help="Cooler poll interval [s] (default 15)")
    p.add_argument("--skip-filter-check", action="store_true",
                   help="Downgrade EFW name mismatch from abort to warning")
    p.add_argument("--skip-settings-check", action="store_true",
                   help="Downgrade commanded-vs-read-back mismatch from abort to warning")
    p.add_argument("--yes", action="store_true",
                   help="Non-interactive: continue at achieved temperature on cooler timeout")
    args = p.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    plan_path = Path(args.plan)
    try:
        config = load_night_plan(plan_path)
    except NightPlanError as exc:
        print(f"plan error: {exc}")
        return 2

    if not args.run:
        _describe(config, plan_path, args.setpoint)
        return 0

    return run(config, plan_path, args)


if __name__ == "__main__":
    raise SystemExit(main())
