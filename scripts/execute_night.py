#!/usr/bin/env python3
"""Generic night runner for the POLITE camera + filter wheel + HWP + mount.

This is the single ``--run`` entry point for any plan -- calibration ladders,
PTC/twilight sweeps, bias/dark bricks, and pointed polarimetry sequences alike.
It drives the QHY268M camera, the ZWO EFW filter wheel, and (only when the plan
needs them) the Optec Pyxis half-wave plate and the PlaneWave mount, via the
fault-isolated per-component connect helpers in ``obs_utils.interactive``.
Everything downstream -- slewing, exposure, filter/HWP selection, FITS
filenames, headers, PolConfig sidecar -- is reused verbatim from the tested
``obs_utils.night_session`` path (``_slew_to_target`` / ``_run_frames`` /
``_run_cal_frames``).

The fail-closed safety gates live in ``obs_utils.night_safety`` so that any
future runner (or a control-notebook cell) gets exactly the same checks:

1. **Settings banner with hardware read-back** -- before any frame, the camera
   is configured from the plan's ``camera:`` block and then every setting is
   read BACK from the hardware (gain, offset, readout mode, binning, ROI,
   cooler setpoint, CCD temperature, cooler power, EFW names/position, HWP
   position when connected) and printed to the screen AND to a session log
   file. A commanded-vs-actual mismatch on gain/offset/readout aborts
   (override: ``--skip-settings-check``).
2. **Cooler stabilization gate** (``--setpoint`` override, ``--no-cooler-wait``
   to skip) -- polls CCD temperature + cooler power until the setpoint holds
   within ``--cooler-tol`` for ``--cooler-stable-s``, logging the approach
   curve. The QHY SDK owns the cooler PID after the setpoint is issued once;
   the gate only observes temperature and power, so it cannot perturb that
   control loop. On timeout the operator chooses continue-at-achieved-T or
   abort; per-frame CCD-TEMP in the headers is authoritative for reduction
   either way.
3. **Conditional HWP gate** (``--hwp auto|on|off``) -- the plan is scanned for
   bricks that step the half-wave plate (``pol_seq``, ``pol_flat``, or any
   frame carrying ``hwp_angle_deg``). If any do, the Pyxis rotator is connected
   and a preflight move to the plan's first angle verifies that the commanded
   angle is actually reached (the Gen3 is open-loop, so this is the only way to
   catch an unhomed or stalled stage before the sequence is unreducible).
   Cal-only plans never touch the rotator. See ``--skip-hwp-check`` /
   ``--no-hwp-preflight``.
4. **Conditional mount gate** (``--mount auto|on|off``) -- the plan is scanned
   for targets that name a sky position (``ra``/``dec`` or ``alt``/``az``). If
   any do, the PWI4 client is created and the mount is connected, both axes are
   energized *with a deadline*, and the mount is homed before the first slew.
   Cal-only plans never touch it. A plan that names coordinates cannot be run
   without pointing by accident: ``--mount off`` aborts unless the operator also
   passes ``--unpointed`` to say so out loud. See ``--skip-mount-check`` /
   ``--no-mount-home`` / ``--park-on-finish``.
5. **Per-invocation output subdirectory** -- FITS names for cal frames encode
   only date/type/exposure/index, so re-running a plan (e.g. the same dark
   ladder at three temperatures) would silently overwrite. Each run therefore
   writes into ``FITSDATA/<session>/<subdir>/``; the default subdir is the plan
   stem plus a ``T<setpoint>C`` tag, override with ``--subdir``.

    python scripts/execute_night.py night_plans/20260717_darkcal.yaml            # dry-run
    python scripts/execute_night.py night_plans/20260717_darkcal.yaml --run --setpoint -10

The dry-run touches no hardware: it prints the expanded plan, the HWP angles the
plan will command, what the mount will do, and a duration estimate so you can
eyeball it before committing.

MOUNT STATUS: POLITE's DEC drive (PWI4 axis 1) does not currently engage. The
pointing path here is complete and wired, but it has NOT been exercised against
a working mount -- treat its first pointed night as commissioning. Nothing is
special-cased to the fault: ``night_safety.verify_mount`` simply refuses to slew
an axis that will not energize, and names axis1 when that is the one that failed.
When the drive is repaired the same gate passes and pointed science runs with no
code change. Until then, a plan with coordinates runs unpointed only via the
explicit ``--mount off --unpointed`` pair, and those frames carry no sky
position -- the 2026-07-09 salvage night is what that looks like in the data.
"""
from __future__ import annotations

import argparse
import logging
import re
import sys
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
    _slew_to_target,
    plan_estimated_duration_s,
    plan_total_frames,
)
from obs_utils.night_display import NightReporter, total_frame_count
from obs_utils import interactive
from obs_utils.night_safety import (
    HWP_DEFAULT_TOL_DEG,
    cooler_gate,
    plan_hwp_angles,
    plan_pointed_targets,
    report_settings,
    verify_filter_wheel,
    verify_hwp,
    verify_mount,
)
from obs_utils.timing import acquire_session_timing_snapshot

logger = logging.getLogger("execnight")

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
    log_path = logs / f"execnight_{stamp}_{subdir or 'root'}.log"
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
    return plan_estimated_duration_s(config)


def _describe(config, plan_path: Path, setpoint: float | None, hwp_mode: str,
              mount_mode: str, unpointed: bool) -> None:
    print(describe(config))
    est = _estimate_s(config)
    sp = setpoint if setpoint is not None else config.capture_context.cooler_setpoint_c
    angles = plan_hwp_angles(config)
    pointed = plan_pointed_targets(config)
    print(f"\ncooler setpoint for --run : {sp:+.1f} C"
          f"{'  (from --setpoint)' if setpoint is not None else '  (from plan camera: block)'}")
    print(f"output subdir             : {_default_subdir(plan_path, setpoint)}  (override: --subdir)")
    print(f"estimated capture time    : {est/60:.0f} min "
          f"(+ cooler stabilization, excluded)")
    if angles:
        print(f"HWP angles commanded      : {', '.join(f'{a:g}' for a in angles)} deg "
              f"({len(angles)} unique)")
        print(f"HWP rotator for --run     : "
              f"{'DISABLED by --hwp off (run would ABORT)' if hwp_mode == 'off' else 'will be connected'}")
    else:
        print("HWP angles commanded      : none (plan does not step the half-wave plate)")
        print(f"HWP rotator for --run     : "
              f"{'connected (--hwp on)' if hwp_mode == 'on' else 'not connected'}")

    if pointed:
        print(f"pointed targets           : {', '.join(pointed)} ({len(pointed)})")
        if mount_mode == "off" and unpointed:
            print("mount for --run           : NOT connected (--mount off --unpointed);"
                  " frames get NO sky position")
        elif mount_mode == "off":
            print("mount for --run           : DISABLED by --mount off (run would ABORT;"
                  " add --unpointed to capture anyway)")
        else:
            print("mount for --run           : will connect, energize both axes, home, "
                  "and slew per target")
    else:
        print("pointed targets           : none (plan names no sky position)")
        print(f"mount for --run           : "
              f"{'connected (--mount on)' if mount_mode == 'on' else 'not connected'}")

    devices = "camera/EFW" + ("/HWP" if angles or hwp_mode == "on" else "")
    mount_note = ("mount" if (pointed and mount_mode != "off") or mount_mode == "on"
                  else "NO mount")
    print(f"\n(dry-run; pass --run to execute against the {devices} -- {mount_note})")


# --------------------------------------------------------------------------- #
# Execute
# --------------------------------------------------------------------------- #
def _resolve_hwp(config, hwp_mode: str) -> tuple[list[float], bool]:
    """Plan HWP angles + whether to connect the rotator. Fail-closed on conflict.

    Runs before any directory, log file, or hardware connection is created, so a
    plan/flag conflict costs nothing and leaves no artifacts behind.
    """
    angles = plan_hwp_angles(config)
    if angles and hwp_mode == "off":
        raise SystemExit(
            "Plan steps the HWP at "
            f"{', '.join(f'{a:g}' for a in angles)} deg but --hwp off was "
            "given.\nFrames would be captured at an unknown half-wave plate "
            "angle. Drop --hwp off, or run a plan without pol bricks."
        )
    connect = bool(angles) if hwp_mode == "auto" else (hwp_mode == "on")
    return angles, connect


def _resolve_mount(config, mount_mode: str, unpointed: bool) -> tuple[list[str], bool]:
    """Plan pointed targets + whether to connect the mount. Fail-closed on conflict.

    Runs before any directory, log file, or hardware connection is created, so a
    plan/flag conflict costs nothing and leaves no artifacts behind.

    Refusing ``--mount off`` on a plan that names coordinates is the same rule
    the HWP gate applies: capturing without pointing is a legitimate mode (it is
    how the 2026-07-09 salvage night ran), but it must be *chosen*, not defaulted
    into, because nothing in the resulting FITS records that the telescope was
    never aimed at the object the plan names.
    """
    pointed = plan_pointed_targets(config)
    if pointed and mount_mode == "off" and not unpointed:
        raise SystemExit(
            f"Plan points at {', '.join(pointed)} but --mount off was given.\n"
            "Frames would be captured wherever the telescope happens to sit, "
            "with no sky position\nrecorded in the headers -- the object named "
            "in the plan would simply not be in them.\n"
            "Add --unpointed to capture anyway (salvage mode), or drop "
            "--mount off to slew."
        )
    connect = bool(pointed) if mount_mode == "auto" else (mount_mode == "on")
    return pointed, connect


def run(config, plan_path: Path, args) -> int:
    from dataclasses import replace as dc_replace

    hwp_angles, connect_hwp = _resolve_hwp(config, args.hwp)
    pointed_targets, connect_mount = _resolve_mount(config, args.mount, args.unpointed)

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

    # --- connect camera + wheel (+ HWP only if the plan needs it); never the mount ---
    logger.info("[connect] camera ...")
    session = interactive.connect_camera(alpaca_config=config.startup.alpaca)
    logger.info("[connect] filter wheel ...")
    session = interactive.connect_filter_wheel(alpaca_config=config.startup.alpaca)
    if connect_hwp:
        logger.info("[connect] HWP rotator (plan commands %d angle(s)) ...", len(hwp_angles))
        try:
            session = interactive.connect_rotator(alpaca_config=config.startup.alpaca)
        except Exception as exc:
            if not args.skip_hwp_check:
                raise SystemExit(
                    f"HWP rotator connect FAILED: {exc}\n"
                    "The plan steps the half-wave plate, so the run cannot "
                    "proceed without it.\nFIX: power the Optec Pyxis, confirm "
                    "the ASCOM Remote server exposes it, and check "
                    "AlpacaConfig.rotator_index.\n"
                    "(override with --skip-hwp-check to capture WITHOUT the HWP)"
                ) from exc
            logger.warning("[hwp] connect failed (%s); --skip-hwp-check set, continuing.", exc)
    else:
        logger.info("[connect] HWP rotator: not needed (plan commands no HWP angles).")

    if connect_mount:
        logger.info("[connect] mount (plan points at %d target(s)) ...", len(pointed_targets))
        try:
            session = interactive.connect_mount(pwi4_config=config.startup.pwi4)
        except Exception as exc:
            if not args.skip_mount_check:
                raise SystemExit(
                    f"PWI4 client creation FAILED: {exc}\n"
                    "The plan names sky positions, so the run cannot proceed "
                    "without the mount.\nFIX: start PlaneWave PWI4 on the "
                    "observatory PC and check Pwi4Config.host/port.\n"
                    "(override with --skip-mount-check, or capture unpointed "
                    "with --mount off --unpointed)"
                ) from exc
            logger.warning("[mount] connect failed (%s); --skip-mount-check set, continuing.", exc)
    elif pointed_targets:
        logger.warning(
            "[mount] UNPOINTED run: plan names %s but --mount off --unpointed was "
            "given. Frames carry NO sky position.", ", ".join(pointed_targets),
        )
    else:
        logger.info("[connect] mount: not needed (plan names no sky position).")

    imaging = session.imaging
    pwi4 = session.pwi4 if connect_mount else None

    verify_filter_wheel(imaging, skip=args.skip_filter_check)
    if connect_hwp:
        verify_hwp(
            imaging, hwp_angles,
            preflight=not args.no_hwp_preflight,
            tol_deg=args.hwp_tol,
            skip=args.skip_hwp_check,
        )
    if pwi4 is not None:
        verify_mount(
            pwi4, pointed_targets,
            home=not args.no_mount_home,
            skip=args.skip_mount_check,
        )

    # Match the normal night runner's per-session timing provenance. DATE-OBS
    # itself comes from the QHY server per exposure; this snapshot documents
    # Windows clock discipline on every FITS.
    ntp_status = acquire_session_timing_snapshot(config.startup.timing)
    logger.info(
        "Timing snapshot: %s TIMEUNC=%.3f s ref=%s offset=%s sync_age=%s",
        ntp_status.timesrc, ntp_status.timeunc_s, ntp_status.timeref or "n/a",
        ntp_status.ntpoffs_s, ntp_status.ntpage_s,
    )
    for warning in ntp_status.warnings:
        logger.warning("Timing: %s", warning)

    # --- apply plan settings, read them back, verify, and log ----------------
    ctx = _apply_session_camera(imaging, ctx)
    config = dc_replace(config, capture_context=ctx)
    report_settings(imaging, ctx, skip_check=args.skip_settings_check)

    # --- cooler stabilization gate -------------------------------------------
    if args.no_cooler_wait:
        logger.info("[cooler] --no-cooler-wait: capturing while cooler settles "
                    "(per-frame CCD-TEMP recorded in headers).")
    else:
        achieved = cooler_gate(
            imaging.camera, ctx.cooler_setpoint_c,
            tol_c=args.cooler_tol, stable_s=args.cooler_stable_s,
            timeout_s=args.cooler_timeout, poll_s=args.cooler_poll,
            assume_yes=args.yes,
        )
        logger.info("[cooler] proceeding at %+.2f C", achieved)
        report_settings(imaging, ctx, skip_check=True)  # re-log post-stabilization state

    # --- capture via the tested night_session path ---------------------------
    state = SimpleNamespace(imaging=imaging, pwi4=pwi4, ntp_status=ntp_status)
    block_log: list = []
    session_id = config.session_name or base_dir.name

    def _flush() -> None:
        if config.capture_context is not None:
            _flush_pol_config_sidecar(
                out_dir, config.capture_context, session_id=session_id, block_log=block_log
            )

    stage = config.calibration_stage
    reporter = NightReporter(
        plan_total_frames(config),
        title=f"NIGHT {session_id}/{subdir}",
        estimated_duration_s=plan_estimated_duration_s(config),
    )
    with reporter:
        devices = "camera+EFW" + ("+HWP" if connect_hwp else "") + ("+mount" if pwi4 else "")
        banner_rows = [
            ("session", session_id),
            ("output dir", out_dir),
            ("mode", f"{'POINTED' if pwi4 else 'NO-MOUNT'} run ({devices})"),
            ("gain/offset/readout", f"{ctx.gain_setting}/{ctx.offset_setting}/{ctx.readout_mode}"),
            ("cooler setpoint", f"{ctx.cooler_setpoint_c:+.1f} C"),
        ]
        if hwp_angles:
            banner_rows.append(
                ("HWP angles", f"{len(hwp_angles)} unique: "
                               f"{', '.join(f'{a:g}' for a in hwp_angles)} deg")
            )
        if pointed_targets:
            banner_rows.append(
                ("pointing", f"{len(pointed_targets)} target(s): "
                             f"{', '.join(pointed_targets)}" if pwi4
                             else f"NONE -- {len(pointed_targets)} target(s) captured UNPOINTED")
            )
        banner_rows.append(("total frames", plan_total_frames(config)))
        reporter.banner(banner_rows)
        if config.calibration_before and stage in ("before", "both"):
            _run_cal_frames(
                state, config, config.calibration_before, out_dir, polite_date,
                None, block_log, "before", reporter=reporter,
            )
            _flush()

        for target in config.targets:
            logger.info("Target: %s", target.name)
            reporter.start_block(target.name, subtitle=f"{total_frame_count(target.frames)} frames")
            if pwi4 is not None:
                _slew_to_target(pwi4, target, limits=config.startup.slew_limits)
            _run_frames(
                imaging, pwi4, target.frames, out_dir, target, config,
                ntp_status=ntp_status, date_str=polite_date,
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
        reporter.note("✓ Night run complete", style="moss")

    # Parking is deliberate, never automatic: it is the one mount action that
    # happens after the data is safe, so a failure here must not look like a
    # failed night.
    if pwi4 is not None and args.park_on_finish:
        logger.info("[mount] parking ...")
        try:
            pwi4.mount_park()
        except Exception:
            logger.exception("[mount] park FAILED -- park it by hand in PWI4.")

    logger.info("Night run complete. Data in %s", out_dir)
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
    p.add_argument("--cooler-tol", type=float, default=0.2,
                   help="Stabilization tolerance [C] (default 0.2)")
    p.add_argument("--cooler-stable-s", type=float, default=30.0,
                   help="Required continuous in-tolerance time [s] (default 30)")
    p.add_argument("--cooler-timeout", type=float, default=600.0,
                   help="Stabilization timeout [s] before prompting (default 600)")
    p.add_argument("--cooler-poll", type=float, default=15.0,
                   help="Cooler poll interval [s] (default 15)")
    p.add_argument("--hwp", choices=("auto", "on", "off"), default="auto",
                   help="Half-wave plate: 'auto' connects the Pyxis rotator only "
                        "when the plan steps it (default), 'on' always connects "
                        "(e.g. to park the HWP), 'off' refuses to run a plan that "
                        "needs it")
    p.add_argument("--hwp-tol", type=float, default=HWP_DEFAULT_TOL_DEG,
                   help=f"HWP commanded-vs-reported tolerance [deg] "
                        f"(default {HWP_DEFAULT_TOL_DEG:g}; the open-loop Gen3's "
                        f"step quantization is ~0.012 deg)")
    p.add_argument("--no-hwp-preflight", action="store_true",
                   help="Skip the preflight move to the plan's first HWP angle "
                        "(the move is physical motion, and is what catches an "
                        "unhomed or stalled rotator)")
    p.add_argument("--mount", choices=("auto", "on", "off"), default="auto",
                   help="Mount: 'auto' connects and slews only when the plan "
                        "names a sky position (default), 'on' always connects "
                        "(e.g. to home or park), 'off' refuses to run a pointed "
                        "plan unless --unpointed is also given")
    p.add_argument("--unpointed", action="store_true",
                   help="With --mount off, capture a pointed plan anyway. The "
                        "frames get NO sky position -- salvage mode, use only "
                        "when the mount is unavailable")
    p.add_argument("--no-mount-home", action="store_true",
                   help="Skip homing before the first slew (only when the mount "
                        "was already homed this power cycle; homing is what "
                        "makes the pointing model valid)")
    p.add_argument("--park-on-finish", action="store_true",
                   help="Park the mount after the last frame is written")
    p.add_argument("--skip-filter-check", action="store_true",
                   help="Downgrade EFW name mismatch from abort to warning")
    p.add_argument("--skip-hwp-check", action="store_true",
                   help="Downgrade HWP connect/verify failure from abort to warning")
    p.add_argument("--skip-mount-check", action="store_true",
                   help="Downgrade mount connect/enable/home failure from abort "
                        "to warning")
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
        _describe(config, plan_path, args.setpoint, args.hwp, args.mount, args.unpointed)
        return 0

    return run(config, plan_path, args)


if __name__ == "__main__":
    raise SystemExit(main())
