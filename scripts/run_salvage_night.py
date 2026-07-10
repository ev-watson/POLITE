#!/usr/bin/env python3
"""Standalone NO-MOUNT salvage runner for a dead-DEC commissioning night.

Drives ONLY the QHY268M camera + ZWO EFW filter wheel + HWP rotator via the
fault-isolated per-component connect helpers in ``obs_utils.interactive``. It
never enables motors, homes, or slews the mount, so it is safe on a night where
engaging DEC auto-disconnects the mount. Everything downstream of pointing --
exposure, filter/HWP selection, FITS filenames, headers, and the PolConfig
sidecar -- is reused verbatim from the tested ``obs_utils.night_session`` path
(``_run_frames``), just called with ``pwi4=None``.

Salvage-plan targets are coordinate-less (``track: false``, no RA/Dec); the
runner captures at whatever fixed pointing the tube is stuck at. Any target
whose name marks a whole-polarimeter rotation (e.g. ``+45rot``) pauses for the
operator to rotate the instrument and confirm both Savart beams are still on the
detector before continuing.

    python scripts/run_salvage_night.py night_plans/20260709_salvage.yaml --dry-run
    python scripts/run_salvage_night.py night_plans/20260709_salvage.yaml --run

The dry-run touches no hardware: it prints the exact execution order (including
the manual pause points) so you can eyeball it before committing.
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from obs_utils.night_plan import NightPlanError, load_night_plan
from obs_utils.night_session import (
    _apply_session_camera,
    _flush_pol_config_sidecar,
    _run_cal_frames,
    _run_frames,
)
from obs_utils import interactive

logger = logging.getLogger("salvage")

# Physical EFW carousel as installed 2026-07-10 (0-indexed ASCOM Position):
#   pos 0 = slot 1 = Clear,  pos 1 = slot 2 = B,  pos 2 = slot 3 = V,
#   pos 3 = slot 4 = R,      pos 4 = slot 5 = Dark.
# The wheel's OWN Names (from the ASCOM Remote EFW driver) are authoritative for
# name->position resolution in obs_utils.alpaca.set_filter_position, so they MUST
# match this order or every "Photometric V" request lands on the wrong slot.
INSTALLED_EFW_NAMES = ["Clear", "Photometric B", "Photometric V", "Photometric R", "Dark"]

# Marks targets that require a manual whole-instrument rotation before capture.
_ROTATION_MARKERS = ("+45", "rot45")


def _is_rotation_target(name: str) -> bool:
    low = name.lower()
    return any(m in low for m in _ROTATION_MARKERS)


def _verify_filter_wheel(imaging, *, skip: bool) -> None:
    """Fail-closed check that name->slot resolution matches installed hardware.

    ``set_filter_position`` prefers the wheel's own ``Names`` list; only if a
    requested name is absent there does it fall back to our config. So an EFW
    driver still labelled B,V,R,Clear,Dark would silently shoot V through B.
    """
    wheel = imaging.filter_wheel
    try:
        live = [str(n) for n in list(wheel.Names)]
    except Exception:
        live = []

    logger.info("[efw] installed truth : %s", INSTALLED_EFW_NAMES)
    logger.info("[efw] driver Names     : %s", live or "(empty -> falls back to config)")
    logger.info("[efw] config filter_names: %s", list(imaging.filter_names or []))

    if not live:
        # Empty driver Names -> resolution uses our (now-corrected) config order.
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
        "Name-based filter selection would land on the WRONG slot (e.g. V->B).\n"
        "FIX: set the filter names in ASCOM Remote's ZWO EFW driver to the\n"
        "installed order, restart the ASCOM Remote server, and rerun. This also\n"
        "fixes the smoke test and every future night."
    )
    if skip:
        logger.warning("[efw] %s", msg)
        logger.warning("[efw] --skip-filter-check set: continuing DESPITE mismatch.")
        return
    raise SystemExit(msg)


def _session_paths(config) -> tuple[Path, str]:
    """Mirror run_night_session's polite-naming session dir + date string."""
    from datetime import datetime, timezone

    now = datetime.now(timezone.utc) if config.use_utc else datetime.now()
    if config.session_name and len(str(config.session_name)) == 8:
        ymd = str(config.session_name)
    else:
        ymd = now.strftime("%Y%m%d")
    base = Path(config.base_data_dir).expanduser().resolve()
    session_dir = base / ymd
    session_dir.mkdir(parents=True, exist_ok=True)
    polite_date = f"{ymd[:4]}-{ymd[4:6]}-{ymd[6:8]}"
    return session_dir, polite_date


def _describe(config) -> None:
    print("SALVAGE execution order (NO MOUNT):")
    if config.calibration_before:
        print(f"  1. calibration_before: {len(config.calibration_before)} frame-plan(s)")
    n = 2 if config.calibration_before else 1
    for target in config.targets:
        pause = "  <-- PAUSE: rotate polarimeter, confirm beams" if _is_rotation_target(target.name) else ""
        print(f"  {n}. target: {target.name} ({len(target.frames)} frame-plan(s)){pause}")
        n += 1
    if config.calibration_after:
        print(f"  {n}. calibration_after: {len(config.calibration_after)} frame-plan(s)")
    print("\n(dry-run; pass --run to execute against the camera/EFW/HWP)")


def run(plan_path: Path, *, skip_filter_check: bool, assume_yes: bool) -> int:
    config = load_night_plan(plan_path)

    # --- connect ONLY camera + wheel + rotator; never touch the mount ----------
    logger.info("[connect] camera ...")
    session = interactive.connect_camera(alpaca_config=config.startup.alpaca)
    logger.info("[connect] filter wheel ...")
    session = interactive.connect_filter_wheel(alpaca_config=config.startup.alpaca)
    logger.info("[connect] HWP rotator ...")
    session = interactive.connect_rotator(alpaca_config=config.startup.alpaca)
    imaging = session.imaging

    _verify_filter_wheel(imaging, skip=skip_filter_check)

    if config.capture_context is not None:
        _apply_session_camera(imaging, config.capture_context)

    session_dir, polite_date = _session_paths(config)
    logger.info("Session data directory: %s", session_dir)

    # Mount-free stand-in for the StartupState that _run_frames/_run_cal_frames expect.
    state = SimpleNamespace(imaging=imaging, pwi4=None, ntp_status=None)
    block_log: list = []
    session_id = config.session_name or session_dir.name

    def _flush() -> None:
        if config.capture_context is not None:
            _flush_pol_config_sidecar(
                session_dir, config.capture_context, session_id=session_id, block_log=block_log
            )

    stage = config.calibration_stage
    if config.calibration_before and stage in ("before", "both"):
        _run_cal_frames(
            state, config, config.calibration_before, session_dir, polite_date,
            None, block_log, "before",
        )
        _flush()

    for target in config.targets:
        if _is_rotation_target(target.name):
            print("\n" + "=" * 70)
            print(f"MANUAL STEP for '{target.name}':")
            print("  Rotate the WHOLE polarimeter (field rotator) +45 deg, then")
            print("  confirm BOTH Savart beams are still on the detector.")
            print("=" * 70)
            if not assume_yes:
                input("Press Enter when the rotation is done and beams confirmed ... ")
        logger.info("Target: %s", target.name)
        _run_frames(
            imaging, None, target.frames, session_dir, target, config,
            ntp_status=None, date_str=polite_date,
            block_log=block_log, block_id=target.name,
        )
        _flush()

    if config.calibration_after and stage in ("after", "both"):
        _run_cal_frames(
            state, config, config.calibration_after, session_dir, polite_date,
            None, block_log, "after",
        )
        _flush()

    if config.capture_context is not None:
        _flush_pol_config_sidecar(
            session_dir, config.capture_context, session_id=session_id,
            block_log=block_log, final=True,
        )
    logger.info("Salvage session complete. Data in %s", session_dir)
    return 0


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("plan", help="Path to the salvage night-plan YAML")
    p.add_argument("--run", action="store_true", help="Execute (default is dry-run)")
    p.add_argument("--skip-filter-check", action="store_true",
                   help="Downgrade the EFW name mismatch from abort to warning (operator confirmed by eye)")
    p.add_argument("--yes", action="store_true",
                   help="Do not pause for manual polarimeter-rotation confirmation")
    args = p.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    try:
        config = load_night_plan(Path(args.plan))
    except NightPlanError as exc:
        print(f"plan error: {exc}")
        return 2

    if not args.run:
        _describe(config)
        return 0

    return run(Path(args.plan), skip_filter_check=args.skip_filter_check, assume_yes=args.yes)


if __name__ == "__main__":
    raise SystemExit(main())
