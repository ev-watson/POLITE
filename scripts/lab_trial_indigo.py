#!/usr/bin/env python3
"""Full lab bench trial (no telescope): all equipment in one run.

Exercises the three lab devices together, each on the backend that actually
works for it on the lab Mac:

  1. Rotator -- Optec Pyxis 2" **Gen3** HWP stage, over its native FTDI serial
     link (``obs_utils.pyxis_gen3``). NOT INDIGO: the Gen3 framed protocol is
     incompatible with INDIGO's legacy ``indigo_rotator_optec`` handshake, so it
     is driven directly over serial. (On the observatory Windows stack the Pyxis
     is instead an Alpaca rotator behind Optec's Universal ASCOM driver -- use
     ``--rotator N`` for that path.)
  2. Filters -- ZWO EFW    (INDIGO ``indigo_wheel_asi`` -> Alpaca FilterWheel)
  3. Photo   -- QHY268M    (INDIGO ``indigo_ccd_qhy2``  -> Alpaca Camera)

Each check is optional and isolated: a failure (or an absent device) in one does
not abort the others, so a partial lab setup can still be exercised. See
``docs/lab_trial_checklist.md`` for the full setup procedure.

Safety: the rotator check is **read-only by default** (connect + ping + status).
Homing and moving cause physical motion and are opt-in (``--pyxis-home`` /
``--pyxis-move``), mirroring ``scripts/pyxis_gen3_test.py``. The camera check
defaults to a light frame; pass ``--dark`` for a shutter-closed dark.

Examples:
    # Lab Mac -- everything at once (rotator read-only, camera dark):
    python scripts/lab_trial_indigo.py --host localhost:11111 \
        --pyxis-serial --filterwheel 0 --camera 0 \
        --filter-slot 1 --exposure 1.0 --dark --out ./lab_trial.fits

    # ...and actually move the HWP (homes first, then goes to 90 deg):
    python scripts/lab_trial_indigo.py --host localhost:11111 \
        --pyxis-serial --pyxis-home --pyxis-move --rotate-to 90 \
        --filterwheel 0 --camera 0 --filter-slot 1 --exposure 1.0 --dark

    # Observatory -- Pyxis as an Alpaca rotator via the Windows Remote Server:
    python scripts/lab_trial_indigo.py --host <windows-ip>:11111 \
        --rotator 2 --filterwheel 0 --camera 0 --rotate-to 90
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from obs_utils.alpaca import (
    connect_camera,
    connect_filter_wheel,
    connect_rotator,
    move_rotator_absolute,
    set_filter_position,
)
from alpyca_tools.camera_ops import ExposureSettings
from alpyca_tools.fits_writer import FitsHeaderConfig, capture_fits


logger = logging.getLogger("lab_trial")


def _default_pyxis_port() -> str:
    try:
        from obs_utils.user_config import PYXIS_CONFIG

        return PYXIS_CONFIG.port
    except Exception:  # pragma: no cover - config optional
        return "/dev/cu.usbserial-OP7XD6WD"


def _default_pyxis_baud() -> int:
    try:
        from obs_utils.user_config import PYXIS_CONFIG

        return PYXIS_CONFIG.baud
    except Exception:  # pragma: no cover - config optional
        return 19200


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--host", default="localhost:11111", help="Alpaca host:port (INDIGO alpaca agent)")

    # Rotator backend (choose one). Lab Mac = --pyxis-serial; observatory = --rotator N.
    p.add_argument("--pyxis-serial", action="store_true",
                   help="Drive the Pyxis Gen3 HWP over native serial (lab Mac; obs_utils.pyxis_gen3)")
    p.add_argument("--rotator", type=int,
                   help="Alpaca device number for the rotator (observatory Alpaca path, NOT lab Gen3)")
    p.add_argument("--filterwheel", type=int, help="Alpaca device number for the ZWO EFW")
    p.add_argument("--camera", type=int, help="Alpaca device number for the QHY268M")

    # Rotator test (shared).
    p.add_argument("--rotate-to", type=float, default=90.0, help="Target position angle [deg]")

    # Native serial (Pyxis Gen3) options.
    p.add_argument("--pyxis-port", default=None,
                   help="Serial device for the Pyxis (default: PYXIS_CONFIG.port)")
    p.add_argument("--pyxis-baud", type=int, default=None,
                   help="Baud to try first (default: PYXIS_CONFIG.baud)")
    p.add_argument("--pyxis-no-autodetect", action="store_true",
                   help="Do not probe other baud rates if the first fails")
    p.add_argument("--pyxis-home", action="store_true",
                   help="Run the Pyxis homing routine (physical motion)")
    p.add_argument("--pyxis-move", action="store_true",
                   help="Move the Pyxis to --rotate-to (physical motion; needs a homed rotator)")
    p.add_argument("--home-timeout", type=float, default=180.0, help="Pyxis homing timeout [s]")
    p.add_argument("--move-timeout", type=float, default=120.0, help="Rotator move timeout [s]")
    p.add_argument("--poll", type=float, default=0.5, help="Status poll interval during move/home [s]")

    # Filter test.
    p.add_argument("--filter-slot", type=int, default=1, help="Target filter slot index (0-based)")

    # Camera test.
    p.add_argument("--exposure", type=float, default=1.0, help="Exposure time [s]")
    p.add_argument("--dark", action="store_true", help="Take a dark frame (shutter closed)")
    p.add_argument("--gain", type=int, help="Camera gain")
    p.add_argument("--offset", type=int, help="Camera offset")
    p.add_argument("--out", default="./lab_trial.fits", help="Output FITS path")

    p.add_argument("--timeout", type=float, default=300.0, help="Exposure timeout [s]")
    return p.parse_args()


def test_rotator_serial(
    port: str,
    baud: int,
    autodetect: bool,
    target_deg: float,
    do_home: bool,
    do_move: bool,
    poll_s: float,
    home_timeout_s: float,
    move_timeout_s: float,
) -> bool:
    """Exercise the Optec Pyxis 2" Gen3 HWP rotator over its native FTDI serial
    link (``obs_utils.pyxis_gen3``) -- NOT INDIGO/Alpaca.

    Read-only by default (ping + status). Homing/moving are opt-in and physical.
    """
    from obs_utils.pyxis_gen3 import connect_pyxis_gen3  # lazy: keeps pyserial optional

    logger.info("[rotator] (serial) connecting Pyxis Gen3 on %s (baud %d, autodetect=%s)",
                port, baud, autodetect)
    dev = connect_pyxis_gen3(port, baud=baud, autodetect=autodetect)
    try:
        logger.info("[rotator] ping nickname = %r", dev.ping())
        status = dev.get_status()
        logger.info("[rotator] status: Current PA=%s  Is Homed=%s  Is Moving=%s",
                    status.get("Current PA"), status.get("Is Homed"), status.get("Is Moving"))

        if do_home:
            logger.info("[rotator] homing (physical motion) ...")
            final = dev.home(poll_s=poll_s, timeout_s=home_timeout_s)
            logger.info("[rotator] homed; Current PA = %.3f deg", final)
            status = dev.get_status()

        if do_move:
            if not dev._flag(status, "Is Homed"):  # noqa: SLF001 - explicit safety check
                logger.warning("[rotator] NOT homed; MOVEPA will likely be rejected "
                               "(error 11). Re-run with --pyxis-home.")
            logger.info("[rotator] MOVEPA -> %.3f deg ...", target_deg)
            final = dev.move_absolute(target_deg, poll_s=poll_s, timeout_s=move_timeout_s)
            logger.info("[rotator] settled at %.3f deg (target %.3f)", final, target_deg)
        else:
            logger.info("[rotator] read-only (no motion); pass --pyxis-home / --pyxis-move to exercise")
        return True
    finally:
        try:
            dev.close()
        except Exception:
            pass


def test_rotator_alpaca(host: str, device_number: int, target_deg: float) -> bool:
    """Exercise the rotator as an ASCOM Alpaca device (observatory Windows path,
    or any remote Alpaca rotator). For the lab Gen3 use ``test_rotator_serial``.
    """
    logger.info("[rotator] (alpaca) connecting device %d on %s", device_number, host)
    rot = connect_rotator(host, device_number)
    try:
        start = float(rot.Position)
        logger.info("[rotator] connected; current position = %.2f deg", start)
        logger.info("[rotator] MoveAbsolute -> %.2f deg", target_deg)
        final = move_rotator_absolute(rot, target_deg)
        logger.info("[rotator] settled at %.2f deg (target %.2f)", final, target_deg)
        return True
    finally:
        try:
            rot.Connected = False
        except Exception:
            pass


def test_filter_wheel(host: str, device_number: int, slot: int) -> bool:
    logger.info("[filters] connecting device %d on %s", device_number, host)
    fw = connect_filter_wheel(host, device_number)
    try:
        try:
            names = list(fw.Names)
        except Exception:
            names = []
        logger.info("[filters] connected; current slot = %d, names = %s", int(fw.Position), names or "(unavailable)")
        logger.info("[filters] moving to slot %d", slot)
        landed = set_filter_position(fw, slot)
        label = names[landed] if 0 <= landed < len(names) else str(landed)
        logger.info("[filters] landed on slot %d (%s)", landed, label)
        return True
    finally:
        try:
            fw.Connected = False
        except Exception:
            pass


def test_camera(host: str, device_number: int, args: argparse.Namespace) -> bool:
    logger.info("[photo] connecting device %d on %s", device_number, host)
    cam = connect_camera(host, device_number)
    try:
        logger.info(
            "[photo] connected; sensor %sx%s, taking %s of %.3fs",
            getattr(cam, "CameraXSize", "?"),
            getattr(cam, "CameraYSize", "?"),
            "DARK" if args.dark else "LIGHT",
            args.exposure,
        )
        exposure = ExposureSettings(
            exposure_s=float(args.exposure),
            is_light=not args.dark,
            gain=args.gain,
            offset=args.offset,
        )
        header_cfg = FitsHeaderConfig(
            imagetyp="DARK" if args.dark else "LIGHT",
            instrument="QHY268M",
            observatory="Lab bench (no telescope)",
        )
        out_path = capture_fits(
            cam,
            exposure,
            header_cfg,
            Path(args.out),
            timeout_s=args.timeout,
        )
        logger.info("[photo] wrote %s", out_path)
        return True
    finally:
        try:
            cam.Connected = False
        except Exception:
            pass


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = parse_args()

    # Rotator backend is exclusive: native serial (lab Gen3) vs Alpaca (observatory).
    if args.pyxis_serial and args.rotator is not None:
        logger.error(
            "Choose one rotator backend: --pyxis-serial (lab Mac, native Gen3 serial) "
            "OR --rotator N (Alpaca / observatory). Not both."
        )
        return 2

    want_rotator = args.pyxis_serial or (args.rotator is not None)
    if not want_rotator and args.filterwheel is None and args.camera is None:
        logger.error(
            "No devices selected. Pass at least one of --pyxis-serial (or --rotator N), "
            "--filterwheel, --camera with its Alpaca device number "
            "(see docs/lab_trial_checklist.md)."
        )
        return 2

    results: dict[str, bool] = {}

    if args.pyxis_serial:
        port = args.pyxis_port or _default_pyxis_port()
        baud = args.pyxis_baud if args.pyxis_baud is not None else _default_pyxis_baud()
        try:
            results["rotator(serial)"] = test_rotator_serial(
                port,
                baud,
                autodetect=not args.pyxis_no_autodetect,
                target_deg=args.rotate_to,
                do_home=args.pyxis_home,
                do_move=args.pyxis_move,
                poll_s=args.poll,
                home_timeout_s=args.home_timeout,
                move_timeout_s=args.move_timeout,
            )
        except Exception as exc:
            logger.exception("[rotator] (serial) FAILED: %s", exc)
            logger.error("Check: cable plugged in (`ls /dev/cu.usbserial*`), 12 V power to "
                         "the rotator, and that no other process holds the port.")
            results["rotator(serial)"] = False
    elif args.rotator is not None:
        try:
            results["rotator(alpaca)"] = test_rotator_alpaca(args.host, args.rotator, args.rotate_to)
        except Exception as exc:
            logger.exception("[rotator] (alpaca) FAILED: %s", exc)
            results["rotator(alpaca)"] = False

    if args.filterwheel is not None:
        try:
            results["filters"] = test_filter_wheel(args.host, args.filterwheel, args.filter_slot)
        except Exception as exc:
            logger.exception("[filters] FAILED: %s", exc)
            results["filters"] = False

    if args.camera is not None:
        try:
            results["photo"] = test_camera(args.host, args.camera, args)
        except Exception as exc:
            logger.exception("[photo] FAILED: %s", exc)
            results["photo"] = False

    logger.info("=" * 48)
    for name, ok in results.items():
        logger.info("  %-16s %s", name, "PASS" if ok else "FAIL")
    logger.info("=" * 48)

    return 0 if all(results.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
