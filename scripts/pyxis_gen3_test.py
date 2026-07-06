#!/usr/bin/env python
"""Bench verification for the native Optec Pyxis 2" Gen3 serial driver.

Drives ``obs_utils.pyxis_gen3.PyxisGen3`` over the FTDI serial link (no INDIGO /
Alpaca). Use this once the Optec cable is plugged in and the rotator has 12 V
power, to confirm the framed protocol and baud before wiring the HWP into the
night session.

Safety: the default run is **read-only** -- it pings (GETDNN) and prints status
(GETSTA) only. Movement is opt-in:

    # read-only: connect, ping, status
    python scripts/pyxis_gen3_test.py --port /dev/cu.usbserial-OP7XD6WD

    # add a homing run (rotator will physically seek the home magnet)
    python scripts/pyxis_gen3_test.py --home

    # home, then move to 90 deg and report the settled position
    python scripts/pyxis_gen3_test.py --home --move 90

Run under the project env::

    /Users/blu3/miniforge3/envs/POLITE/bin/python scripts/pyxis_gen3_test.py --port ...

Protocol reference: reference-sheets/hardware/pyxis-command-processing-rev2.md
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

logger = logging.getLogger("pyxis_gen3_test")


def _default_port() -> str:
    try:
        from obs_utils.user_config import PYXIS_CONFIG

        return PYXIS_CONFIG.port
    except Exception:  # pragma: no cover - config optional
        return "/dev/cu.usbserial-OP7XD6WD"


def _log_status(dev, label: str) -> dict:
    status = dev.get_status()
    logger.info("[status] %s:", label)
    for key, val in status.items():
        logger.info("           %-14s = %s", key, val)
    return status


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--port", default=None,
                   help="Serial device (default: PYXIS_CONFIG.port)")
    p.add_argument("--baud", type=int, default=19200,
                   help="Baud to try first (default 19200)")
    p.add_argument("--no-autodetect", action="store_true",
                   help="Do not probe other baud rates if the first fails")
    p.add_argument("--home", action="store_true",
                   help="Run the homing routine (moves the rotator)")
    p.add_argument("--move", type=float, default=None, metavar="DEG",
                   help="After connecting/homing, move to this absolute PA [deg]")
    p.add_argument("--move-timeout", type=float, default=120.0,
                   help="Timeout [s] for a move (default 120)")
    p.add_argument("--home-timeout", type=float, default=180.0,
                   help="Timeout [s] for homing (default 180)")
    p.add_argument("--poll", type=float, default=0.5,
                   help="Status poll interval [s] during move/home (default 0.5)")
    p.add_argument("--read-timeout", type=float, default=2.0,
                   help="Per-command serial read timeout [s] (default 2.0)")
    p.add_argument("--dtr", choices=["default", "low", "high"], default="default",
                   help="Force DTR/RTS level on open (default: leave pyserial default)")
    p.add_argument("-v", "--verbose", action="store_true",
                   help="Debug logging (shows raw TX/RX frames)")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )

    from obs_utils.pyxis_gen3 import (
        PyxisError,
        PyxisTimeout,
        connect_pyxis_gen3,
    )

    port = args.port or _default_port()
    toggle_dtr = {"default": None, "low": False, "high": True}[args.dtr]
    logger.info("Connecting to Pyxis Gen3 on %s (baud %d, autodetect=%s)",
                port, args.baud, not args.no_autodetect)

    try:
        dev = connect_pyxis_gen3(
            port,
            baud=args.baud,
            autodetect=not args.no_autodetect,
            read_timeout_s=args.read_timeout,
            toggle_dtr=toggle_dtr,
        )
    except (ConnectionError, OSError) as exc:
        logger.error("Connection failed: %s", exc)
        logger.error("Check: cable plugged in (`ls /dev/cu.usbserial*`), 12 V "
                     "power to the rotator, and that no other process holds the port.")
        return 1

    try:
        logger.info("[ping] nickname = %r", dev.ping())
        status = _log_status(dev, "initial")

        if args.home:
            logger.info("[home] starting homing routine ...")
            final = dev.home(poll_s=args.poll, timeout_s=args.home_timeout)
            logger.info("[home] homed; Current PA = %.3f deg", final)
            status = _log_status(dev, "after home")

        if args.move is not None:
            if not dev._flag(status, "Is Homed"):  # noqa: SLF001 - explicit safety check
                logger.warning("[move] rotator is NOT homed; the controller will "
                               "likely reject MOVEPA (error 11). Pass --home first.")
            logger.info("[move] -> %.3f deg ...", args.move)
            final = dev.move_absolute(
                args.move, poll_s=args.poll, timeout_s=args.move_timeout
            )
            logger.info("[move] settled; Current PA = %.3f deg (requested %.3f)",
                        final, args.move)
            _log_status(dev, "after move")

        logger.info("Pyxis Gen3 bench test: PASS")
        return 0
    except (PyxisError, PyxisTimeout) as exc:
        logger.error("Command failed: %s", exc)
        return 2
    finally:
        dev.close()


if __name__ == "__main__":
    raise SystemExit(main())
