#!/usr/bin/env python
"""Observatory end-to-end smoke test -- runs ON the Windows observatory PC.

Fastest possible "is the whole chain alive?" check before real observing. It:

  1. Homes the PlaneWave mount (PWI4).
  2. Slews to a random field near zenith (safe, high, minimal airmass).
  3. Rotates the HWP (Optec Pyxis, Alpaca rotator via the Optec Universal ASCOM
     driver on the ASCOM Remote Server) to an arbitrary angle.
  4. Selects an arbitrary filter (ZWO EFW, Alpaca).
  5. Takes one short light frame (QHY268M, Alpaca) and writes a FITS.

Deployment profile: **observatory / Windows (production)**. Mount + field
rotator are on PWI4 (:8220); camera + EFW + HWP-Pyxis are Alpaca devices on the
Windows ASCOM Remote Server (:11111). Because this script runs on that same PC,
both default to ``localhost``.

Priorities: minimal time, minimal checks. There is **no** plate-solve, pointing
verification, focus, cooling wait, or autoguiding -- each step just fires and we
confirm it did not raise. To go even faster on a re-run of an already-homed
mount, pass ``--skip-home``.

The HWP move uses ``obs_utils.alpaca.move_rotator_absolute`` (MoveAbsolute +
poll IsMoving). Its settle read after a driver that reports ``IsMoving`` slowly
is an OPEN site-verification item; a real Optec Alpaca driver is expected to be
fine for this smoke test.

Example (on the observatory PC)::

    python scripts/observatory_smoke_test.py --rotator-index 0
    python scripts/observatory_smoke_test.py --exposure 3 --gain 100 --out D:/tmp/smoke.fits
"""
from __future__ import annotations

import argparse
import dataclasses
import logging
import random
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from alpyca_tools.fits_writer import FitsHeaderConfig, PolarimetryCards

from obs_utils import imaging
from obs_utils.mount import connect_mount, enable_motors, home_mount, wait_for_slew
from obs_utils.pwi4_client import PWI4
from obs_utils.user_config import ALPACA_CONFIG, PWI4_CONFIG

logger = logging.getLogger("obs_smoke")

# Arbitrary HWP angles to pick from (standard half-wave-plate positions).
HWP_ANGLES = (0.0, 22.5, 45.0, 67.5, 90.0, 112.5, 135.0, 157.5)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    # Endpoints (localhost when run on the observatory PC).
    p.add_argument("--pwi4-host", default=PWI4_CONFIG.host, help="PWI4 host (default: config)")
    p.add_argument("--pwi4-port", type=int, default=PWI4_CONFIG.port, help="PWI4 port (default: 8220)")
    p.add_argument("--alpaca-host", default=ALPACA_CONFIG.host,
                   help="Alpaca host:port of the ASCOM Remote Server (default: config)")
    p.add_argument("--camera-index", type=int, default=ALPACA_CONFIG.camera_index,
                   help="Alpaca Camera # (QHY268M)")
    p.add_argument("--filterwheel-index", type=int,
                   default=ALPACA_CONFIG.filterwheel_index if ALPACA_CONFIG.filterwheel_index is not None else 0,
                   help="Alpaca FilterWheel # (ZWO EFW)")
    p.add_argument("--rotator-index", type=int,
                   default=ALPACA_CONFIG.rotator_index if ALPACA_CONFIG.rotator_index is not None else 0,
                   help="Alpaca Rotator # for the HWP Pyxis (Remote Server device #; default 0)")

    # Behaviour.
    p.add_argument("--skip-home", action="store_true", help="Skip mount homing (faster re-run)")
    p.add_argument("--exposure", type=float, default=2.0, help="Exposure time [s] (default 2)")
    p.add_argument("--gain", type=int, help="Camera gain (optional)")
    p.add_argument("--offset", type=int, help="Camera offset (optional)")
    p.add_argument("--out", default=None, help="Output FITS path (default: ./obs_smoke_<UTC>.fits)")
    p.add_argument("--seed", type=int, help="RNG seed for reproducible 'random' picks")
    return p.parse_args()


def _hms(hours: float) -> str:
    h = hours % 24.0
    hh = int(h)
    m = (h - hh) * 60.0
    mm = int(m)
    ss = (m - mm) * 60.0
    return f"{hh:02d}:{mm:02d}:{ss:05.2f}"


def _dms(deg: float) -> str:
    sign = "-" if deg < 0 else "+"
    a = abs(deg)
    dd = int(a)
    m = (a - dd) * 60.0
    mm = int(m)
    ss = (m - mm) * 60.0
    return f"{sign}{dd:02d}:{mm:02d}:{ss:04.1f}"


def _pick_near_zenith(pwi4: PWI4, rng: random.Random) -> tuple[str, float, float]:
    """Command a slew to a random field near zenith. Non-blocking (mount starts
    slewing immediately). Returns (mode, coord0, coord1) for logging.

    Uses the site's LMST + latitude to hit the true zenith apparent RA/Dec with a
    few-degree random offset. Falls back to a high Alt/Az goto if PWI4 does not
    report site coordinates.
    """
    st = pwi4.status()
    lmst = getattr(st.site, "lmst_hours", None)
    lat = getattr(st.site, "latitude_degs", None)

    if lmst is not None and lat is not None:
        ra_h = (lmst + rng.uniform(-0.20, 0.20)) % 24.0            # ~+-3 deg in RA
        dec_d = max(min(lat + rng.uniform(-4.0, 4.0), 89.0), -89.0)  # ~+-4 deg in Dec
        logger.info("[slew] near-zenith apparent RA=%s Dec=%s (LMST=%.4fh, lat=%.3f)",
                    _hms(ra_h), _dms(dec_d), lmst, lat)
        pwi4.mount_goto_ra_dec_apparent(ra_h, dec_d)
        return "radec_apparent", ra_h, dec_d

    alt = rng.uniform(85.0, 88.0)
    az = rng.uniform(0.0, 360.0)
    logger.warning("[slew] site LMST/latitude unavailable; using Alt/Az fallback "
                   "alt=%.2f az=%.2f", alt, az)
    pwi4.mount_goto_alt_az(alt, az)
    return "altaz", alt, az


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = parse_args()
    rng = random.Random(args.seed)

    out_path = Path(args.out) if args.out else Path(f"./obs_smoke_{time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())}.fits")

    t_start = time.time()
    session = None
    pwi4 = None
    try:
        # --- 1. Mount: connect + home -------------------------------------
        pwi4 = PWI4(host=args.pwi4_host, port=args.pwi4_port)
        logger.info("[mount] connecting PWI4 at %s:%d", args.pwi4_host, args.pwi4_port)
        connect_mount(pwi4)
        enable_motors(pwi4)
        if args.skip_home:
            logger.info("[mount] --skip-home: not homing")
        else:
            t0 = time.time()
            home_mount(pwi4)
            logger.info("[mount] homed (%.1fs)", time.time() - t0)

        # --- 2. Alpaca session: camera + EFW + HWP rotator ----------------
        cfg = dataclasses.replace(
            ALPACA_CONFIG,
            host=args.alpaca_host,
            camera_index=args.camera_index,
            filterwheel_index=args.filterwheel_index,
            rotator_index=args.rotator_index,
        )
        logger.info("[alpaca] opening session on %s (cam=%d, efw=%d, rot=%d)",
                    cfg.host, cfg.camera_index, cfg.filterwheel_index, cfg.rotator_index)
        session = imaging.open_session(cfg)

        # --- 3. Slew near zenith (non-blocking) + overlap HWP/filter ------
        # The mount goto returns immediately, so the HWP and filter moves run
        # concurrently with the slew -> minimal wall-clock time.
        mode, c0, c1 = _pick_near_zenith(pwi4, rng)

        hwp_angle = rng.choice(HWP_ANGLES)
        logger.info("[hwp] -> %.1f deg", hwp_angle)
        hwp_final = imaging.select_hwp_angle(session, hwp_angle)
        logger.info("[hwp] settled at %.2f deg", hwp_final)

        names = cfg.filter_names or []
        light_slots = [i for i, n in enumerate(names) if n.lower() != "dark"] or list(range(max(len(names), 1)))
        filter_slot = rng.choice(light_slots)
        filter_name = names[filter_slot] if 0 <= filter_slot < len(names) else str(filter_slot)
        logger.info("[filter] -> slot %d (%s)", filter_slot, filter_name)
        landed = imaging.select_filter(session, filter_slot)
        logger.info("[filter] landed on slot %d", landed)

        logger.info("[slew] waiting for mount to settle ...")
        t0 = time.time()
        wait_for_slew(pwi4)
        st = pwi4.status()
        logger.info("[slew] settled (%.1fs) at RA=%.4fh Dec=%.3f Alt=%.2f",
                    time.time() - t0,
                    getattr(st.mount, "ra_j2000_hours", float("nan")),
                    getattr(st.mount, "dec_j2000_degs", float("nan")),
                    getattr(st.mount, "altitude_degs", float("nan")))

        # --- 4. Capture one light frame -----------------------------------
        if mode == "radec_apparent":
            ra_str, dec_str = _hms(c0), _dms(c1)
        else:
            ra_str = dec_str = None
        header = FitsHeaderConfig(
            imagetyp="LIGHT",
            object_name="ZENITH-SMOKE",
            instrument="QHY268M",
            telescope="PlaneWave CDK20",
            observatory="POLITE observatory",
            filter_name=filter_name,
            ra=ra_str,
            dec=dec_str,
            polarimetry=PolarimetryCards(hwp_angle_deg=hwp_angle),
        )
        request = imaging.CaptureRequest(
            exposure_s=args.exposure,
            is_light=True,
            gain=args.gain,
            offset=args.offset,
        )
        logger.info("[photo] exposing %.2fs -> %s", args.exposure, out_path)
        written = imaging.capture_fits_file(
            session, request, header, out_path, timeout_s=args.exposure + 60.0
        )

        logger.info("=" * 56)
        logger.info("OBSERVATORY SMOKE TEST: PASS  (%.1fs total)", time.time() - t_start)
        logger.info("  mount   homed=%s, slewed near zenith (%s)", not args.skip_home, mode)
        logger.info("  hwp     %.1f deg", hwp_angle)
        logger.info("  filter  slot %d (%s)", filter_slot, filter_name)
        logger.info("  frame   %s", written)
        logger.info("=" * 56)
        return 0

    except Exception as exc:  # minimal checks: any failure aborts loudly
        logger.exception("OBSERVATORY SMOKE TEST: FAIL (%.1fs in) -- %s",
                         time.time() - t_start, exc)
        return 1
    finally:
        if session is not None:
            try:
                session.close()
            except Exception:
                pass


if __name__ == "__main__":
    raise SystemExit(main())
