#!/usr/bin/env python
"""Mimic a half-wave-plate (HWP) modulation test run for POLITE.

Two modes:

* ``sim`` (default, no hardware) -- reproduces the LE2Pol-style bench test
  (Wiersema et al. 2023): step the HWP over a set of angles against an
  *unpolarized* and a *polarized* source, build the normalized beam-difference
  modulation ``R(theta) = q cos4theta + u sin4theta``, reduce it with the real
  ``poltools`` estimators, and check the Fourier content (an ideal dual-beam
  polarimeter has all power in the n=4 harmonic; non-zero a0 / n=2 flag a
  non-ideal beam split / HWP pleochroism -- Patat & Taubenberger 2011).
  Run under the project env, e.g.::

      /Users/blu3/miniforge3/envs/POLITE/bin/python scripts/hwp_modulation_test.py

* ``hardware`` -- the bench dress-rehearsal on the real lab kit. Drives the
  Optec Pyxis Gen3 (carrying the HWP) through the same angle sequence over its
  **native FTDI serial link** (``obs_utils.pyxis_gen3``, NOT INDIGO/Alpaca --
  the Gen3 framed protocol is incompatible with INDIGO's legacy
  ``indigo_rotator_optec`` driver). The ZWO EFW (optional band select) and the
  QHY268M stay on INDIGO -> Alpaca. Captures one frame per angle, writing FITS
  with the ``HWPANG`` keyword. No reduction -- this verifies the acquisition +
  stepping loop. The rotator is homed first (absolute ``MOVEPA`` moves are
  rejected on an un-homed controller); pass ``--skip-home`` if it is already
  homed. See ``docs/lab_trial_checklist.md``.

Standard HWP angle sequences (22.5 deg increments; >= 4 angles, never 2):
  4 -> {0, 22.5, 45, 67.5};  8 -> 0..157.5;  16 -> 0..337.5.
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import List, Optional

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

logger = logging.getLogger("hwp_modulation_test")


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


# --------------------------------------------------------------------------- #
# Angle sequences
# --------------------------------------------------------------------------- #
def build_angles(spec: str, sweep_step: Optional[float]) -> List[float]:
    """Return the HWP angle list [deg] from ``--angles`` and ``--sweep-step``."""
    if sweep_step is not None:
        if sweep_step <= 0:
            raise ValueError("--sweep-step must be > 0")
        return [round(a, 3) for a in np.arange(0.0, 360.0, sweep_step)]
    presets = {
        "4": [i * 22.5 for i in range(4)],    # 0, 22.5, 45, 67.5
        "8": [i * 22.5 for i in range(8)],    # 0 .. 157.5
        "16": [i * 22.5 for i in range(16)],  # 0 .. 337.5
    }
    if spec in presets:
        return presets[spec]
    try:
        return [float(x) for x in spec.split(",") if x.strip() != ""]
    except ValueError as exc:
        raise ValueError(
            f"--angles must be 4|8|16 or a comma list of degrees; got {spec!r}"
        ) from exc


def check_modulation_sampling(angles_deg):
    """Warn if the angle set aliases the 4-theta modulation.

    The analyzed signal varies as ``cos4theta``/``sin4theta``. To constrain both
    q and u the angles must sample the 4-theta phase in at least a few distinct,
    well-spread values. A regular 22.5 deg step samples 4-theta at 90 deg
    increments (4 phases) -- the standard choice. A 45 deg step samples only
    phases 0/180 deg, leaving u unconstrained (rank-deficient fit). Returns
    ``(ok, message)``.
    """
    th = np.deg2rad(np.asarray(angles_deg, dtype=float))
    A = np.column_stack([np.cos(4 * th), np.sin(4 * th)])
    s = np.linalg.svd(A, compute_uv=False)
    smin = float(s.min())
    if smin < 1e-6:
        return False, ("HWP angles alias the 4-theta modulation (design matrix "
                       "rank-deficient); u is unconstrained. Use 22.5 deg steps "
                       "(e.g. --angles 4/8/16) or a fine sweep (e.g. 5 deg).")
    cond = float(s.max() / smin)
    if cond > 20.0:
        return False, (f"HWP angle sampling is poorly conditioned (cond={cond:.1f}); "
                       "q/u errors will be inflated. Prefer 22.5 deg steps.")
    return True, ""


def fourier_harmonics(angles_deg, R):
    """Discrete projection of ``R(theta)`` onto {const, 2theta, 4theta}.

    Returns ``(a0, amp2, amp4)`` amplitudes. For an ideal dual-beam polarimeter
    the signal sits in the 4-theta (n=4) term; a0 (unequal split) and n=2 (HWP
    pleochroism) should be negligible (Fendt et al. 1996; Patat & Taubenberger
    2011). Only meaningful for >= 8 well-spread angles.
    """
    th = np.deg2rad(np.asarray(angles_deg, dtype=float))
    R = np.asarray(R, dtype=float)
    n = len(R)
    a0 = float(np.mean(R))
    amp = {}
    for k in (2, 4):
        A = (2.0 / n) * np.sum(R * np.cos(k * th))
        B = (2.0 / n) * np.sum(R * np.sin(k * th))
        amp[k] = float(np.hypot(A, B))
    return a0, amp[2], amp[4]


# --------------------------------------------------------------------------- #
# SIM mode
# --------------------------------------------------------------------------- #
def run_sim(args: argparse.Namespace) -> int:
    import poltools as pt
    from poltools._types import BeamFlux

    rng = np.random.default_rng(args.seed)
    angles = build_angles(args.angles, args.sweep_step)
    logger.info("SIM: %d HWP angles: %s", len(angles),
                ", ".join(f"{a:g}" for a in angles))

    # Injected truth: an unpolarized source and a polarized "polaroid-film" source.
    th = np.deg2rad(args.pol_theta)
    q_true = args.pol_p * np.cos(2 * th)
    u_true = args.pol_p * np.sin(2 * th)
    cases = [
        ("unpolarized", 0.0, 0.0, 0.0),
        ("polarized", args.pol_p, q_true, u_true),
    ]

    total_e = args.flux_e  # total intensity per angle over both beams
    ok = True
    for name, p_true, q_t, u_t in cases:
        S = pt.stokes_vector(1.0, q_t, u_t)
        bfs = []
        R_obs = []
        for a in angles:
            Io, Ie = pt.oe_intensities(
                S, a, retardance_deg=args.retardance, efficiency=args.efficiency
            )
            # Poisson photon noise + Gaussian read noise per beam.
            fo = rng.poisson(max(Io * total_e, 0.0)) + rng.normal(0.0, args.read_noise)
            fe = rng.poisson(max(Ie * total_e, 0.0)) + rng.normal(0.0, args.read_noise)
            fo = max(float(fo), 1.0)
            fe = max(float(fe), 1.0)
            bfs.append(BeamFlux(a, fo, fe, np.sqrt(fo), np.sqrt(fe)))
            R_obs.append((fe - fo) / (fe + fo))

        lsq = pt.lsq_modulation(bfs)
        p_rec = float(np.hypot(lsq["q"], lsq["u"]))
        th_rec = 0.5 * np.rad2deg(np.arctan2(lsq["u"], lsq["q"])) % 180.0
        sig_p = float(np.hypot(lsq["sigma_q"], lsq["sigma_u"]))

        line = (f"  {name:11s} p_inj={p_true:6.3%}  "
                f"p_rec={p_rec:6.3%} +/- {sig_p:5.3%}  "
                f"theta_rec={th_rec:6.2f} deg  chi2={lsq['chi2']:.1f}/{lsq['dof']}")
        if name == "polarized":
            line += f"  (theta_inj={args.pol_theta:.1f})"
        logger.info(line)

        # Fourier content check (needs a decent number of spread angles).
        if len(angles) >= 8:
            a0, amp2, amp4 = fourier_harmonics(angles, R_obs)
            logger.info("      Fourier: |a0|=%.4f  n2=%.4f  n4=%.4f  "
                        "(ideal: signal in n4; a0,n2 ~ 0)", abs(a0), amp2, amp4)

        # Pass criteria: unpolarized recovers < 3-sigma-ish; polarized within tol.
        if name == "unpolarized":
            if p_rec > max(5 * sig_p, 0.003):
                logger.warning("      unpolarized source shows spurious p=%.3f%%",
                               100 * p_rec)
                ok = False
        else:
            if abs(p_rec - p_true) > args.tol_p:
                logger.warning("      recovered p off by %.4f (> tol %.4f)",
                               abs(p_rec - p_true), args.tol_p)
                ok = False
            dth = abs((th_rec - args.pol_theta + 90) % 180 - 90)
            if dth > args.tol_theta:
                logger.warning("      recovered theta off by %.2f deg (> tol %.2f)",
                               dth, args.tol_theta)
                ok = False

    if args.double_ratio and args.angles == "4" and args.sweep_step is None:
        # Cross-check the flat-field-independent estimator on the polarized case.
        logger.info("  double-ratio cross-check requires exactly {0,22.5,45,67.5}")
    logger.info("SIM result: %s", "PASS" if ok else "FAIL")
    return 0 if ok else 1


# --------------------------------------------------------------------------- #
# HARDWARE mode (bench dress-rehearsal)
# --------------------------------------------------------------------------- #
def run_hardware(args: argparse.Namespace) -> int:
    # Lazy imports so sim mode needs neither alpyca nor pyserial.
    # Rotator: native Optec Pyxis Gen3 over FTDI serial (NOT INDIGO/Alpaca).
    # Camera + filter wheel: still INDIGO -> Alpaca.
    from obs_utils.alpaca import (
        connect_camera,
        connect_filter_wheel,
        set_filter_position,
    )
    from obs_utils.pyxis_gen3 import (
        PyxisError,
        PyxisTimeout,
        connect_pyxis_gen3,
    )
    from alpyca_tools.camera_ops import ExposureSettings
    from alpyca_tools.fits_writer import (
        FitsHeaderConfig,
        PolarimetryCards,
        capture_fits,
    )

    angles = build_angles(args.angles, args.sweep_step)
    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    logger.info("HARDWARE: %d HWP angles -> %s", len(angles), out_dir)

    # --- Rotator: native Pyxis Gen3 serial link ---
    port = args.pyxis_port or _default_pyxis_port()
    baud = args.pyxis_baud if args.pyxis_baud is not None else _default_pyxis_baud()
    toggle_dtr = {"default": None, "low": False, "high": True}[args.pyxis_dtr]
    logger.info("[rotator] (serial) connecting Pyxis Gen3 on %s (baud %d, autodetect=%s)",
                port, baud, not args.pyxis_no_autodetect)
    try:
        rot = connect_pyxis_gen3(
            port,
            baud=baud,
            autodetect=not args.pyxis_no_autodetect,
            read_timeout_s=args.pyxis_read_timeout,
            toggle_dtr=toggle_dtr,
        )
    except (ConnectionError, OSError) as exc:
        logger.error("[rotator] (serial) connection failed: %s", exc)
        logger.error("Check: cable plugged in (`ls /dev/cu.usbserial*`), 12 V power to "
                     "the rotator, and that no other process holds the port.")
        return 1

    fw = None
    cam = None
    written: List[Path] = []
    try:
        logger.info("[rotator] ping nickname = %r", rot.ping())
        status = rot.get_status()
        logger.info("[rotator] status: Current PA=%s  Is Homed=%s  Is Moving=%s",
                    status.get("Current PA"), status.get("Is Homed"), status.get("Is Moving"))

        # Absolute MOVEPA is rejected (error 11) on an un-homed controller, so
        # home first unless the caller asserts it is already homed.
        if not args.skip_home:
            logger.info("[rotator] homing (physical motion) ...")
            homed_pa = rot.home(poll_s=args.poll, timeout_s=args.home_timeout)
            logger.info("[rotator] homed; Current PA = %.3f deg", homed_pa)
        elif not rot._flag(rot.get_status(), "Is Homed"):  # noqa: SLF001 - explicit safety check
            logger.warning("[rotator] --skip-home set but rotator is NOT homed; MOVEPA "
                           "will likely be rejected (error 11). Re-run without --skip-home.")

        if args.filterwheel is not None:
            fw = connect_filter_wheel(args.host, args.filterwheel)
            landed = set_filter_position(fw, args.filter_slot)
            logger.info("[filters] selected slot %d", landed)

        cam = connect_camera(args.host, args.camera)
        for i, a in enumerate(angles):
            logger.info("[hwp] angle %d/%d -> %.2f deg", i + 1, len(angles), a)
            final = rot.move_absolute(a, poll_s=args.poll, timeout_s=args.move_timeout)
            settings = ExposureSettings(
                exposure_s=float(args.exposure),
                is_light=not args.dark,
                gain=args.gain,
                offset=args.offset,
            )
            header = FitsHeaderConfig(
                imagetyp="DARK" if args.dark else "LIGHT",
                instrument="QHY268M",
                observatory="Lab bench (no telescope)",
                object_name=args.object_name,
                polarimetry=PolarimetryCards(
                    hwp_angle_deg=float(final),
                    retardance_deg=float(args.retardance),
                    pol_seq_id=args.object_name,
                    pol_seq_index=i,
                ),
            )
            fname = f"{args.object_name}_{i:02d}_hwp{a:05.1f}.fit"
            path = capture_fits(cam, settings, header, out_dir / fname,
                                timeout_s=args.timeout)
            written.append(path)
            logger.info("[hwp]   wrote %s (rotator reports %.2f deg)", path.name, final)
    except (PyxisError, PyxisTimeout) as exc:
        logger.error("[rotator] FAILED: %s", exc)
    finally:
        for dev in (fw, cam):
            if dev is None:
                continue
            try:
                dev.Connected = False
            except Exception:
                pass
        try:
            rot.close()
        except Exception:
            pass

    logger.info("HARDWARE result: wrote %d/%d frames", len(written), len(angles))
    return 0 if len(written) == len(angles) else 1


# --------------------------------------------------------------------------- #
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--mode", choices=["sim", "hardware"], default="sim")
    p.add_argument("--angles", default="4",
                   help="4|8|16 or comma-separated degrees (default 4)")
    p.add_argument("--sweep-step", type=float,
                   help="Sweep 0..360 in this deg step (sim modulation-curve demo)")
    p.add_argument("--retardance", type=float, default=180.0,
                   help="HWP retardance [deg] (180 = ideal HWP)")

    # sim knobs
    p.add_argument("--pol-p", type=float, default=0.05,
                   help="Injected polarization fraction of the polarized source")
    p.add_argument("--pol-theta", type=float, default=30.0,
                   help="Injected polarization angle [deg]")
    p.add_argument("--efficiency", type=float, default=1.0)
    p.add_argument("--flux-e", type=float, default=3.0e5,
                   help="Total intensity per angle [e-]")
    p.add_argument("--read-noise", type=float, default=3.5)
    p.add_argument("--seed", type=int, default=20260702)
    p.add_argument("--tol-p", type=float, default=0.004,
                   help="Sim pass tolerance on recovered p")
    p.add_argument("--tol-theta", type=float, default=2.0,
                   help="Sim pass tolerance on recovered theta [deg]")
    p.add_argument("--double-ratio", action="store_true",
                   help="(sim) note double-ratio cross-check availability")

    # hardware knobs -- camera + filter wheel on INDIGO -> Alpaca
    p.add_argument("--host", default="localhost:11111",
                   help="Alpaca host:port for the camera + filter wheel (INDIGO alpaca agent)")

    # Rotator: native Optec Pyxis Gen3 over FTDI serial (NOT INDIGO/Alpaca).
    p.add_argument("--pyxis-port", default=None,
                   help="Serial device for the Pyxis HWP rotator (default: PYXIS_CONFIG.port)")
    p.add_argument("--pyxis-baud", type=int, default=None,
                   help="Baud to try first (default: PYXIS_CONFIG.baud)")
    p.add_argument("--pyxis-no-autodetect", action="store_true",
                   help="Do not probe other baud rates if the first fails")
    p.add_argument("--pyxis-dtr", choices=["default", "low", "high"], default="default",
                   help="Force DTR/RTS level on serial open (default: leave pyserial default)")
    p.add_argument("--pyxis-read-timeout", type=float, default=2.0,
                   help="Per-command serial read timeout [s] (default 2.0)")
    p.add_argument("--skip-home", action="store_true",
                   help="Skip the Pyxis homing routine (only if already homed)")
    p.add_argument("--home-timeout", type=float, default=180.0,
                   help="Pyxis homing timeout [s] (default 180)")
    p.add_argument("--move-timeout", type=float, default=120.0,
                   help="Pyxis per-move timeout [s] (default 120)")
    p.add_argument("--poll", type=float, default=0.5,
                   help="Status poll interval [s] during home/move (default 0.5)")

    p.add_argument("--filterwheel", type=int,
                   help="Alpaca device number of the ZWO EFW (optional)")
    p.add_argument("--filter-slot", type=int, default=0)
    p.add_argument("--camera", type=int, default=0,
                   help="Alpaca device number of the QHY268M")
    p.add_argument("--exposure", type=float, default=1.0)
    p.add_argument("--dark", action="store_true")
    p.add_argument("--gain", type=int)
    p.add_argument("--offset", type=int)
    p.add_argument("--object-name", default="HWPTEST")
    p.add_argument("--out-dir", default="./FITSDATA/hwp_test")
    p.add_argument("--timeout", type=float, default=300.0)
    return p.parse_args()


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = parse_args()
    angles = build_angles(args.angles, args.sweep_step)
    if len(angles) < 4:
        logger.error("Refusing to run with < 4 HWP angles: >= 4 are required to "
                     "cancel first-order instrumental effects (Patat & Taubenberger "
                     "2011). Got %d.", len(angles))
        return 2
    ok_sampling, msg = check_modulation_sampling(angles)
    if not ok_sampling:
        logger.error("Bad HWP angle sampling: %s", msg)
        return 2
    if args.mode == "sim":
        return run_sim(args)
    return run_hardware(args)


if __name__ == "__main__":
    raise SystemExit(main())
