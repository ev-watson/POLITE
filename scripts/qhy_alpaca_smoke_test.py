#!/usr/bin/env python
"""Minimal POLITE camera smoke test against the SDK-direct Alpaca server.

Uses localhost:11112 / camera 0 by default (config.windows.yaml). EFW and
rotator are not exercised here — only the QHY268M capture path.

    python scripts/qhy_alpaca_smoke_test.py --exposure 2 --out D:/tmp/qhy_smoke.fits
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from alpyca_tools.fits_writer import FitsHeaderConfig, capture_fits
from obs_utils.alpaca import connect_camera


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--host", default="localhost:11112", help="QHY Alpaca server (default SDK-direct)")
    p.add_argument("--camera-index", type=int, default=0)
    p.add_argument("--exposure", type=float, default=2.0)
    p.add_argument("--gain", type=int, default=0)
    p.add_argument("--offset", type=int, default=30)
    p.add_argument("--readout-mode", type=int, default=0)
    p.add_argument("--out", type=Path, default=Path("./qhy_alpaca_smoke.fits"))
    return p.parse_args()


def main() -> int:
    args = parse_args()
    print(f"Connecting camera at {args.host} device {args.camera_index} ...")
    cam = connect_camera(args.host, args.camera_index)

    t0 = time.time()
    while not cam.Connected:
        if time.time() - t0 > 60:
            print("Timed out waiting for Connected=True")
            return 1
        time.sleep(0.25)

    print(f"Connected. Sensor {cam.CameraXSize}x{cam.CameraYSize}, gain range {cam.GainMin}-{cam.GainMax}")
    cam.Gain = args.gain
    cam.Offset = args.offset
    cam.ReadoutMode = args.readout_mode
    if getattr(cam, "CanSetCCDTemperature", False):
        cam.SetCCDTemperature = -20.0

    header = FitsHeaderConfig(
        object_name="QHY_SMOKE",
        frame_type="Light",
        filter_name="Clear",
        instrument="QHY268M",
        gain_setting=args.gain,
        readout_mode=args.readout_mode,
        offset_setting=args.offset,
        cooler_setpoint_c=-20.0,
        pixel_size_um=3.76,
    )

    from alpyca_tools.camera_ops import ExposureSettings

    out = capture_fits(
        cam,
        ExposureSettings(
            exposure_s=args.exposure,
            is_light=True,
            gain=args.gain,
            offset=args.offset,
            readout_mode=args.readout_mode,
        ),
        header,
        args.out,
    )
    print(f"Wrote {out}")
    cam.Connected = False
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
