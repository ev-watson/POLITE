#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from alpyca_tools.camera_device import CameraDevice
from alpyca_tools.camera_ops import ExposureSettings
from alpyca_tools.fits_writer import FitsHeaderConfig, capture_fits


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Take an Alpaca exposure and write FITS.")
    parser.add_argument("--host", default="localhost:11111", help="Alpaca host or host:port")
    parser.add_argument("--camera", type=int, default=0, help="Camera device number")
    parser.add_argument("--exposure", type=float, default=1.0, help="Exposure time in seconds")
    parser.add_argument("--light", action="store_true", help="Light frame (default)")
    parser.add_argument("--dark", action="store_true", help="Dark frame")
    parser.add_argument("--binx", type=int, default=1)
    parser.add_argument("--biny", type=int, default=1)
    parser.add_argument("--startx", type=int, default=0)
    parser.add_argument("--starty", type=int, default=0)
    parser.add_argument("--numx", type=int)
    parser.add_argument("--numy", type=int)
    parser.add_argument("--gain", type=int)
    parser.add_argument("--offset", type=int)
    parser.add_argument("--readout-mode", type=int)
    parser.add_argument("--out", required=True, help="Output FITS path")
    parser.add_argument("--object", dest="object_name")
    parser.add_argument("--observer")
    parser.add_argument("--telescope")
    parser.add_argument("--observatory")
    parser.add_argument("--instrument")
    parser.add_argument("--filter", dest="filter_name")
    parser.add_argument("--airmass", type=float)
    parser.add_argument("--ra")
    parser.add_argument("--dec")
    parser.add_argument("--ha")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    camera = CameraDevice(args.host, args.camera)
    camera.Connected = True
    try:
        exposure = ExposureSettings(
            exposure_s=float(args.exposure),
            is_light=not args.dark,
            binx=args.binx,
            biny=args.biny,
            startx=args.startx,
            starty=args.starty,
            numx=args.numx,
            numy=args.numy,
            gain=args.gain,
            offset=args.offset,
            readout_mode=args.readout_mode,
        )

        header_cfg = FitsHeaderConfig(
            imagetyp="LIGHT" if not args.dark else "DARK",
            object_name=args.object_name,
            observer=args.observer,
            telescope=args.telescope,
            observatory=args.observatory,
            instrument=args.instrument,
            filter_name=args.filter_name,
            airmass=args.airmass,
            ra=args.ra,
            dec=args.dec,
            ha=args.ha,
        )

        out_path = capture_fits(camera, exposure, header_cfg, Path(args.out))
        print(str(out_path))
    finally:
        try:
            camera.Connected = False
        except Exception:
            pass


if __name__ == "__main__":
    main()
