#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from alpyca_tools.camera_device import CameraDevice
from alpyca_tools.camera_ops import ExposureSettings, capture_image


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Basic Alpaca camera diagnostics.")
    parser.add_argument("--host", default="localhost:11111", help="Alpaca host or host:port")
    parser.add_argument("--camera", type=int, default=0, help="Camera device number")
    parser.add_argument("--exposure", type=float, help="Run a short exposure (seconds)")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    camera = CameraDevice(args.host, args.camera)

    camera.Connected = True
    try:
        print(f"Name: {camera.Name}")
        print(f"Driver: {camera.DriverInfo} (v{camera.DriverVersion})")
        print(f"Sensor: {camera.SensorName}")
        print(f"Size: {camera.CameraXSize} x {camera.CameraYSize}")
        print(f"CanSetTemp: {camera.CanSetCCDTemperature}")
        print(f"CCD Temp: {camera.CCDTemperature:.2f} C")
        print(f"CoolerOn: {camera.CoolerOn}")

        if args.exposure is not None:
            settings = ExposureSettings(exposure_s=float(args.exposure))
            data, info, dtype = capture_image(camera, settings)
            print(f"Exposure OK: dtype={dtype} rank={info.rank} shape={data.shape}")
    finally:
        try:
            camera.Connected = False
        except Exception:
            pass


if __name__ == "__main__":
    main()
