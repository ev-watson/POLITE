#!/usr/bin/env python
"""List QHY cameras visible to qhyccd.dll (no Alpaca server required).

Run on the observatory Windows PC with EZCAP and all QHY apps closed::

    python scripts/scan_qhy_cameras.py

Optional: point at a specific DLL::

    set QHYCCD_DLL=C:\\Program Files\\QHYCCD\\SDK\\qhyccd.dll
    python scripts/scan_qhy_cameras.py
"""
from __future__ import annotations

import sys
from ctypes import c_char_p, c_uint32, create_string_buffer
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from libqhyccd import QHY_SUCCESS, load_qhyccd_library, resolve_qhyccd_library  # noqa: E402


def main() -> int:
    lib_path = resolve_qhyccd_library("")
    print(f"Using SDK: {lib_path}")

    lib = load_qhyccd_library(lib_path)
    res = lib.InitQHYCCDResource()
    if res != QHY_SUCCESS:
        print(f"InitQHYCCDResource failed: 0x{res:08x}")
        return 1

    try:
        count = lib.ScanQHYCCD()
        print(f"Found {count} camera(s)")
        if not count:
            print("No cameras - close EZCAP, replug USB, retry.")
            return 2

        for i in range(count):
            buf = create_string_buffer(64)
            res = lib.GetQHYCCDId(c_uint32(i), buf)
            if res == QHY_SUCCESS:
                cam_id = buf.value.decode(errors="replace")
                print(f"  [{i}] {cam_id}")
            else:
                print(f"  [{i}] GetQHYCCDId failed: 0x{res:08x}")
    finally:
        lib.ReleaseQHYCCDResource()

    print("\nCopy the serial fragment into config.windows.yaml serial_number if needed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
