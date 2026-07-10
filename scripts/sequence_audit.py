#!/usr/bin/env python3
"""End-of-night HWP sequence completeness audit."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from obs_utils.qa_lib import run_sequence_audit


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("directory", help="Session FITS directory (e.g. FITSDATA/20260709)")
    args = p.parse_args()
    result = run_sequence_audit(args.directory)
    print(result.to_json())
    return 0 if result.passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
