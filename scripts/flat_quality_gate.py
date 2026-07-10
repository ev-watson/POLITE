#!/usr/bin/env python3
"""Flat quality gate: lsq vs double_ratio discrepancy (plan 5.3)."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from obs_utils.qa_lib import run_flat_quality_gate


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("paths", nargs="+", help="Polarized standard FITS files or directory")
    p.add_argument("--pol-config", type=Path, default=None)
    p.add_argument("--qu-tol", type=float, default=0.001)
    args = p.parse_args()
    result = run_flat_quality_gate(
        args.paths,
        pol_config_path=args.pol_config,
        qu_tol=args.qu_tol,
    )
    print(result.to_json())
    return 0 if result.passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
