#!/usr/bin/env python3
"""Bias histogram and read-noise QA (first-light plan 5.4 Lab 1)."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from obs_utils.qa_lib import run_bias_qa


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("paths", nargs="+", help="Bias FITS files or directory")
    p.add_argument("--ron-target", type=float, default=3.5)
    p.add_argument("--ron-tol", type=float, default=0.5)
    args = p.parse_args()
    result = run_bias_qa(args.paths, ron_target_e=args.ron_target, ron_tol_e=args.ron_tol)
    print(result.to_json())
    return 0 if result.passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
