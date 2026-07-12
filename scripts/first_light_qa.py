#!/usr/bin/env python3
"""First-light polarimetric QA gate after HD 154445 (plan 6.5)."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from obs_utils.qa_lib import run_first_light_qa


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("paths", nargs="+", help="Science FITS files or directory")
    p.add_argument("--pol-config", type=Path, default=None)
    p.add_argument("--band", default="V")
    p.add_argument("--reference", default="HD 154445")
    p.add_argument("--abort", action="store_true")
    args = p.parse_args()
    result = run_first_light_qa(
        args.paths,
        pol_config_path=args.pol_config,
        ref_name=args.reference,
        band=args.band,
        abort_on_fail=args.abort,
    )
    print(result.to_json())
    return 0 if result.passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
