#!/usr/bin/env python3
"""Load a declarative brick-based night plan and preview or run it.

    scripts/plan_night.py night_plans/example.yaml            # dry-run preview
    scripts/plan_night.py night_plans/example.yaml --run      # execute the night

Dry-run expands the palette bricks into the concrete frame timeline (with an
open-shutter exposure total) without touching hardware. ``--run`` hands the
resulting NightSessionConfig to ``run_night_session`` (needs INDIGO/Alpaca +
PWI4 up). See scripts/README.md and obs_utils/night_plan.py.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from obs_utils.night_plan import NightPlanError, describe, load_night_plan


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("plan", help="Path to a per-night plan YAML file")
    p.add_argument("--run", action="store_true",
                   help="Execute the night (default is a dry-run preview)")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    try:
        config = load_night_plan(args.plan)
    except (NightPlanError, FileNotFoundError) as exc:
        print(f"plan error: {exc}", file=sys.stderr)
        return 2

    print(describe(config))

    if not args.run:
        print("\n(dry-run; pass --run to execute)")
        return 0

    from obs_utils.night_session import run_night_session
    run_night_session(config)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
