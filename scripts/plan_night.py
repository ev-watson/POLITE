#!/usr/bin/env python3
"""Preview a declarative brick-based night plan (no hardware, no capture).

    scripts/plan_night.py night_plans/example.yaml

Expands the palette bricks into the concrete frame timeline (with an
open-shutter exposure total) so you can eyeball a plan before committing.
This tool NEVER touches hardware.

To actually run a plan, use the no-mount runner, which owns the settings
banner, cooler gate, and per-invocation output subdirs::

    scripts/execute_night.py night_plans/<plan>.yaml --run

See scripts/README.md and obs_utils/night_plan.py.
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
    return p.parse_args()


def main() -> int:
    args = parse_args()
    try:
        config = load_night_plan(args.plan)
    except (NightPlanError, FileNotFoundError) as exc:
        print(f"plan error: {exc}", file=sys.stderr)
        return 2

    print(describe(config))
    print("\n(preview only; run with scripts/execute_night.py <plan> --run)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
