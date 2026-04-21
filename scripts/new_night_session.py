#!/usr/bin/env python3
from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path


TEMPLATE = """from obs_utils.night_session import FramePlan, TargetPlan, NightSessionConfig, run_night_session
from obs_utils.startup import StartupConfig
from obs_utils.logging import LoggingConfig

try:
    from obs_utils.user_config import ALPACA_CONFIG, PWI4_CONFIG
except Exception:
    ALPACA_CONFIG = None
    PWI4_CONFIG = None


startup_kwargs = {}
if PWI4_CONFIG is not None:
    startup_kwargs[\"pwi4\"] = PWI4_CONFIG
if ALPACA_CONFIG is not None:
    startup_kwargs[\"alpaca\"] = ALPACA_CONFIG

config = NightSessionConfig(
    startup=StartupConfig(
        logging=LoggingConfig(session_name=\"{session_name}\"),
        **startup_kwargs,
    ),
    targets=[],
    calibration_frames=[],
    # Optional global metadata (auto-filled where possible):
    # observer=\"\",
    # telescope=\"\",
    # observatory=\"\",
    # instrument=\"\",
)

# Example target structure (copy and edit):
# config.targets = [
#     TargetPlan(
#         name=\"M42\",
#         ra_hours=5.591,  # J2000
#         dec_deg=-5.389,
#         frames=[
#             FramePlan(frame_type=\"LIGHT\", exposure_s=60.0, count=10, filter=\"L\"),
#         ],
#     ),
# ]
# config.calibration_frames = [
#     FramePlan(frame_type=\"DARK\", exposure_s=60.0, count=10),
#     FramePlan(frame_type=\"BIAS\", exposure_s=0.0, count=20),
# ]

run_night_session(config)
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create a blank night session script.")
    parser.add_argument("--out-dir", default="night_sessions", help="Directory for generated script")
    parser.add_argument("--session-name", help="Session name used for logs and filenames")
    parser.add_argument("--utc", action="store_true", help="Use UTC date for default session name")
    parser.add_argument("--force", action="store_true", help="Overwrite if file exists")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    now = datetime.now(timezone.utc) if args.utc else datetime.now()
    session_name = args.session_name or now.strftime("%Y%m%d")

    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    filename = f"night_session_{session_name}.py"
    out_path = out_dir / filename

    if out_path.exists() and not args.force:
        raise SystemExit(f"Refusing to overwrite existing file: {out_path}")

    out_path.write_text(TEMPLATE.format(session_name=session_name), encoding="utf-8")
    print(str(out_path))


if __name__ == "__main__":
    main()
