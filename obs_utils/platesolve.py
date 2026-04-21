from __future__ import annotations

import platform
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


@dataclass
class PlateSolveConfig:
    ps3cli_exe: Path = Path("~/ps3cli/ps3cli.exe").expanduser()
    catalog_path: Optional[Path] = None


def _is_linux() -> bool:
    return platform.system() == "Linux"


def default_catalog_location() -> Path:
    if _is_linux():
        return Path("~/Kepler").expanduser()
    return Path("~\\Documents\\Kepler").expanduser()


def platesolve(
    image_file: Path,
    arcsec_per_pixel: float,
    config: PlateSolveConfig,
) -> dict[str, float]:
    output_file = Path(tempfile.gettempdir()) / "ps3cli_results.txt"

    catalog_path = config.catalog_path or default_catalog_location()

    ps3cli_exe = config.ps3cli_exe.expanduser()
    args = [
        str(ps3cli_exe),
        str(image_file),
        str(arcsec_per_pixel),
        str(output_file),
        str(catalog_path),
    ]

    if _is_linux():
        args.insert(0, "mono")

    result = subprocess.run(args, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(
            "PlateSolve failed with code "
            f"{result.returncode}: {result.stderr.strip()}"
        )

    return _parse_output(output_file)


def _parse_output(output_file: Path) -> dict[str, float]:
    results: dict[str, float] = {}
    with output_file.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            fields = line.split("=")
            if len(fields) != 2:
                continue
            key, value = fields
            results[key] = float(value)
    return results
