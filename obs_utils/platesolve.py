"""PlateSolve3 adapter for supervised POLITE pointing commissioning.

PlaneWave PWI4 reports mount state and consumes solved sky coordinates for a
pointing model, but it is not the image solver.  This module runs PlaneWave's
``ps3cli`` against one existing camera FITS frame and returns its validated raw
numeric payload.  It has no camera or mount side effects.

The POLITE Savart analyzer doubles every star.  Whether PlateSolve3 can solve
such a raw field is unvalidated; use :func:`platesolve` first in the read-only
commissioning path and do not treat a failed solve as a reason to alter source
data automatically.
"""

from __future__ import annotations

import math
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping


@dataclass(frozen=True)
class PlateSolveConfig:
    """Local PlateSolve3 executable/catalog configuration.

    Both paths are site-supplied deliberately: the repository has no verified
    installation location or catalog path.  Set ``command_prefix=("mono",)``
    only on a host where that invocation has been tested.
    """

    ps3cli_exe: Path
    catalog_path: Path
    command_prefix: tuple[str, ...] = ()
    timeout_s: float = 120.0


@dataclass(frozen=True)
class PlateSolveResult:
    """Raw numeric payload returned by one PlateSolve3 invocation.

    The first real PS3CLI result has not yet been recorded for POLITE.  Keep
    its keys raw rather than assuming their spellings or units map to PWI4's
    J2000 model-point API.
    """

    raw_fields: Mapping[str, float]
    stdout: str
    stderr: str


def _validate_paths(
    image_file: Path,
    ps3cli_exe: Path,
    catalog_path: Path,
) -> None:
    if not image_file.is_file():
        raise FileNotFoundError(f"plate-solve frame does not exist: {image_file}")
    if not ps3cli_exe.is_file():
        raise FileNotFoundError(
            f"PlateSolve3 executable does not exist: {ps3cli_exe}\n"
            "Set PlateSolveConfig(ps3cli_exe=...) for this observatory PC."
        )
    if not catalog_path.is_dir():
        raise FileNotFoundError(
            f"PlateSolve3 Kepler catalog directory does not exist: {catalog_path}\n"
            "Set PlateSolveConfig(catalog_path=...) for this observatory PC."
        )


def platesolve(
    image_file: Path,
    arcsec_per_pixel: float,
    config: PlateSolveConfig,
    *,
    runner: Callable[..., Any] = subprocess.run,
) -> PlateSolveResult:
    """Solve one existing FITS frame with PlateSolve3.

    This call is read-only with respect to camera data and PWI4.  It only
    creates a unique temporary PS3CLI result file, which is removed after
    parsing.  The caller must explicitly decide whether a returned solution is
    suitable for a PWI4 model point.
    """
    image_file = Path(image_file).expanduser()

    if (
        not isinstance(arcsec_per_pixel, (int, float))
        or not math.isfinite(arcsec_per_pixel)
        or arcsec_per_pixel <= 0
    ):
        raise ValueError("arcsec_per_pixel must be a positive finite value")
    if (
        not isinstance(config.timeout_s, (int, float))
        or not math.isfinite(config.timeout_s)
        or config.timeout_s <= 0
    ):
        raise ValueError("PlateSolveConfig.timeout_s must be positive")
    ps3cli_exe = Path(config.ps3cli_exe).expanduser()
    catalog_path = Path(config.catalog_path).expanduser()
    _validate_paths(image_file, ps3cli_exe, catalog_path)

    with tempfile.TemporaryDirectory(prefix="polite_ps3_") as temp_dir:
        output_file = Path(temp_dir) / "ps3cli_results.txt"
        args = [*config.command_prefix,
            str(ps3cli_exe),
            str(image_file),
            str(float(arcsec_per_pixel)),
            str(output_file),
            str(catalog_path),
        ]
        try:
            result = runner(
                args, capture_output=True, text=True, timeout=config.timeout_s,
                check=False,
            )
        except subprocess.TimeoutExpired as exc:
            raise RuntimeError(
                f"PlateSolve3 exceeded its {config.timeout_s:.0f}-s timeout"
            ) from exc
        except OSError as exc:
            raise RuntimeError(f"could not start PlateSolve3: {exc}") from exc
        if result.returncode != 0:
            detail = (result.stderr or result.stdout).strip()
            raise RuntimeError(
                f"PlateSolve3 failed with code {result.returncode}: {detail}"
            )
        if not output_file.is_file():
            raise RuntimeError("PlateSolve3 completed without writing a result file")
        raw_fields = _parse_output(output_file)
        if not raw_fields:
            raise RuntimeError("PlateSolve3 wrote no numeric key=value result fields")
        return PlateSolveResult(
            raw_fields=raw_fields,
            stdout=result.stdout,
            stderr=result.stderr,
        )


def _parse_output(output_file: Path) -> dict[str, float]:
    """Read numeric ``key=value`` fields emitted by PS3CLI."""
    results: dict[str, float] = {}
    with output_file.open("r", encoding="utf-8") as handle:
        for line in handle:
            key, separator, value = line.strip().partition("=")
            if not separator or not key or not value:
                continue
            try:
                results[key] = float(value)
            except ValueError:
                continue
    return results
