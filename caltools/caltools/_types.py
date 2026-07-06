"""
caltools._types — Core data structures and type aliases.

Provides SensorConfig (frozen dataclass for detector parameters),
AnalysisResult (uniform return type for all analyses), and
type aliases for frame arrays.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Tuple

import numpy as np

# ---------------------------------------------------------------------------
# Type aliases
# ---------------------------------------------------------------------------
Frame = np.ndarray       # 2-D array (ny, nx)
FrameCube = np.ndarray   # 3-D array (n_frames, ny, nx)
ROI = Tuple[slice, slice]  # (row_slice, col_slice)


# ---------------------------------------------------------------------------
# SensorConfig
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class SensorConfig:
    """Immutable detector configuration.

    Parameters
    ----------
    nx, ny : int
        Detector dimensions in pixels (columns, rows).
    pixel_size_um : float
        Pixel pitch in micrometres.
    gain_e_per_adu : float
        Conversion gain (e-/ADU).
    temperature_c : float
        Detector temperature during acquisition (deg C).
    bitdepth : int
        ADC bit depth.
    sensor_name : str
        Detector identifier string.
    """

    nx: int
    ny: int
    pixel_size_um: float
    gain_e_per_adu: float
    temperature_c: float
    bitdepth: int = 16
    sensor_name: str = "Unknown"

    def central_roi(self, height: int = 2000, width: int = 3000) -> ROI:
        """Return a centred ROI as ``(row_slice, col_slice)``."""
        cy, cx = self.ny // 2, self.nx // 2
        ry = slice(cy - height // 2, cy + height // 2)
        rx = slice(cx - width // 2, cx + width // 2)
        return ry, rx

    def with_gain(self, gain_e_per_adu: float) -> SensorConfig:
        """Return a new config with updated gain (frozen — no mutation)."""
        return SensorConfig(
            nx=self.nx,
            ny=self.ny,
            pixel_size_um=self.pixel_size_um,
            gain_e_per_adu=gain_e_per_adu,
            temperature_c=self.temperature_c,
            bitdepth=self.bitdepth,
            sensor_name=self.sensor_name,
        )


# ---------------------------------------------------------------------------
# AnalysisResult
# ---------------------------------------------------------------------------
@dataclass
class AnalysisResult:
    """Uniform return container for all caltools analyses.

    Parameters
    ----------
    name : str
        Human-readable analysis name (e.g. ``"read_noise"``).
    scalar_summary : dict
        Key scalar results (e.g. ``{"ron_adu": 7.1, "ron_e": 3.5}``).
    maps : dict
        2-D or 3-D array results keyed by name.
    metadata : dict
        Auxiliary information (method, N frames, ROI used, etc.).
    """

    name: str
    scalar_summary: Dict[str, float] = field(default_factory=dict)
    maps: Dict[str, np.ndarray] = field(default_factory=dict)
    metadata: Dict[str, object] = field(default_factory=dict)

    def __repr__(self) -> str:
        scalars = ", ".join(f"{k}={v}" for k, v in self.scalar_summary.items())
        maps_keys = list(self.maps.keys())
        return (
            f"AnalysisResult('{self.name}', "
            f"scalars=[{scalars}], "
            f"maps={maps_keys})"
        )
