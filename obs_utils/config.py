from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class Pwi4Config:
    host: str = "localhost"
    port: int = 8220


@dataclass
class AlpacaConfig:
    host: str = "localhost:11111"
    camera_index: int = 0  # ASCOM Remote Server device index for SBIGImagingCamera
    guide_camera_index: Optional[int] = 1  # ASCOM Remote Server device index for SBIGGuidingCamera
    filterwheel_index: Optional[int] = 0  # ASCOM Remote Server device index for SBIG filter wheel
    filter_names: List[str] = field(
        default_factory=lambda: ["Clear/Luminance", "Red", "Blue", "Green", "Halpha"]
    )


@dataclass
class SkyRegionLimit:
    name: str
    alt_min_deg: float
    alt_max_deg: float
    az_min_deg: float
    az_max_deg: float


@dataclass
class SlewLimits:
    max_slew_rate_deg_s: Optional[float] = 5.0
    enforce_rate: bool = False
    regions: List[SkyRegionLimit] = field(default_factory=list)
    enforce_regions: bool = True


def default_sky_regions() -> List[SkyRegionLimit]:
    return [
        SkyRegionLimit(
            name="placeholder_sky_box",
            alt_min_deg=15.0,
            alt_max_deg=85.0,
            az_min_deg=0.0,
            az_max_deg=360.0,
        )
    ]
