"""Per-session detector settings and instrument metadata for FITS provenance."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Tuple

from poltools._types import FilterConfig, default_efw_filters


@dataclass
class SessionCaptureContext:
    """Immutable per-session detector settings + measured values for capture."""

    readout_mode: int = 0
    readout_mode_name: str = "Mode 0"
    gain_setting: int = 0
    offset_setting: int = 0
    egain_e_per_adu: float = 1.0
    ron_e: Optional[float] = 3.5
    cooler_setpoint_c: float = -15.0
    cooler_policy: str = "at_or_below"
    cooler_tolerance_c: float = 0.2
    cooler_stable_s: float = 30.0
    cooler_timeout_s: float = 600.0
    pixel_size_um: float = 3.76
    plate_scale_arcsec: float = 0.224
    observer: Optional[str] = None
    observatory: str = "Julian, CA"
    telescope: str = "CDK20"
    filters: Tuple[FilterConfig, ...] = field(default_factory=default_efw_filters)

    def filter_wavelength_nm(self, filter_name: Optional[str]) -> Optional[float]:
        if filter_name is None:
            return None
        for f in self.filters:
            if f.name == filter_name:
                return f.eff_wavelength_nm
        return None


def session_context_from_yaml(camera: Optional[Dict[str, Any]]) -> SessionCaptureContext:
    """Build a :class:`SessionCaptureContext` from a night-plan ``camera:`` block."""
    if not camera:
        return SessionCaptureContext()

    ctx = SessionCaptureContext()
    updates: Dict[str, Any] = {}

    if "readout_mode" in camera:
        updates["readout_mode"] = int(camera["readout_mode"])
    if "readout_mode_name" in camera:
        updates["readout_mode_name"] = str(camera["readout_mode_name"])
    if "gain" in camera:
        updates["gain_setting"] = int(camera["gain"])
    if "offset" in camera:
        updates["offset_setting"] = int(camera["offset"])
    if "egain_e_per_adu" in camera:
        updates["egain_e_per_adu"] = float(camera["egain_e_per_adu"])
    if "ron_e" in camera:
        updates["ron_e"] = float(camera["ron_e"])
    if "cooler_setpoint_c" in camera:
        updates["cooler_setpoint_c"] = float(camera["cooler_setpoint_c"])
    if "cooler_policy" in camera:
        policy = str(camera["cooler_policy"]).lower().replace("-", "_")
        if policy not in {"exact", "at_or_below"}:
            raise ValueError(
                "camera.cooler_policy must be 'exact' or 'at_or_below'"
            )
        updates["cooler_policy"] = policy
    if "cooler_tolerance_c" in camera:
        updates["cooler_tolerance_c"] = float(camera["cooler_tolerance_c"])
    if "cooler_stable_s" in camera:
        updates["cooler_stable_s"] = float(camera["cooler_stable_s"])
    if "cooler_timeout_s" in camera:
        updates["cooler_timeout_s"] = float(camera["cooler_timeout_s"])
    if "pixel_size_um" in camera:
        updates["pixel_size_um"] = float(camera["pixel_size_um"])
    if "plate_scale_arcsec" in camera:
        updates["plate_scale_arcsec"] = float(camera["plate_scale_arcsec"])
    if "observer" in camera:
        updates["observer"] = str(camera["observer"])
    if "observatory" in camera:
        updates["observatory"] = str(camera["observatory"])
    if "telescope" in camera:
        updates["telescope"] = str(camera["telescope"])

    return SessionCaptureContext(**{**ctx.__dict__, **updates})
