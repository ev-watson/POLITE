"""Per-session detector settings and instrument metadata for FITS provenance."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Tuple

from poltools._types import FilterConfig, default_efw_filters

from .block_log import JULIAN_ELEV_M, JULIAN_LAT_DEG, JULIAN_LON_DEG


@dataclass
class SessionCaptureContext:
    """Immutable per-session detector settings + measured values for capture."""

    readout_mode: int = 0
    readout_mode_name: str = "Mode 0"
    gain_setting: int = 0
    offset_setting: int = 0
    cooler_setpoint_c: float = -15.0
    cooler_policy: str = "at_or_below"
    cooler_tolerance_c: float = 0.2
    cooler_stable_s: float = 30.0
    cooler_timeout_s: float = 600.0
    pixel_size_um: float = 3.76
    plate_scale_arcsec: float = 0.224
    # Optics — PlaneWave CDK20 standard f/6.8 (consistent with plate_scale_arcsec)
    focal_length_mm: float = 3454.0
    focal_ratio: float = 6.8
    aperture_mm: float = 508.0
    # Half-wave-plate PA uncertainty — Pyxis Gen3 open-loop step quantization
    # (no encoder; commanded-vs-recorded error is step quantization ~0.012 deg).
    hwp_uncert_deg: float = 0.012
    observer: Optional[str] = None
    observatory: str = "Julian, CA"
    telescope: str = "CDK20"
    origin: str = "POLITE"
    # Site geolocation (single source: obs_utils.block_log)
    site_lat_deg: float = JULIAN_LAT_DEG
    site_lon_deg: float = JULIAN_LON_DEG
    site_elev_m: float = JULIAN_ELEV_M
    filters: Tuple[FilterConfig, ...] = field(default_factory=default_efw_filters)
    # NB: conversion gain (e-/ADU) and read noise (e-) intentionally absent — they
    # are per-night characterization results the analyst supplies at reduction time,
    # never as-acquired capture state.

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
    if "origin" in camera:
        updates["origin"] = str(camera["origin"])
    if "focal_length_mm" in camera:
        updates["focal_length_mm"] = float(camera["focal_length_mm"])
    if "focal_ratio" in camera:
        updates["focal_ratio"] = float(camera["focal_ratio"])
    if "aperture_mm" in camera:
        updates["aperture_mm"] = float(camera["aperture_mm"])
    if "hwp_uncert_deg" in camera:
        updates["hwp_uncert_deg"] = float(camera["hwp_uncert_deg"])
    if "site_lat_deg" in camera:
        updates["site_lat_deg"] = float(camera["site_lat_deg"])
    if "site_lon_deg" in camera:
        updates["site_lon_deg"] = float(camera["site_lon_deg"])
    if "site_elev_m" in camera:
        updates["site_elev_m"] = float(camera["site_elev_m"])

    return SessionCaptureContext(**{**ctx.__dict__, **updates})
