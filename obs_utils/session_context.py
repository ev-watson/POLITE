"""Per-session detector settings and instrument metadata for FITS provenance."""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Tuple

from poltools._types import FilterConfig, default_efw_filters

from .block_log import JULIAN_ELEV_M, JULIAN_LAT_DEG, JULIAN_LON_DEG

logger = logging.getLogger(__name__)


@dataclass
class SessionCaptureContext:
    """Immutable per-session detector settings + measured values for capture.

    Detector default is **Mode 5, gain 56, offset 20** — the project operating
    point for polarimetry.  QHY reports Mode 5 as the QHY268M's lowest-read-noise
    mode; gain 56 is the lowest gain in that mode that still reaches the low-noise
    floor, so it buys the noise without spending more dynamic range than it has to
    (raising gain past this point keeps lowering read noise but costs full well and
    brings other effects with it).  Dynamic range is a first-order polarimetric
    concern, not a nicety: Cole (2010, SAS, ``reference-sheets/polarimetry/
    instruments/reno-commissioning.md``) §4 defocuses the PSF over ~6x6 px purely
    because the per-pixel well caps how many photons a single exposure can collect,
    and his §6 acquisition loop responds to a saturated max pixel by cutting the
    exposure 25 % and **restarting the whole waveplate sequence** — a saturated
    frame does not cost one frame, it costs the modulation set.

    **Mode 3, gain 0** is the other supported operating point: highest full well of
    any mode, and high (though lower than Mode 5 / gain 56) dynamic range.  Use it
    when well depth is the binding constraint.  Set it explicitly in the plan's
    ``camera:`` block; it is deliberately not the default.
    """

    readout_mode: int = 5
    readout_mode_name: str = "Mode 5"
    gain_setting: int = 56
    offset_setting: int = 20
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


def _cooler_policy(value: Any) -> str:
    policy = str(value).lower().replace("-", "_")
    if policy not in {"exact", "at_or_below"}:
        raise ValueError("camera.cooler_policy must be 'exact' or 'at_or_below'")
    return policy


# Accepted ``camera:`` keys -> (context field, coercion).  Anything outside this
# table is reported by name rather than silently dropped: a typo such as
# ``offest: 20`` used to leave the night at the default offset while the plan read
# as though it had been set.
_CAMERA_KEYS: Dict[str, Tuple[str, Any]] = {
    "readout_mode": ("readout_mode", int),
    "readout_mode_name": ("readout_mode_name", str),
    "gain": ("gain_setting", int),
    "offset": ("offset_setting", int),
    "cooler_setpoint_c": ("cooler_setpoint_c", float),
    "cooler_policy": ("cooler_policy", _cooler_policy),
    "cooler_tolerance_c": ("cooler_tolerance_c", float),
    "cooler_stable_s": ("cooler_stable_s", float),
    "cooler_timeout_s": ("cooler_timeout_s", float),
    "pixel_size_um": ("pixel_size_um", float),
    "plate_scale_arcsec": ("plate_scale_arcsec", float),
    "observer": ("observer", str),
    "observatory": ("observatory", str),
    "telescope": ("telescope", str),
    "origin": ("origin", str),
    "focal_length_mm": ("focal_length_mm", float),
    "focal_ratio": ("focal_ratio", float),
    "aperture_mm": ("aperture_mm", float),
    "hwp_uncert_deg": ("hwp_uncert_deg", float),
    "site_lat_deg": ("site_lat_deg", float),
    "site_lon_deg": ("site_lon_deg", float),
    "site_elev_m": ("site_elev_m", float),
}


def session_context_from_yaml(camera: Optional[Dict[str, Any]]) -> SessionCaptureContext:
    """Build a :class:`SessionCaptureContext` from a night-plan ``camera:`` block.

    An absent block and an unrecognized key both WARN and continue -- running on
    the defaults is a legitimate choice, but it must be a visible one.  Only a
    value that cannot mean anything (a bad ``cooler_policy``) raises.
    """
    ctx = SessionCaptureContext()
    if not camera:
        logger.warning(
            "No camera: block in plan -- using defaults: %s (readout_mode=%d), "
            "gain %d, offset %d, cooler %+.1f C. Set camera: to override.",
            ctx.readout_mode_name, ctx.readout_mode, ctx.gain_setting,
            ctx.offset_setting, ctx.cooler_setpoint_c,
        )
        return ctx

    unknown = [k for k in camera if k not in _CAMERA_KEYS]
    if unknown:
        logger.warning(
            "camera: has %d unrecognized key(s), IGNORED (check for typos): %s",
            len(unknown), ", ".join(sorted(unknown)),
        )
        logger.warning("camera: accepts: %s", ", ".join(sorted(_CAMERA_KEYS)))

    updates = {
        field_name: coerce(camera[key])
        for key, (field_name, coerce) in _CAMERA_KEYS.items()
        if key in camera
    }
    return SessionCaptureContext(**{**ctx.__dict__, **updates})
