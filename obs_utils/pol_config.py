"""Observatory capture helpers for pol_config sidecar files (wraps poltools)."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from poltools import (
    SessionDetectorConfig,
    load_pol_config_sidecar,
    nominal_beam_separation_px,
    polconfig_from_detector,
    polconfig_snapshot,
    write_pol_config_sidecar as _write_sidecar,
)
from poltools import PolConfig

from .session_context import SessionCaptureContext


def _ctx_to_detector(ctx: SessionCaptureContext) -> SessionDetectorConfig:
    # Fall back to the manufacturer-nominal Savart separation (0.9 mm ≈ 239 px on
    # the QHY268M), NOT an arbitrary placeholder, so an uncharacterized session
    # still writes a physically anchored geometry. Derive from the actual pixel
    # size when the context carries one.
    pixel_um = getattr(ctx, "pixel_size_um", None) or 3.76
    beam_sep = nominal_beam_separation_px(pixel_um)
    beam_pa = 0.0
    beam_characterized = False
    if ctx.filters:
        beam_sep = ctx.filters[0].beam.separation_px
        beam_pa = ctx.filters[0].beam.position_angle_deg
        beam_characterized = ctx.filters[0].characterized
    return SessionDetectorConfig(
        readout_mode=ctx.readout_mode,
        readout_mode_name=ctx.readout_mode_name,
        gain_setting=ctx.gain_setting,
        offset=ctx.offset_setting,
        egain_e_per_adu=ctx.egain_e_per_adu,
        ron_e=float(ctx.ron_e) if ctx.ron_e is not None else 3.5,
        cooler_setpoint_c=ctx.cooler_setpoint_c,
        pixel_size_um=ctx.pixel_size_um,
        plate_scale_arcsec=ctx.plate_scale_arcsec,
        beam_separation_px=beam_sep,
        beam_position_angle_deg=beam_pa,
        beam_geometry_characterized=beam_characterized,
    )


def session_pol_config_to_polconfig(
    ctx: SessionCaptureContext,
    filter_name: str,
    *,
    instrument_rotator_deg: float = 0.0,
    hwp_angles_deg: Optional[tuple] = None,
) -> PolConfig:
    """Map capture session context + filter to :class:`PolConfig`."""
    return polconfig_from_detector(
        _ctx_to_detector(ctx),
        filter_name,
        instrument_rotator_deg=instrument_rotator_deg,
        hwp_angles_deg=hwp_angles_deg,
    )


def write_pol_config_sidecar(
    path: Union[str, Path],
    ctx: SessionCaptureContext,
    *,
    session_id: str,
    blocks: Optional[List[Dict[str, Any]]] = None,
) -> Path:
    """Write ``pol_config.yaml`` from live capture context."""
    return _write_sidecar(
        path,
        _ctx_to_detector(ctx),
        session_id=session_id,
        blocks=blocks,
    )


__all__ = [
    "session_pol_config_to_polconfig",
    "write_pol_config_sidecar",
    "load_pol_config_sidecar",
    "polconfig_snapshot",
]
