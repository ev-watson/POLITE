"""
poltools.pol_config - Session sidecar I/O for :class:`PolConfig`.

Standalone reduction helper: reads/writes ``pol_config.yaml`` produced by
observatory capture software. No dependency on observatory-specific packages.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import yaml
from astropy.io import fits

import caltools as ct
from caltools import SensorConfig

from ._types import BeamGeometry, PolConfig, default_efw_filters


@dataclass
class SessionDetectorConfig:
    """Detector settings serialized in a session ``pol_config.yaml``."""

    readout_mode: int = 0
    readout_mode_name: str = "Mode 0"
    gain_setting: int = 0
    offset: int = 0
    egain_e_per_adu: float = 1.0
    ron_e: float = 3.5
    cooler_setpoint_c: float = -10.0
    pixel_size_um: float = 3.76
    plate_scale_arcsec: float = 0.224
    sensor_name: str = "QHY268M"
    nx: int = 6280
    ny: int = 4210
    beam_separation_px: float = 60.0
    beam_position_angle_deg: float = 0.0
    beam_geometry_characterized: bool = False


def _detector_from_mapping(det: Dict[str, Any]) -> SessionDetectorConfig:
    return SessionDetectorConfig(
        readout_mode=int(det.get("readout_mode", 0)),
        readout_mode_name=str(det.get("readout_mode_name", "Mode 0")),
        gain_setting=int(det.get("gain_setting", det.get("gain", 0))),
        offset=int(det.get("offset", 0)),
        egain_e_per_adu=float(det.get("egain_e_per_adu", 1.0)),
        ron_e=float(det.get("ron_e", 3.5)),
        cooler_setpoint_c=float(det.get("cooler_setpoint_c", -10.0)),
        pixel_size_um=float(det.get("pixel_size_um", 3.76)),
        plate_scale_arcsec=float(det.get("plate_scale_arcsec", 0.224)),
        sensor_name=str(det.get("sensor_name", "QHY268M")),
        nx=int(det.get("nx", 6280)),
        ny=int(det.get("ny", 4210)),
        beam_separation_px=float(det.get("beam_separation_px", 60.0)),
        beam_position_angle_deg=float(det.get("beam_position_angle_deg", 0.0)),
        beam_geometry_characterized=bool(
            det.get("beam_geometry_characterized", False)
        ),
    )


def polconfig_from_detector(
    det: SessionDetectorConfig,
    filter_name: str,
    *,
    instrument_rotator_deg: float = 0.0,
    hwp_angles_deg: Optional[tuple] = None,
) -> PolConfig:
    """Build :class:`PolConfig` from session detector settings + filter name."""
    angles = hwp_angles_deg or (0.0, 22.5, 45.0, 67.5)
    sensor = SensorConfig(
        nx=det.nx,
        ny=det.ny,
        pixel_size_um=det.pixel_size_um,
        gain_e_per_adu=det.egain_e_per_adu,
        temperature_c=float("nan"),
        bitdepth=16,
        sensor_name=det.sensor_name,
    )
    filters = default_efw_filters(
        separation_px=det.beam_separation_px,
        position_angle_deg=det.beam_position_angle_deg,
        characterized=det.beam_geometry_characterized,
    )
    cfg = PolConfig(
        sensor=sensor,
        beam=BeamGeometry(
            separation_px=det.beam_separation_px,
            position_angle_deg=det.beam_position_angle_deg,
        ),
        plate_scale_arcsec=det.plate_scale_arcsec,
        hwp_angles_deg=tuple(float(a) for a in angles),
        instrument_rotator_deg=instrument_rotator_deg,
        filter_name=filter_name,
        filters=filters,
        read_noise_e=float(det.ron_e),
        readout_mode=det.readout_mode_name.replace(" ", ""),
        gain_setting=det.gain_setting,
    )
    try:
        return cfg.for_filter(filter_name)
    except KeyError:
        return cfg


def _resolve_filter_name(
    doc: Dict[str, Any],
    filter_name: Optional[str],
) -> str:
    if filter_name:
        return filter_name
    blocks = doc.get("blocks") or []
    for block in reversed(blocks):
        if isinstance(block, dict) and block.get("filter"):
            return str(block["filter"])
    return "Photometric V"


def load_pol_config_sidecar(
    path: Union[str, Path],
    *,
    filter_name: Optional[str] = None,
) -> PolConfig:
    """Load :class:`PolConfig` from a session ``pol_config.yaml`` sidecar."""
    with open(path, "r", encoding="utf-8") as fh:
        doc = yaml.safe_load(fh)
    if not isinstance(doc, dict):
        raise ValueError(f"{path}: sidecar must be a YAML mapping")
    det = _detector_from_mapping(doc.get("detector", {}))
    fname = _resolve_filter_name(doc, filter_name)
    return polconfig_from_detector(det, fname)


def write_pol_config_sidecar(
    path: Union[str, Path],
    detector: SessionDetectorConfig,
    *,
    session_id: str,
    blocks: Optional[List[Dict[str, Any]]] = None,
    created_utc: Optional[str] = None,
) -> Path:
    """Write or overwrite a session ``pol_config.yaml`` sidecar."""
    out = Path(path).expanduser().resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    det_dict = asdict(detector)
    doc: Dict[str, Any] = {
        "session": session_id,
        "created_utc": created_utc or datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "detector": det_dict,
        "blocks": blocks or [],
    }
    with open(out, "w", encoding="utf-8") as fh:
        yaml.safe_dump(doc, fh, sort_keys=False, default_flow_style=False)
    return out


def polconfig_from_fits_headers(
    path: Union[str, Path],
    *,
    filter_name: Optional[str] = None,
    gain_setting: Optional[int] = None,
    readout_mode: Optional[int] = None,
    egain_e_per_adu: Optional[float] = None,
    pixel_size_um: Optional[float] = None,
    sensor_name: Optional[str] = None,
    plate_scale_arcsec: Optional[float] = None,
    ron_e: Optional[float] = None,
    beam_separation_px: Optional[float] = None,
    beam_position_angle_deg: Optional[float] = None,
    instrument_rotator_deg: Optional[float] = None,
) -> PolConfig:
    """Build a :class:`PolConfig` from the shared FITS metadata contract.

    Required detector and polarimetry metadata come from the FITS header.
    Explicit keyword arguments are supported for deliberately incomplete
    commissioning frames; no camera-model values are inferred.
    """
    hdr = fits.getheader(str(path), ignore_missing_end=True)
    fname = filter_name or hdr.get("FILTER")
    if not fname:
        raise KeyError(f"{path}: FITS header missing FILTER; pass filter_name= explicitly")

    sensor = ct.sensor_config_from_header(
        str(path),
        gain=egain_e_per_adu,
        pixel_size_um=pixel_size_um,
        sensor_name=sensor_name,
    )

    def _required_or(value, card: str, label: str):
        if value is not None:
            return value
        if card not in hdr:
            raise KeyError(f"{path}: FITS header missing {card}; pass {label}= explicitly")
        return hdr[card]

    gain_idx = int(_required_or(gain_setting, "GAIN", "gain_setting"))
    mode_idx = int(_required_or(readout_mode, "READMODE", "readout_mode"))
    plate_scale = float(_required_or(plate_scale_arcsec, "PIXSCALE", "plate_scale_arcsec"))
    read_noise = float(_required_or(ron_e, "RON", "ron_e"))
    beam_sep = float(_required_or(beam_separation_px, "BEAMSEP", "beam_separation_px"))
    beam_pa = float(_required_or(beam_position_angle_deg, "BEAMPA", "beam_position_angle_deg"))
    instrot = float(_required_or(instrument_rotator_deg, "INSTROT", "instrument_rotator_deg"))

    det = SessionDetectorConfig(
        gain_setting=gain_idx,
        offset=int(hdr.get("OFFSET", 0)),
        egain_e_per_adu=sensor.gain_e_per_adu,
        readout_mode=mode_idx,
        readout_mode_name=str(hdr.get("RMODE", f"Mode {mode_idx}")),
        pixel_size_um=sensor.pixel_size_um,
        plate_scale_arcsec=plate_scale,
        ron_e=read_noise,
        sensor_name=sensor.sensor_name,
        nx=sensor.nx,
        ny=sensor.ny,
        beam_separation_px=beam_sep,
        beam_position_angle_deg=beam_pa,
        beam_geometry_characterized=True,
    )
    return polconfig_from_detector(det, fname, instrument_rotator_deg=instrot)


def polconfig_snapshot(cfg: PolConfig) -> Dict[str, Any]:
    """Key scalars from a :class:`PolConfig` for logging in sidecar blocks."""
    return {
        "filter_name": cfg.filter_name,
        "readout_mode": cfg.readout_mode,
        "gain_setting": cfg.gain_setting,
        "egain_e_per_adu": cfg.sensor.gain_e_per_adu,
        "ron_e": cfg.read_noise_e,
        "plate_scale_arcsec": cfg.plate_scale_arcsec,
        "instrument_rotator_deg": cfg.instrument_rotator_deg,
        "retardance_deg": cfg.retardance_deg,
        "beam_separation_px": cfg.beam.separation_px,
        "beam_position_angle_deg": cfg.beam.position_angle_deg,
        "beam_geometry_characterized": bool(
            cfg.active_filter() is not None and cfg.active_filter().characterized
        ),
    }
