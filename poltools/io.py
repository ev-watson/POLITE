"""
poltools.io — Polarimetry FITS input/output and sequence grouping.

Loads frames through ``caltools.load_frame`` (``memmap=False`` so unsigned
16-bit data with ``BZERO=32768`` scale correctly). Polarimetry FITS keywords
follow the POLITE acquisition convention.
"""

from __future__ import annotations

from collections import OrderedDict, defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
from astropy.io import fits

import caltools as ct
from alpyca_tools.fits_writer import (
    DetectorCards,
    FitsHeaderConfig,
    PolarimetryCards,
    build_header,
    write_fits,
)
from ._types import PolConfig

POL_KEYWORDS = {
    "HWPANG": "Half-wave-plate angle [deg]",
    "RETARD": "Retarder retardance delta [deg]",
    "INSTROT": "Field-rotator angle [deg]",
    "POLBEAM": "Beam(s) recorded (ordinary/extraordinary/dual)",
    "POLSEQ": "Polarimetry sequence identifier",
    "POLSEQN": "Index within polarimetry sequence",
    "POLEFF": "Polarization (modulation) efficiency",
    "PIXSCALE": "Plate scale [arcsec/pixel]",
    "READMODE": "Readout mode index",
    "SET-TEMP": "Cooler setpoint [C]",
    "SAVMAT": "Savart-plate material",
    "SAVTHK": "Savart-plate thickness [mm]",
    "WAVELEN": "Filter effective wavelength [nm]",
    "WPUNCERT": "HWP angle uncertainty [deg]",
}

SAVART_MATERIAL = "alpha-BBO"
SAVART_THICKNESS_MM = 18.0


def write_pol_fits(
    path: Union[str, Path],
    data_adu: np.ndarray,
    hwp_deg: float,
    cfg: PolConfig,
    *,
    object_name: str = "SIM",
    exptime_s: float = 1.0,
    imagetyp: str = "LIGHT",
    seq_id: str = "SIM",
    seq_index: int = 0,
    efficiency: float = 1.0,
    date_obs: Optional[str] = None,
    extra: Optional[Dict[str, object]] = None,
) -> Path:
    """Write a polarimetry frame with standard metadata.

    Data are stored as ``uint16`` physical ADU with ``BZERO=32768`` using the
    same writer and header contract as live Alpaca acquisition.
    """
    out = Path(path).expanduser().resolve()
    out.parent.mkdir(parents=True, exist_ok=True)

    arr = np.clip(np.rint(data_adu), 0, (1 << cfg.sensor.bitdepth) - 1).astype(np.uint16)
    from astropy.time import Time

    header_cfg = FitsHeaderConfig(
        imagetyp=imagetyp,
        exposure_s=float(exptime_s),
        date_obs=date_obs or Time.now().isot,
        object_name=object_name,
        instrument=cfg.sensor.sensor_name,
        filter_name=cfg.filter_name,
        detector=DetectorCards(
            gain_setting=int(cfg.gain_setting),
            readout_mode=int("".join(ch for ch in cfg.readout_mode if ch.isdigit()) or 0),
            readout_mode_name=str(cfg.readout_mode),
            cooler_setpoint_c=float(cfg.sensor.temperature_c),
            pixel_size_um=float(cfg.sensor.pixel_size_um),
        ),
        polarimetry=PolarimetryCards(
            hwp_angle_deg=float(hwp_deg),
            retardance_deg=float(cfg.retardance_deg),
            instrument_rotator_deg=float(cfg.instrument_rotator_deg),
            pol_beam="dual",
            pol_seq_id=str(seq_id),
            pol_seq_index=int(seq_index),
            pol_efficiency=float(efficiency),
            beam_sep_px=float(cfg.beam.separation_px),
            beam_pa_deg=float(cfg.beam.position_angle_deg),
            savart_material=SAVART_MATERIAL,
            savart_thickness_mm=SAVART_THICKNESS_MM,
        ),
        extra_cards={"PIXSCALE": (float(cfg.plate_scale_arcsec), POL_KEYWORDS["PIXSCALE"])},
    )
    _fcfg = cfg.active_filter()
    if _fcfg is not None and _fcfg.eff_wavelength_nm is not None:
        header_cfg.polarimetry.eff_wavelength_nm = float(_fcfg.eff_wavelength_nm)

    if extra:
        header_cfg.extra_cards.update(extra)

    header = build_header(None, header_cfg, data_dtype=arr.dtype, shape=arr.shape)
    return write_fits(out, arr, header, add_checksum=header_cfg.add_checksum)


def _read_hwp_angle(hdr: fits.Header, path: Union[str, Path]) -> float:
    if "HWPANG" not in hdr:
        raise ValueError(f"{path}: missing required FITS keyword 'HWPANG'")
    ang = float(hdr["HWPANG"])
    if not np.isfinite(ang):
        raise ValueError(f"{path}: non-finite HWPANG ({hdr['HWPANG']!r})")
    return ang


def read_pol_frame(path: Union[str, Path], roi=None
                   ) -> Tuple[np.ndarray, float, fits.Header]:
    """Read a polarimetry frame. Returns ``(data_float32, hwp_deg, header)``."""
    data = ct.load_frame(str(path), roi=roi)
    hdr = fits.getheader(str(path))
    hwp = _read_hwp_angle(hdr, path)
    return data, hwp, hdr


def group_by_hwp_angle(paths: List[Union[str, Path]]) -> Dict[float, List[str]]:
    """Group frame paths by HWP angle from FITS headers."""
    groups: Dict[float, List[str]] = defaultdict(list)
    for p in paths:
        hdr = fits.getheader(str(p))
        groups[round(_read_hwp_angle(hdr, p), 3)].append(str(p))
    return {k: sorted(v) for k, v in groups.items()}


def group_by_filter(paths: List[Union[str, Path]]) -> Dict[str, List[str]]:
    """Group frame paths by FITS ``FILTER`` card.

    Missing filter → ``"UNKNOWN"`` bucket.
    """
    groups: Dict[str, List[str]] = defaultdict(list)
    for p in paths:
        hdr = fits.getheader(str(p))
        name = str(hdr.get("FILTER", "UNKNOWN")).strip() or "UNKNOWN"
        groups[name].append(str(p))
    return {k: sorted(v) for k, v in groups.items()}


def group_pol_sequence(
    paths: List[Union[str, Path]],
) -> "OrderedDict[float, List[str]]":
    """Order one sequence by HWP angle without dropping repeat frames."""
    groups = group_by_hwp_angle(paths)
    return OrderedDict((angle, groups[angle]) for angle in sorted(groups))


def group_by_pol_sequence(
    paths: List[Union[str, Path]],
) -> Dict[Tuple[str, str, str], List[str]]:
    """Group paths by ``(OBJECT, POLSEQ, FILTER)`` provenance.

    Missing ``POLSEQ`` is represented explicitly as ``"UNKNOWN"`` so callers
    cannot accidentally merge unrelated unlabelled observations.
    """
    groups: Dict[Tuple[str, str, str], List[str]] = defaultdict(list)
    for p in paths:
        hdr = fits.getheader(str(p), ignore_missing_end=True)
        key = (
            str(hdr.get("OBJECT", "UNKNOWN")),
            str(hdr.get("POLSEQ", "UNKNOWN")),
            str(hdr.get("FILTER", "UNKNOWN")),
        )
        groups[key].append(str(p))
    return {key: sorted(values) for key, values in groups.items()}


def load_pol_config_sidecar(path: Union[str, Path], *, filter_name: Optional[str] = None) -> PolConfig:
    """Load :class:`PolConfig` from a capture-session ``pol_config.yaml`` sidecar."""
    from .pol_config import load_pol_config_sidecar as _load
    return _load(path, filter_name=filter_name)
