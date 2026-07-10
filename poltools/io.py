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
from astropy.time import Time

import caltools as ct
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
    "EGAIN": "Conversion gain [e-/ADU]",
    "READMODE": "Readout mode index",
    "SET-TEMP": "Cooler setpoint [C]",
    "SAVMAT": "Savart-plate material",
    "SAVTHK": "Savart-plate thickness [mm]",
    "WAVELEN": "Filter effective wavelength [nm]",
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

    Data are stored as ``uint16`` physical ADU with ``BZERO=32768``.
    """
    out = Path(path).expanduser().resolve()
    out.parent.mkdir(parents=True, exist_ok=True)

    arr = np.clip(np.rint(data_adu), 0, (1 << cfg.sensor.bitdepth) - 1).astype(np.uint16)
    hdr = fits.Header()
    hdr["COMMENT"] = "POLITE simulated polarimetry frame (poltools.simulate)"

    hdr["EXPTIME"] = (float(exptime_s), "Exposure time [s]")
    hdr["EXPOSURE"] = (float(exptime_s), "Exposure time [s]")
    hdr["DATE-OBS"] = (date_obs or Time.now().isot, "Exposure start (UTC)")
    hdr["TIMESYS"] = ("UTC", "Time system")
    hdr["IMAGETYP"] = (imagetyp.upper(), "Image type")
    hdr["OBJECT"] = (object_name, "Target name")
    hdr["INSTRUME"] = (cfg.sensor.sensor_name, "Instrument/sensor name")
    hdr["XPIXSZ"] = (float(cfg.sensor.pixel_size_um), "Pixel size [um]")
    hdr["YPIXSZ"] = (float(cfg.sensor.pixel_size_um), "Pixel size [um]")
    hdr["CCD-TEMP"] = (float(cfg.sensor.temperature_c), "Detector temperature [C]")
    hdr["FILTER"] = (cfg.filter_name, "Filter name")
    hdr["XBINNING"] = (1, "Binning factor in X")
    hdr["YBINNING"] = (1, "Binning factor in Y")

    hdr["EGAIN"] = (float(cfg.sensor.gain_e_per_adu), POL_KEYWORDS["EGAIN"])
    hdr["PIXSCALE"] = (float(cfg.plate_scale_arcsec), POL_KEYWORDS["PIXSCALE"])
    hdr["HWPANG"] = (float(hwp_deg), POL_KEYWORDS["HWPANG"])
    hdr["RETARD"] = (float(cfg.retardance_deg), POL_KEYWORDS["RETARD"])
    hdr["INSTROT"] = (float(cfg.instrument_rotator_deg), POL_KEYWORDS["INSTROT"])
    hdr["POLBEAM"] = ("dual", POL_KEYWORDS["POLBEAM"])
    hdr["POLSEQ"] = (str(seq_id), POL_KEYWORDS["POLSEQ"])
    hdr["POLSEQN"] = (int(seq_index), POL_KEYWORDS["POLSEQN"])
    hdr["POLEFF"] = (float(efficiency), POL_KEYWORDS["POLEFF"])
    hdr["BEAMSEP"] = (float(cfg.beam.separation_px), "o<->e beam separation [px]")
    hdr["BEAMPA"] = (float(cfg.beam.position_angle_deg), "o->e split position angle [deg]")
    hdr["SAVMAT"] = (SAVART_MATERIAL, POL_KEYWORDS["SAVMAT"])
    hdr["SAVTHK"] = (SAVART_THICKNESS_MM, POL_KEYWORDS["SAVTHK"])
    _fcfg = cfg.active_filter()
    if _fcfg is not None and _fcfg.eff_wavelength_nm is not None:
        hdr["WAVELEN"] = (float(_fcfg.eff_wavelength_nm), POL_KEYWORDS["WAVELEN"])

    if extra:
        for k, v in extra.items():
            hdr[k] = v

    hdr["HISTORY"] = "Created by poltools.io.write_pol_fits"
    hdu = fits.PrimaryHDU(data=arr, header=hdr)
    hdu.writeto(out, overwrite=True, output_verify="fix")
    return out


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


def group_pol_sequence(paths: List[Union[str, Path]]) -> "OrderedDict[float, str]":
    """Order frames of one sequence by HWP angle."""
    pairs = []
    for p in paths:
        hdr = fits.getheader(str(p))
        pairs.append((_read_hwp_angle(hdr, p), str(p)))
    pairs.sort(key=lambda t: t[0])
    return OrderedDict(pairs)


def load_pol_config_sidecar(path: Union[str, Path], *, filter_name: Optional[str] = None) -> PolConfig:
    """Load :class:`PolConfig` from a capture-session ``pol_config.yaml`` sidecar."""
    from .pol_config import load_pol_config_sidecar as _load
    return _load(path, filter_name=filter_name)
