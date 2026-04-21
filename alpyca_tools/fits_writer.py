from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union

import numpy as np
from astropy.io import fits

from .camera_device import CameraDevice
from .camera_ops import ExposureSettings, capture_image


@dataclass
class FitsHeaderConfig:
    imagetyp: str = "LIGHT"
    object_name: Optional[str] = None
    observer: Optional[str] = None
    telescope: Optional[str] = None
    observatory: Optional[str] = None
    instrument: Optional[str] = None
    filter_name: Optional[str] = None
    airmass: Optional[float] = None
    ra: Optional[str] = None
    dec: Optional[str] = None
    ha: Optional[str] = None
    equinox: Optional[float] = 2000.0
    wcs_cards: Dict[str, Any] = field(default_factory=dict)
    extra_cards: Dict[str, Any] = field(default_factory=dict)
    add_checksum: bool = True


def _set_card(hdr: fits.Header, key: str, value: Any, comment: Optional[str] = None) -> None:
    if value is None:
        return
    if comment is None:
        hdr[key] = value
    else:
        hdr[key] = (value, comment)


def build_header(
    camera: CameraDevice,
    cfg: FitsHeaderConfig,
    data_dtype: np.dtype,
    shape: Tuple[int, ...],
) -> fits.Header:
    hdr = fits.Header()
    hdr["COMMENT"] = "FITS header populated by alpyca_tools"

    if data_dtype == np.dtype(np.uint16):
        _set_card(hdr, "BZERO", 32768.0, "Data zero point")
        _set_card(hdr, "BSCALE", 1.0, "Data scale factor")

    _set_card(hdr, "EXPTIME", float(camera.LastExposureDuration), "Exposure time [s]")
    _set_card(hdr, "EXPOSURE", float(camera.LastExposureDuration), "Exposure time [s]")
    _set_card(hdr, "DATE-OBS", str(camera.LastExposureStartTime), "Exposure start time")
    _set_card(hdr, "TIMESYS", "UTC", "Time system")

    _set_card(hdr, "XBINNING", int(camera.BinX), "Binning factor in X")
    _set_card(hdr, "YBINNING", int(camera.BinY), "Binning factor in Y")

    _set_card(hdr, "IMAGETYP", cfg.imagetyp.upper(), "Image type (LIGHT/DARK/BIAS/FLAT)")
    _set_card(hdr, "OBSTYPE", cfg.imagetyp.upper(), "Image type")

    _set_card(hdr, "OBJECT", cfg.object_name, "Target name")
    _set_card(hdr, "OBSERVER", cfg.observer, "Observer")
    _set_card(hdr, "TELESCOP", cfg.telescope, "Telescope")
    _set_card(hdr, "OBSERVAT", cfg.observatory, "Observatory/site")

    instrume = cfg.instrument if cfg.instrument is not None else getattr(camera, "SensorName", None)
    _set_card(hdr, "INSTRUME", instrume, "Instrument/sensor name")

    try:
        _set_card(hdr, "GAIN", int(camera.Gain), "Camera gain setting")
    except Exception:
        pass

    try:
        off = camera.Offset
        if isinstance(off, int):
            _set_card(hdr, "OFFSET", int(off), "Camera offset setting")
            _set_card(hdr, "PEDESTAL", int(off), "Bias pedestal (if applicable)")
        else:
            _set_card(hdr, "OFFSET", off, "Camera offset setting")
    except Exception:
        pass

    try:
        _set_card(hdr, "CCD-TEMP", float(camera.CCDTemperature), "Detector temperature [C]")
    except Exception:
        pass

    _set_card(hdr, "FILTER", cfg.filter_name, "Filter name")

    _set_card(hdr, "AIRMASS", cfg.airmass, "Airmass at start")
    _set_card(hdr, "RA", cfg.ra, "Right Ascension (sexagesimal)")
    _set_card(hdr, "DEC", cfg.dec, "Declination (sexagesimal)")
    _set_card(hdr, "HA", cfg.ha, "Hour angle (sexagesimal)")
    if cfg.equinox is not None:
        _set_card(hdr, "EQUINOX", float(cfg.equinox), "Equinox of celestial coordinates")

    for k, v in (cfg.wcs_cards or {}).items():
        _set_card(hdr, k, v)

    for k, v in (cfg.extra_cards or {}).items():
        _set_card(hdr, k, v)

    _set_card(hdr, "HISTORY", "Created using alpyca_tools + astropy.io.fits")

    return hdr


def write_fits(
    out_file: Union[str, Path],
    data: np.ndarray,
    header: fits.Header,
    add_checksum: bool = True,
) -> Path:
    out_path = Path(out_file).expanduser().resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    hdu = fits.PrimaryHDU(data=data, header=header)
    hdul = fits.HDUList([hdu])

    if add_checksum:
        hdul.add_checksum()

    hdul.writeto(out_path, overwrite=True, output_verify="fix")
    return out_path


def capture_fits(
    camera: CameraDevice,
    exposure: ExposureSettings,
    header_cfg: FitsHeaderConfig,
    out_file: Union[str, Path],
    poll_s: float = 0.5,
    timeout_s: float = 300.0,
) -> Path:
    data, _info, dtype = capture_image(
        camera,
        exposure,
        poll_s=poll_s,
        timeout_s=timeout_s,
    )
    header = build_header(camera, header_cfg, data_dtype=dtype, shape=data.shape)
    return write_fits(out_file, data, header, add_checksum=header_cfg.add_checksum)
