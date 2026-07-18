"""Compatibility wrapper for the POLITE FITS acquisition writer.

New acquisition code should import from :mod:`alpyca_tools.fits_writer`.
This module remains for older scripts that use ``CaptureConfig`` and the
``obs_utils`` capture entry point; it delegates all header construction and
FITS serialization to the authoritative writer and contains no schema copy.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Tuple, Union

import numpy as np
from astropy.io import fits

from alpyca_tools.fits_writer import (
    DetectorCards,
    FitsHeaderConfig,
    PolarimetryCards,
    build_header,
    write_fits,
)

if TYPE_CHECKING:
    from alpyca_tools.camera_device import CameraDevice


@dataclass
class CaptureConfig(FitsHeaderConfig):
    """Legacy capture options plus the shared :class:`FitsHeaderConfig`.

    ``host`` defaults to the POLITE-owned QHY Alpaca camera server. The
    separately configured EFW/HWP Alpaca endpoint is not used for pixels.
    """

    host: str = "localhost:11112"
    camera_index: int = 0
    exposure_s: float = 1.0
    is_light: bool = True
    binx: int = 1
    biny: int = 1
    startx: int = 0
    starty: int = 0
    numx: int | None = None
    numy: int | None = None
    poll_s: float = 0.5
    timeout_s: float = 300.0
    verbose: bool = True


def _alpaca_image_to_numpy(camera: CameraDevice) -> Tuple[np.ndarray, np.dtype]:
    """Decode Alpaca ImageBytes through the shared camera transport helper."""
    from alpyca_tools.camera_ops import download_image

    data, _info, dtype = download_image(camera)
    return data, dtype


def _build_header(
    camera: CameraDevice,
    cfg: CaptureConfig,
    data_dtype: np.dtype,
    shape: Tuple[int, ...],
) -> fits.Header:
    """Compatibility name delegating to ``alpyca_tools.fits_writer``."""
    return build_header(camera, cfg, data_dtype=data_dtype, shape=shape)


def capture_fits(cfg: CaptureConfig, out_file: Union[str, Path]) -> Path:
    """Capture one frame using the authoritative POLITE FITS writer."""
    from alpyca_tools.camera_device import CameraDevice
    from alpyca_tools.camera_ops import ExposureSettings, capture_image

    camera = CameraDevice(cfg.host, cfg.camera_index)
    camera.Connected = True
    try:
        exposure = ExposureSettings(
            exposure_s=float(cfg.exposure_s),
            is_light=bool(cfg.is_light),
            binx=int(cfg.binx),
            biny=int(cfg.biny),
            startx=int(cfg.startx),
            starty=int(cfg.starty),
            numx=cfg.numx,
            numy=cfg.numy,
        )
        data, _info, dtype = capture_image(
            camera,
            exposure,
            poll_s=float(cfg.poll_s),
            timeout_s=float(cfg.timeout_s),
        )
        header = build_header(camera, cfg, data_dtype=dtype, shape=data.shape)
        return write_fits(out_file, data, header, add_checksum=cfg.add_checksum)
    finally:
        try:
            camera.Connected = False
        except Exception:
            pass


__all__ = [
    "CaptureConfig",
    "DetectorCards",
    "FitsHeaderConfig",
    "PolarimetryCards",
    "_build_header",
    "capture_fits",
]
