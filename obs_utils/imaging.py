from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple, Union

import numpy as np

from alpyca_tools.camera_ops import ExposureSettings, capture_image
from alpyca_tools.fits_writer import FitsHeaderConfig, capture_fits

from .alpaca import ImagingSession, open_imaging_session, set_filter_position
from .config import AlpacaConfig


@dataclass
class CaptureRequest:
    exposure_s: float
    is_light: bool = True
    binx: int = 1
    biny: int = 1
    startx: int = 0
    starty: int = 0
    numx: Optional[int] = None
    numy: Optional[int] = None
    gain: Optional[int] = None
    offset: Optional[int] = None
    readout_mode: Optional[int] = None
    sub_exposure_duration: Optional[float] = None
    fast_readout: Optional[bool] = None


def open_session(config: AlpacaConfig) -> ImagingSession:
    return open_imaging_session(
        host=config.host,
        camera_index=config.camera_index,
        guide_camera_index=config.guide_camera_index,
        filterwheel_index=config.filterwheel_index,
        filter_names=config.filter_names,
    )


def close_session(session: ImagingSession) -> None:
    session.close()


def capture_image_array(
    session: ImagingSession,
    request: CaptureRequest,
    poll_s: float = 0.5,
    timeout_s: float = 300.0,
) -> Tuple[np.ndarray, np.dtype]:
    exposure = ExposureSettings(
        exposure_s=request.exposure_s,
        is_light=request.is_light,
        binx=request.binx,
        biny=request.biny,
        startx=request.startx,
        starty=request.starty,
        numx=request.numx,
        numy=request.numy,
        gain=request.gain,
        offset=request.offset,
        readout_mode=request.readout_mode,
        sub_exposure_duration=request.sub_exposure_duration,
        fast_readout=request.fast_readout,
    )

    data, _info, dtype = capture_image(
        session.camera,
        exposure,
        poll_s=poll_s,
        timeout_s=timeout_s,
    )
    return data, dtype


def capture_fits_file(
    session: ImagingSession,
    request: CaptureRequest,
    header: FitsHeaderConfig,
    out_path: Path,
    poll_s: float = 0.5,
    timeout_s: float = 300.0,
) -> Path:
    exposure = ExposureSettings(
        exposure_s=request.exposure_s,
        is_light=request.is_light,
        binx=request.binx,
        biny=request.biny,
        startx=request.startx,
        starty=request.starty,
        numx=request.numx,
        numy=request.numy,
        gain=request.gain,
        offset=request.offset,
        readout_mode=request.readout_mode,
        sub_exposure_duration=request.sub_exposure_duration,
        fast_readout=request.fast_readout,
    )

    return capture_fits(
        session.camera,
        exposure,
        header,
        out_path,
        poll_s=poll_s,
        timeout_s=timeout_s,
    )


def select_filter(
    session: ImagingSession,
    position: Union[int, str],
    poll_s: float = 0.5,
    timeout_s: float = 30.0,
) -> int:
    if session.filter_wheel is None:
        raise RuntimeError("No filter wheel connected")
    return set_filter_position(
        session.filter_wheel,
        position,
        names_override=session.filter_names,
        poll_s=poll_s,
        timeout_s=timeout_s,
    )
