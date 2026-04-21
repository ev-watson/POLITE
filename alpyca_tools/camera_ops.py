from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Callable, Optional, Tuple

import numpy as np
from alpaca.camera import ImageArrayElementTypes

from .camera_device import CameraDevice
from .schema import CameraState


@dataclass
class ExposureSettings:
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


def configure_camera(camera: CameraDevice, settings: ExposureSettings) -> None:
    camera.BinX = int(settings.binx)
    camera.BinY = int(settings.biny)
    camera.StartX = int(settings.startx)
    camera.StartY = int(settings.starty)

    if settings.numx is None:
        camera.NumX = int(camera.CameraXSize // camera.BinX)
    else:
        camera.NumX = int(settings.numx)

    if settings.numy is None:
        camera.NumY = int(camera.CameraYSize // camera.BinY)
    else:
        camera.NumY = int(settings.numy)

    if settings.gain is not None:
        camera.Gain = int(settings.gain)

    if settings.offset is not None:
        camera.Offset = int(settings.offset)

    if settings.readout_mode is not None:
        camera.ReadoutMode = int(settings.readout_mode)

    if settings.sub_exposure_duration is not None:
        camera.SubExposureDuration = float(settings.sub_exposure_duration)

    if settings.fast_readout is not None:
        camera.FastReadout = bool(settings.fast_readout)


def wait_ready(
    camera: CameraDevice,
    poll_s: float = 0.5,
    timeout_s: float = 300.0,
    progress_cb: Optional[Callable[[int], None]] = None,
) -> None:
    t0 = time.time()
    last_pct = None
    while not camera.ImageReady:
        try:
            state = int(camera.CameraState)
        except Exception:
            state = None
        if state == CameraState.Error:
            raise RuntimeError("Camera entered error state during exposure")

        if (time.time() - t0) > timeout_s:
            raise TimeoutError(f"Exposure timed out after {timeout_s} s")

        try:
            pct = int(camera.PercentCompleted)
        except Exception:
            pct = None

        if pct is not None and pct != last_pct:
            last_pct = pct
            if progress_cb:
                progress_cb(pct)

        time.sleep(poll_s)


def image_array_to_numpy(
    image_array: list,
    info: Any,
    max_adu: Optional[int] = None,
    prefer_uint16: bool = True,
) -> Tuple[np.ndarray, np.dtype]:
    elem_type = getattr(info, "ImageElementType", None)
    rank = getattr(info, "Rank", None)

    dtype = np.int32
    if elem_type == ImageArrayElementTypes.Int16:
        dtype = np.int16
    elif elem_type == ImageArrayElementTypes.Int32:
        dtype = np.int32
    elif elem_type == ImageArrayElementTypes.Double:
        dtype = np.float64
    elif elem_type == ImageArrayElementTypes.Single:
        dtype = np.float32
    elif elem_type == ImageArrayElementTypes.UInt16:
        dtype = np.uint16

    if prefer_uint16 and dtype == np.int32 and max_adu is not None and max_adu <= 65535:
        dtype = np.uint16

    if rank == 2:
        nda = np.array(image_array, dtype=dtype).transpose()
    elif rank == 3:
        nda = np.array(image_array, dtype=dtype).transpose(2, 1, 0)
    else:
        nda = np.array(image_array, dtype=dtype)

    return nda, np.dtype(dtype)


def download_image(camera: CameraDevice) -> Tuple[np.ndarray, Any, np.dtype]:
    image = camera.ImageArray
    info = camera.ImageArrayInfo
    data, dtype = image_array_to_numpy(image, info, max_adu=camera.MaxADU)
    return data, info, dtype


def capture_image(
    camera: CameraDevice,
    settings: ExposureSettings,
    poll_s: float = 0.5,
    timeout_s: float = 300.0,
    progress_cb: Optional[Callable[[int], None]] = None,
) -> Tuple[np.ndarray, Any, np.dtype]:
    configure_camera(camera, settings)
    camera.StartExposure(settings.exposure_s, settings.is_light)
    wait_ready(camera, poll_s=poll_s, timeout_s=timeout_s, progress_cb=progress_cb)
    return download_image(camera)
