from __future__ import annotations

import struct
import time
from dataclasses import dataclass
from typing import Any, Callable, Optional, Tuple

import numpy as np
from alpaca.camera import ImageArrayElementTypes, ImageMetadata
from alpaca.device import Device

from .camera_device import CameraDevice
from .schema import CameraState

# Alpaca ImageBytes transmission element type -> numpy dtype used to read the
# raw byte stream. Keyed by the integer TransmissionElementType from the
# 44-byte ImageBytes metadata header (ASCOM AlpacaImageBytes spec).
_XMSN_TO_NP: dict[int, np.dtype] = {
    1: np.dtype(np.int16),    # Int16
    2: np.dtype(np.int32),    # Int32
    3: np.dtype(np.float64),  # Double
    4: np.dtype(np.float32),  # Single
    5: np.dtype(np.uint64),   # UInt64
    6: np.dtype(np.uint8),    # Byte
    7: np.dtype(np.int64),    # Int64
    8: np.dtype(np.uint16),   # UInt16
    9: np.dtype(np.uint32),   # UInt32
}

_IMAGEBYTES_HEADER_LEN = 44


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


def _storage_dtype(
    elem_type: Any,
    max_adu: Optional[int] = None,
    prefer_uint16: bool = True,
) -> np.dtype:
    """Pick the numpy dtype used to *store* the image (FITS BITPIX)."""
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

    return np.dtype(dtype)


def image_array_to_numpy(
    image_array: list,
    info: Any,
    max_adu: Optional[int] = None,
    prefer_uint16: bool = True,
) -> Tuple[np.ndarray, np.dtype]:
    elem_type = getattr(info, "ImageElementType", None)
    rank = getattr(info, "Rank", None)

    dtype = _storage_dtype(elem_type, max_adu=max_adu, prefer_uint16=prefer_uint16)

    if rank == 2:
        nda = np.array(image_array, dtype=dtype).transpose()
    elif rank == 3:
        nda = np.array(image_array, dtype=dtype).transpose(2, 1, 0)
    else:
        nda = np.array(image_array, dtype=dtype)

    return nda, dtype


def _decode_imagebytes(
    content: bytes,
    max_adu: Optional[int] = None,
    prefer_uint16: bool = True,
) -> Tuple[np.ndarray, ImageMetadata, np.dtype]:
    """Decode a raw Alpaca ImageBytes payload directly into a numpy array.

    This deliberately bypasses ``alpaca.camera``'s ImageBytes-to-nested-list
    reconstruction, which decodes Int32 pixels with ``array.array('l')``. On
    64-bit macOS/Linux (LP64) ``'l'`` is 8 bytes rather than the assumed 4, so
    that path reads only half the pixels and returns a ragged list. Reading the
    stream with ``np.frombuffer`` and the true element size is both correct and
    far faster for full-frame science images.
    """
    if len(content) < _IMAGEBYTES_HEADER_LEN:
        raise ValueError(
            f"ImageBytes payload too short ({len(content)} bytes) for a metadata header"
        )

    hdr = struct.unpack("<11i", content[:_IMAGEBYTES_HEADER_LEN])
    (
        meta_version,
        err_num,
        _client_txn,
        _server_txn,
        data_start,
        image_elem_type,
        xmsn_elem_type,
        rank,
        dim1,
        dim2,
        dim3,
    ) = hdr

    if err_num != 0:
        msg = content[_IMAGEBYTES_HEADER_LEN:].decode("utf-8", errors="replace")
        raise RuntimeError(f"Alpaca ImageBytes error {err_num}: {msg}")

    if meta_version != 1:
        raise ValueError(f"Unsupported ImageBytes metadata version: {meta_version}")

    xmsn_np = _XMSN_TO_NP.get(xmsn_elem_type)
    if xmsn_np is None:
        raise ValueError(f"Unsupported ImageBytes transmission element type: {xmsn_elem_type}")

    if rank == 3:
        expected = dim1 * dim2 * dim3
    else:
        expected = dim1 * dim2

    flat = np.frombuffer(content, dtype=xmsn_np, count=expected, offset=data_start)

    # ASCOM ImageArray is [x][y(, plane)] in row-major order (Dim1=NumX,
    # Dim2=NumY). Transpose to (y, x[, plane]) so downstream FITS orientation
    # matches the JSON path in image_array_to_numpy().
    if rank == 3:
        arr = flat.reshape(dim1, dim2, dim3).transpose(2, 1, 0)
    else:
        arr = flat.reshape(dim1, dim2).transpose()

    storage = _storage_dtype(image_elem_type, max_adu=max_adu, prefer_uint16=prefer_uint16)
    data = arr.astype(storage, copy=True)

    info = ImageMetadata(meta_version, image_elem_type, xmsn_elem_type, rank, dim1, dim2, dim3)
    return data, info, storage


def _fetch_imagebytes(camera: CameraDevice, timeout_s: float = 120.0) -> Optional[bytes]:
    """Fetch the imagearray endpoint as ImageBytes; None if server sent JSON."""
    hdrs = {"Accept": "application/imagebytes"}
    if camera.address.startswith("[") and not camera.address.startswith("[::1]"):
        hdrs["Host"] = f'{camera.address.split("%")[0]}]'

    try:
        Device._ctid_lock.acquire()
        params = {
            "ClientTransactionID": f"{Device._client_trans_id}",
            "ClientID": f"{Device._client_id}",
        }
        response = camera.rqs.get(
            f"{camera.base_url}/imagearray",
            params=params,
            headers=hdrs,
            timeout=timeout_s,
        )
        Device._client_trans_id += 1
    finally:
        Device._ctid_lock.release()

    if response.status_code not in range(200, 204):
        raise RuntimeError(
            f"ImageArray request failed: {response.status_code} {response.reason} "
            f"(URL {response.url})"
        )

    ct = (response.headers.get("content-type") or "").lower()
    if "application/imagebytes" not in ct:
        return None
    return response.content


def download_image(
    camera: CameraDevice,
    timeout_s: float = 120.0,
) -> Tuple[np.ndarray, Any, np.dtype]:
    content = _fetch_imagebytes(camera, timeout_s=timeout_s)
    if content is not None:
        data, info, dtype = _decode_imagebytes(content, max_adu=camera.MaxADU)
        camera.img_desc = info
        return data, info, dtype

    # Fallback for servers that only speak JSON ImageArray.
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
