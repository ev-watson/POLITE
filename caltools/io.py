"""
caltools.io — FITS loading, grouping, and header parsing.

Uses ``memmap=False`` so astropy applies BZERO/BSCALE scaling in memory
(required for QHY uint16 frames with BZERO=32768).
"""

from __future__ import annotations

import warnings
from collections import defaultdict
from contextlib import ExitStack
from typing import Dict, Generator, List, Optional, Tuple

import numpy as np
from astropy.io import fits

from ._types import Frame, FrameCube, ROI, SensorConfig

def load_frame(
    path: str,
    roi: Optional[ROI] = None,
    dtype: type = np.float32,
) -> Frame:
    """Load a single FITS frame.

    Uses ``memmap=False`` so astropy can apply in-memory scaling for
    FITS files that use BZERO/BSCALE.

    Parameters
    ----------
    path : str
        Path to FITS file.
    roi : ROI, optional
        ``(row_slice, col_slice)`` to crop.
    dtype : type
        Output dtype (default ``float32``).
    """
    data = fits.getdata(path, memmap=False).astype(dtype)
    if roi is not None:
        data = data[roi[0], roi[1]]
    return data


def load_cube(
    paths: List[str],
    roi: Optional[ROI] = None,
    dtype: type = np.float32,
) -> FrameCube:
    """Load multiple FITS frames into a 3-D cube ``(n, ny, nx)``.

    Warns if the resulting cube exceeds 4 GB.
    """
    first = load_frame(paths[0], roi=roi, dtype=dtype)
    ny, nx = first.shape
    n = len(paths)

    nbytes = n * ny * nx * np.dtype(dtype).itemsize
    if nbytes > 4e9:
        warnings.warn(
            f"Cube will be {nbytes / 1e9:.1f} GB — consider load_cube_chunked()",
            stacklevel=2,
        )

    cube = np.empty((n, ny, nx), dtype=dtype)
    cube[0] = first
    for i in range(1, n):
        cube[i] = load_frame(paths[i], roi=roi, dtype=dtype)
    return cube


def load_cube_chunked(
    paths: List[str],
    chunk_rows: int = 50,
    roi: Optional[ROI] = None,
    dtype: type = np.float32,
) -> Generator[Tuple[slice, FrameCube], None, None]:
    """Yield ``(row_slice, sub_cube)`` chunks for memory-bounded processing.

    Each yielded sub_cube has shape ``(n_frames, chunk_height, nx)``.

    Parameters
    ----------
    paths : list of str
        FITS file paths.
    chunk_rows : int
        Number of rows per chunk (default 50).
    roi : ROI, optional
        Global ROI applied before chunking (column slice preserved,
        row slice subdivided).
    dtype : type
        Output dtype.
    """
    if not paths:
        raise ValueError("paths must contain at least one FITS frame")
    if chunk_rows <= 0:
        raise ValueError(f"chunk_rows must be positive; got {chunk_rows}")

    header = fits.getheader(paths[0])
    full_ny = int(header["NAXIS2"])
    full_nx = int(header["NAXIS1"])
    row_sel, col_sel = roi or (slice(None), slice(None))
    rows = np.arange(full_ny)[row_sel]
    cols = np.arange(full_nx)[col_sel]
    ny, nx = len(rows), len(cols)
    n = len(paths)

    # Open every input once.  ``do_not_scale_image_data`` keeps uint16 FITS
    # (BITPIX=16, BZERO=32768) memory-mappable; scaling is applied only to each
    # requested strip below.  The previous implementation loaded every entire
    # 50 MB frame once per strip, turning a 25-frame master bias into ~100 GB of
    # avoidable I/O.
    with ExitStack() as stack:
        hdus = [
            stack.enter_context(
                fits.open(p, memmap=True, do_not_scale_image_data=True)
            )[0]
            for p in paths
        ]
        for p, hdu in zip(paths, hdus):
            if hdu.data is None or hdu.data.shape != (full_ny, full_nx):
                shape = None if hdu.data is None else hdu.data.shape
                raise ValueError(
                    f"{p}: frame shape {shape} != ({full_ny}, {full_nx})"
                )

        for r0 in range(0, ny, chunk_rows):
            r1 = min(r0 + chunk_rows, ny)
            row_sl = slice(r0, r1)
            row_idx = rows[row_sl]
            chunk = np.empty((n, r1 - r0, nx), dtype=dtype)
            selector = np.ix_(row_idx, cols)
            for i, hdu in enumerate(hdus):
                raw = np.asarray(hdu.data[selector], dtype=dtype)
                bscale = float(hdu.header.get("BSCALE", 1.0))
                bzero = float(hdu.header.get("BZERO", 0.0))
                if bscale != 1.0:
                    raw *= bscale
                if bzero != 0.0:
                    raw += bzero
                chunk[i] = raw
            yield row_sl, chunk


def sensor_config_from_header(
    path: str,
    gain: Optional[float] = None,
    pixel_size_um: Optional[float] = None,
    sensor_name: Optional[str] = None,
) -> SensorConfig:
    """Build a ``SensorConfig`` from FITS header keywords.

    Conversion gain is a per-night characterization value, not header state: an
    explicit ``gain=`` wins; a legacy ``EGAIN`` card is used only as a fallback;
    otherwise gain is left unset (``None``) for the analyst to supply at reduction.
    """
    hdr = fits.getheader(path)

    if gain is not None:
        gain_val = float(gain)
    elif "EGAIN" in hdr:  # legacy pre-2026-07 headers; advisory only
        gain_val = float(hdr["EGAIN"])
    else:
        gain_val = None

    if "XPIXSZ" in hdr:
        pix = float(hdr["XPIXSZ"])
    elif pixel_size_um is not None:
        pix = float(pixel_size_um)
    else:
        raise KeyError(
            "FITS header is missing XPIXSZ; pass pixel_size_um explicitly."
        )

    if "DET-TEMP" in hdr:
        temp = float(hdr["DET-TEMP"])
    elif "CCD-TEMP" in hdr:  # legacy spelling on pre-2026-07 data
        temp = float(hdr["CCD-TEMP"])
    else:
        temp = float("nan")
    if "INSTRUME" in hdr:
        detector_name = str(hdr["INSTRUME"])
    elif sensor_name is not None:
        detector_name = str(sensor_name)
        warnings.warn(
            f"{path}: FITS header missing INSTRUME; using supplied sensor_name={detector_name!r}",
            stacklevel=2,
        )
    else:
        raise KeyError(
            f"{path}: FITS header missing INSTRUME; pass sensor_name= explicitly."
        )

    return SensorConfig(
        nx=int(hdr["NAXIS1"]),
        ny=int(hdr["NAXIS2"]),
        pixel_size_um=pix,
        gain_e_per_adu=gain_val,
        temperature_c=temp,
        bitdepth=int(hdr.get("BITPIX", 16)),
        sensor_name=detector_name,
    )


def group_by_type_and_exposure(
    paths: List[str],
    *,
    exposure_decimals: int = 6,
) -> Dict[Tuple[str, float], List[str]]:
    """Group FITS files by image type and exposure time.

    Read ``IMAGETYP`` and ``EXPTIME`` from every FITS header.

    Filenames are deliberately ignored. POLITE files may use the standard
    ``YYYY-MM-DD_target_filter_exposure`` name, an arbitrary generated name, or
    a copied/archive name; acquisition metadata lives in the FITS header.

    Both cards are required. Silently assigning ``Unknown`` or ``0`` would
    merge malformed files into calibration groups and hide acquisition errors.

    Returns a dict keyed by ``(image_type, exptime)`` with sorted file lists.
    """
    groups: Dict[Tuple[str, float], List[str]] = defaultdict(list)

    for p in sorted(paths):
        hdr = fits.getheader(p)
        if "IMAGETYP" not in hdr:
            raise KeyError(f"{p}: FITS header missing required IMAGETYP")
        if "EXPTIME" not in hdr:
            raise KeyError(f"{p}: FITS header missing required EXPTIME")
        itype = str(hdr["IMAGETYP"]).strip().upper()
        if not itype:
            raise ValueError(f"{p}: FITS header IMAGETYP is empty")
        exp = float(hdr["EXPTIME"])
        if not np.isfinite(exp) or exp < 0:
            raise ValueError(f"{p}: FITS header EXPTIME must be finite and non-negative")
        # Camera APIs often return binary-floating artifacts such as
        # 0.2000000000109 s.  Canonicalizing at microsecond precision prevents
        # one requested exposure from fragmenting into several calibration
        # groups while retaining real millisecond-level differences (0.050 vs
        # 0.051 s).
        exp = round(exp, exposure_decimals)
        groups[(itype, exp)].append(p)

    return {k: sorted(v) for k, v in groups.items()}


def get_timestamps(paths: List[str]) -> np.ndarray:
    """Extract DATE-OBS from each file as ``datetime64`` array."""
    stamps = []
    for p in paths:
        hdr = fits.getheader(p)
        datestr = hdr.get("DATE-OBS", "")
        if datestr:
            stamps.append(np.datetime64(datestr))
        else:
            stamps.append(np.datetime64("NaT"))
    return np.array(stamps)
