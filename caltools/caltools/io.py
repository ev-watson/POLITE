"""
caltools.io — FITS loading, file grouping, and header parsing.

Uses ``memmap=False`` on reads so astropy applies BSCALE/BZERO scaling
in-memory. Some FITS writers store unsigned integer image data with
``BZERO=32768``; applying the scaling on read avoids unscaled-view
pitfalls in downstream array operations.
"""

from __future__ import annotations

import os
import re
import warnings
from collections import defaultdict
from typing import Dict, Generator, List, Optional, Tuple

import numpy as np
from astropy.io import fits

from ._types import Frame, FrameCube, ROI, SensorConfig

# TheSkyX naming convention: 8-digit index + ImageType + ExptimeSecs
_FILENAME_RE = re.compile(r'^(\d{8})(Bias|Dark|FlatField|Light)([\d.]+)secs')


def load_frame(
    path: str,
    roi: Optional[ROI] = None,
    dtype: type = np.float32,
) -> Frame:
    """Load a single FITS frame.

    Uses ``memmap=False`` so astropy applies BSCALE/BZERO scaling
    in-memory at read time, avoiding unscaled-view pitfalls when the
    header stores unsigned integer data with ``BZERO``.

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
    if not paths:
        raise ValueError("load_cube: paths is empty")

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
    """Yield ``(row_slice, sub_cube)`` chunks of the cube row-strip by row-strip.

    Each yielded sub_cube has shape ``(n_frames, chunk_height, nx)``. The
    full cube is loaded once; chunking limits per-iteration allocations
    in downstream reducers (e.g. master_bias) but does not bound the
    cube's own memory — for that, downsample paths or use ROI.

    Parameters
    ----------
    paths : list of str
        FITS file paths.
    chunk_rows : int
        Number of rows per chunk (default 50).
    roi : ROI, optional
        Global ROI applied before chunking.
    dtype : type
        Output dtype.
    """
    if not paths:
        raise ValueError("load_cube_chunked: paths is empty")

    cube = load_cube(paths, roi=roi, dtype=dtype)
    ny = cube.shape[1]
    for r0 in range(0, ny, chunk_rows):
        r1 = min(r0 + chunk_rows, ny)
        yield slice(r0, r1), cube[:, r0:r1, :]


def sensor_config_from_header(
    path: str,
    gain: float = 1.0,
    pixel_size_um: Optional[float] = None,
) -> SensorConfig:
    """Build a ``SensorConfig`` from FITS header keywords.

    Parameters
    ----------
    path : str
        Path to any FITS file from the session.
    gain : float
        Conversion gain in e-/ADU. Many acquisition systems do not write
        a reliable ``GAIN`` keyword, so this must be supplied or measured
        separately.
    pixel_size_um : float, optional
        Override pixel pitch when ``XPIXSZ`` is missing. If both are
        absent, raises rather than guessing.

    Notes
    -----
    Detector temperature falls back to NaN (not 0 C) when ``CCD-TEMP`` is
    absent, so downstream code can detect missing data instead of
    silently treating it as a real reading.
    """
    hdr = fits.getheader(path)

    if "XPIXSZ" in hdr:
        pix = float(hdr["XPIXSZ"])
    elif pixel_size_um is not None:
        pix = float(pixel_size_um)
    else:
        raise KeyError(
            "FITS header is missing XPIXSZ; pass pixel_size_um explicitly."
        )

    temp = float(hdr["CCD-TEMP"]) if "CCD-TEMP" in hdr else float("nan")

    return SensorConfig(
        nx=int(hdr["NAXIS1"]),
        ny=int(hdr["NAXIS2"]),
        pixel_size_um=pix,
        gain_e_per_adu=gain,
        temperature_c=temp,
        bitdepth=int(hdr.get("BITPIX", 16)),
        sensor_name=str(hdr.get("INSTRUME", "Unknown")),
    )


def group_by_type_and_exposure(
    paths: List[str],
) -> Dict[Tuple[str, float], List[str]]:
    """Group FITS files by image type and exposure time.

    Primary: parse the TheSkyX filename convention
    ``########TypeExptimesecs``.
    Fallback: read IMAGETYP and EXPTIME from FITS headers.

    Returns a dict keyed by ``(image_type, exptime)`` with sorted file lists.
    """
    groups: Dict[Tuple[str, float], List[str]] = defaultdict(list)

    for p in sorted(paths):
        basename = os.path.basename(p)
        m = _FILENAME_RE.match(basename)
        if m:
            itype = m.group(2)
            exp = float(m.group(3))
        else:
            # Fallback to FITS header
            hdr = fits.getheader(p)
            itype = str(hdr.get("IMAGETYP", "Unknown"))
            exp = float(hdr.get("EXPTIME", 0.0))
        groups[(itype, exp)].append(p)

    # Sort file lists within each group
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


def get_file_index(path: str) -> Optional[int]:
    """Extract the 8-digit sequence index from a TheSkyX filename."""
    basename = os.path.basename(path)
    m = _FILENAME_RE.match(basename)
    if m:
        return int(m.group(1))
    return None
