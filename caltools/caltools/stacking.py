"""
caltools.stacking — Master frame construction (bias, dark, flat).

Combines calibration-frame stacks with median or mean reducers and
optional row-strip processing.
"""

from __future__ import annotations

from typing import List, Optional

import numpy as np

from ._types import Frame, ROI
from .io import load_cube_chunked, load_frame


def master_bias(
    paths: List[str],
    method: str = "median",
    chunk_rows: int = 50,
    roi: Optional[ROI] = None,
) -> Frame:
    """Construct a master bias from a stack of bias frames.

    Uses row-strip processing via ``load_cube_chunked``. Median combination
    is the default because it suppresses transient pixel excursions in
    larger stacks.

    Parameters
    ----------
    paths : list of str
        Bias frame file paths.
    method : ``"median"`` or ``"mean"``
        Combination method.
    chunk_rows : int
        Rows per processing chunk.
    roi : ROI, optional
        Restrict to a sub-region.
    """
    reduce_fn = np.median if method == "median" else np.mean

    # Determine output shape from first frame
    first = load_frame(paths[0], roi=roi)
    ny, nx = first.shape
    master = np.empty((ny, nx), dtype=np.float32)

    for row_sl, chunk in load_cube_chunked(paths, chunk_rows=chunk_rows, roi=roi):
        master[row_sl, :] = reduce_fn(chunk, axis=0).astype(np.float32)

    return master


def master_dark(
    paths: List[str],
    bias: Frame,
    method: str = "median",
    chunk_rows: int = 50,
    roi: Optional[ROI] = None,
) -> Frame:
    """Construct a bias-subtracted master dark.

    Parameters
    ----------
    paths : list of str
        Dark frame file paths (all same exposure time).
    bias : Frame
        Master bias to subtract (must match frame/ROI dimensions).
    method : ``"median"`` or ``"mean"``
        Combination method.
    chunk_rows : int
        Rows per chunk.
    roi : ROI, optional
        Restrict to a sub-region.
    """
    reduce_fn = np.median if method == "median" else np.mean

    first = load_frame(paths[0], roi=roi)
    ny, nx = first.shape

    # Adjust bias for ROI if needed
    bias_use = bias
    if bias.shape != (ny, nx):
        if roi is not None:
            bias_use = bias[roi[0], roi[1]]
        if bias_use.shape != (ny, nx):
            raise ValueError(
                f"Bias shape {bias_use.shape} != frame shape ({ny}, {nx})"
            )

    master = np.empty((ny, nx), dtype=np.float32)

    for row_sl, chunk in load_cube_chunked(paths, chunk_rows=chunk_rows, roi=roi):
        # Bias-subtract each frame in the chunk
        bias_strip = bias_use[row_sl, :].astype(np.float32)
        chunk_sub = chunk - bias_strip[np.newaxis, :, :]
        master[row_sl, :] = reduce_fn(chunk_sub, axis=0).astype(np.float32)

    return master


def master_flat(
    paths: List[str],
    bias: Frame,
    dark: Optional[Frame] = None,
    normalize: bool = True,
    method: str = "median",
    chunk_rows: int = 50,
    roi: Optional[ROI] = None,
) -> Frame:
    """Construct a normalized master flat.

    Parameters
    ----------
    paths : list of str
        Flat frame file paths (all same exposure time).
    bias : Frame
        Master bias.
    dark : Frame, optional
        Scaled master dark to subtract (same exposure as flats).
    normalize : bool
        If True, divide by the median level so the master flat
        has unit median.
    method : ``"median"`` or ``"mean"``
        Combination method.
    chunk_rows : int
        Rows per chunk.
    roi : ROI, optional
        Restrict to a sub-region.
    """
    reduce_fn = np.median if method == "median" else np.mean

    first = load_frame(paths[0], roi=roi)
    ny, nx = first.shape

    bias_use = bias
    if bias.shape != (ny, nx):
        if roi is not None:
            bias_use = bias[roi[0], roi[1]]
        if bias_use.shape != (ny, nx):
            raise ValueError(
                f"Bias shape {bias_use.shape} != frame shape ({ny}, {nx})"
            )

    dark_use = dark
    if dark is not None and dark.shape != (ny, nx):
        if roi is not None:
            dark_use = dark[roi[0], roi[1]]
        if dark_use.shape != (ny, nx):
            raise ValueError(
                f"Dark shape {dark_use.shape} != frame shape ({ny}, {nx})"
            )

    master = np.empty((ny, nx), dtype=np.float32)

    for row_sl, chunk in load_cube_chunked(paths, chunk_rows=chunk_rows, roi=roi):
        bias_strip = bias_use[row_sl, :].astype(np.float32)
        chunk_sub = chunk - bias_strip[np.newaxis, :, :]
        if dark_use is not None:
            dark_strip = dark_use[row_sl, :].astype(np.float32)
            chunk_sub -= dark_strip[np.newaxis, :, :]
        master[row_sl, :] = reduce_fn(chunk_sub, axis=0).astype(np.float32)

    if normalize:
        med = np.median(master)
        if med > 0:
            master /= med

    return master
