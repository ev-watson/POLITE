"""
caltools.prnu — Photo-Response Non-Uniformity from flat-field stacks.

PRNU is the pixel-to-pixel sensitivity variation, measured as the
standard deviation of the normalized master flat.
"""

from __future__ import annotations

from typing import List, Optional

import numpy as np

from ._types import AnalysisResult, Frame, ROI, SensorConfig
from .stacking import master_flat


def prnu_map(
    flat_paths: List[str],
    bias: Frame,
    config: SensorConfig,
    dark: Optional[Frame] = None,
    roi: Optional[ROI] = None,
) -> AnalysisResult:
    """Compute PRNU from a flat-field stack.

    PRNU = std(normalized_master_flat) expressed as a percentage.
    The master flat is normalized to unit median.

    Parameters
    ----------
    flat_paths : list of str
        Flat frame file paths (all same exposure, uniform illumination).
    bias : Frame
        Master bias.
    config : SensorConfig
        Detector configuration.
    dark : Frame, optional
        Scaled master dark (same exposure as flats).
    roi : ROI, optional
        Central ROI.

    Returns
    -------
    AnalysisResult with PRNU percentage and deviation map.
    """
    mf = master_flat(flat_paths, bias, dark=dark, normalize=True, roi=roi)

    # PRNU: deviation from unity in the normalized flat
    prnu = mf - 1.0
    prnu_std = float(np.std(prnu))
    prnu_percent = prnu_std * 100.0

    return AnalysisResult(
        name="prnu",
        scalar_summary={
            "prnu_std": prnu_std,
            "prnu_percent": prnu_percent,
            "flat_median": float(np.median(mf)),
            "flat_mean": float(np.mean(mf)),
            "n_frames": len(flat_paths),
        },
        maps={
            "normalized_flat": mf,
            "prnu_map": prnu.astype(np.float32),
        },
        metadata={
            "sensor_name": config.sensor_name,
        },
    )
