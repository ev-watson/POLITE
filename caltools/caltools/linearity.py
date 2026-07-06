"""
caltools.linearity — Detector linearity tests and residual error metrics.

Measures signal vs exposure linearity from a flat-field ramp,
computes residuals, and reports fractional linearity error.
"""

from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np

from ._types import AnalysisResult, Frame, ROI, SensorConfig
from .io import load_frame


def linearity_test(
    flat_groups: Dict[float, List[str]],
    bias: Frame,
    config: SensorConfig,
    roi: Optional[ROI] = None,
) -> AnalysisResult:
    """Measure signal vs exposure linearity from a flat-field ramp.

    Parameters
    ----------
    flat_groups : dict
        ``{exptime_s: [file_paths]}`` — flat frames at each exposure.
    bias : Frame
        Master bias.
    config : SensorConfig
        Detector configuration.
    roi : ROI, optional
        Region of interest for statistics.

    Returns
    -------
    AnalysisResult with linear fit parameters, R², and per-point residuals.
    """
    g = config.gain_e_per_adu

    bias_roi = bias
    if roi is not None:
        bias_roi = bias[roi[0], roi[1]]

    exptimes = []
    mean_signals = []
    std_signals = []

    for exp in sorted(flat_groups.keys()):
        paths = flat_groups[exp]
        # Compute mean of bias-subtracted frames
        signals = []
        for p in paths:
            frame = load_frame(p, roi=roi) - bias_roi
            signals.append(float(np.mean(frame)))
        exptimes.append(exp)
        mean_signals.append(float(np.mean(signals)))
        std_signals.append(float(np.std(signals)))

    exptimes = np.array(exptimes)
    mean_signals = np.array(mean_signals)
    std_signals = np.array(std_signals)

    # Linear fit: S = slope * t + intercept
    coeffs = np.polyfit(exptimes, mean_signals, 1)
    slope = float(coeffs[0])
    intercept = float(coeffs[1])

    # Predicted and residuals
    predicted = np.polyval(coeffs, exptimes)
    residuals = mean_signals - predicted

    # R²
    ss_res = np.sum(residuals ** 2)
    ss_tot = np.sum((mean_signals - np.mean(mean_signals)) ** 2)
    r_squared = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0

    return AnalysisResult(
        name="linearity_test",
        scalar_summary={
            "slope_adu_per_s": slope,
            "slope_e_per_s": slope * g,
            "intercept_adu": intercept,
            "r_squared": r_squared,
            "rms_residual_adu": float(np.sqrt(np.mean(residuals ** 2))),
            "max_residual_adu": float(np.max(np.abs(residuals))),
        },
        metadata={
            "exptimes": exptimes,
            "mean_signals_adu": mean_signals,
            "std_signals_adu": std_signals,
            "predicted_adu": predicted,
            "residuals_adu": residuals,
            "fit_coeffs": coeffs,
            "gain_e_per_adu": g,
        },
    )


def linearity_error(
    linearity_result: AnalysisResult,
) -> AnalysisResult:
    """Compute fractional linearity error from linearity test results.

    LE = residual / fitted_value * 100 (%)

    Parameters
    ----------
    linearity_result : AnalysisResult
        Output of ``linearity_test()``.

    Returns
    -------
    AnalysisResult with per-point LE, max |LE|, and linear range.
    """
    meta = linearity_result.metadata
    exptimes = meta["exptimes"]
    residuals = meta["residuals_adu"]
    predicted = meta["predicted_adu"]

    # Linearity error: residual / fit * 100%
    # Avoid division by zero at zero signal
    safe_pred = np.where(np.abs(predicted) > 1.0, predicted, 1.0)
    le_percent = (residuals / safe_pred) * 100.0

    max_le = float(np.max(np.abs(le_percent)))

    # Linear range: exposure range where |LE| < 1%
    linear_mask = np.abs(le_percent) < 1.0
    if np.any(linear_mask):
        linear_exptimes = exptimes[linear_mask]
        linear_range_s = (float(linear_exptimes.min()), float(linear_exptimes.max()))
    else:
        linear_range_s = (0.0, 0.0)

    return AnalysisResult(
        name="linearity_error",
        scalar_summary={
            "max_le_percent": max_le,
            "linear_range_min_s": linear_range_s[0],
            "linear_range_max_s": linear_range_s[1],
        },
        metadata={
            "exptimes": exptimes,
            "le_percent": le_percent,
            "residuals_adu": residuals,
            "predicted_adu": predicted,
        },
    )
