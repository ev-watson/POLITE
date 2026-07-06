"""
caltools.dark — Dark current analysis, temperature dependence, warm pixels.

Dark-current measurements from dark frames, with optional Arrhenius
temperature fitting where the data support it.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy.optimize import curve_fit

from ._types import AnalysisResult, Frame, ROI, SensorConfig
from .stats import outlier_mask
from .stacking import master_dark


def dark_current_vs_exposure(
    dark_groups: Dict[float, List[str]],
    bias: Frame,
    config: SensorConfig,
    roi: Optional[ROI] = None,
) -> AnalysisResult:
    """Measure dark current from linear fit of mean signal vs exposure time.

    Parameters
    ----------
    dark_groups : dict
        ``{exptime_s: [file_paths]}`` for each dark exposure level.
    bias : Frame
        Master bias.
    config : SensorConfig
        Detector configuration.
    roi : ROI, optional
        Region of interest.

    Returns
    -------
    AnalysisResult with dark current in e-/pix/s.
    """
    g = config.gain_e_per_adu
    exptimes = []
    mean_signals = []
    std_signals = []

    bias_roi = bias
    if roi is not None:
        bias_roi = bias[roi[0], roi[1]]

    for exp in sorted(dark_groups.keys()):
        paths = dark_groups[exp]
        md = master_dark(paths, bias, roi=roi)
        mean_s = float(np.mean(md))
        std_s = float(np.std(md))
        exptimes.append(exp)
        mean_signals.append(mean_s)
        std_signals.append(std_s)

    exptimes = np.array(exptimes)
    mean_signals = np.array(mean_signals)

    # Linear fit: signal = dark_rate * t + offset
    coeffs = np.polyfit(exptimes, mean_signals, 1)
    dark_rate_adu = float(coeffs[0])  # ADU/pix/s
    offset_adu = float(coeffs[1])
    dark_rate_e = dark_rate_adu * g

    # R² for fit quality
    predicted = np.polyval(coeffs, exptimes)
    ss_res = np.sum((mean_signals - predicted) ** 2)
    ss_tot = np.sum((mean_signals - np.mean(mean_signals)) ** 2)
    r_squared = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0

    return AnalysisResult(
        name="dark_current_vs_exposure",
        scalar_summary={
            "dark_rate_adu_per_s": dark_rate_adu,
            "dark_rate_e_per_s": dark_rate_e,
            "offset_adu": offset_adu,
            "r_squared": r_squared,
        },
        metadata={
            "exptimes": exptimes,
            "mean_signals_adu": mean_signals,
            "std_signals_adu": np.array(std_signals),
            "fit_coeffs": coeffs,
            "gain_e_per_adu": g,
        },
    )


def _arrhenius(T_celsius: np.ndarray, A: float, Ea: float) -> np.ndarray:
    """Arrhenius dark current model: I = A * exp(-Ea / kT).

    Parameters
    ----------
    T_celsius : array
        Temperature in Celsius.
    A : float
        Pre-exponential factor.
    Ea : float
        Activation energy in eV.
    """
    k_B = 8.617e-5  # eV/K
    T_K = T_celsius + 273.15
    return A * np.exp(-Ea / (k_B * T_K))


def dark_current_vs_temperature(
    dark_groups: Dict[float, Tuple[List[str], float]],
    bias: Frame,
    config: SensorConfig,
    roi: Optional[ROI] = None,
) -> AnalysisResult:
    """Fit dark current vs temperature with an Arrhenius model.

    Parameters
    ----------
    dark_groups : dict
        ``{temperature_C: (file_paths, exptime_s)}`` — dark frames grouped
        by detector temperature.
    bias : Frame
        Master bias.
    config : SensorConfig
        Detector configuration.
    roi : ROI, optional
        Region of interest.

    Returns
    -------
    AnalysisResult. Note: with a narrow temperature range (~6 C) the
    Arrhenius fit will be poorly constrained; covariance-based parameter
    uncertainties are reported when available.
    """
    g = config.gain_e_per_adu
    temps = []
    dark_rates = []

    for temp_c in sorted(dark_groups.keys()):
        paths, exptime = dark_groups[temp_c]
        md = master_dark(paths, bias, roi=roi)
        rate_adu = float(np.mean(md)) / exptime
        temps.append(temp_c)
        dark_rates.append(rate_adu * g)

    temps = np.array(temps)
    dark_rates = np.array(dark_rates)

    scalars: Dict[str, float] = {}
    fit_success = False

    temp_range = float(temps.max() - temps.min()) if temps.size else 0.0
    if len(temps) >= 3 and temp_range >= 1.0:
        try:
            popt, pcov = curve_fit(
                _arrhenius, temps, dark_rates,
                p0=[1e10, 0.6],
                maxfev=10000,
            )
            perr = np.sqrt(np.diag(pcov))
            if np.all(np.isfinite(perr)):
                scalars["arrhenius_A"] = float(popt[0])
                scalars["arrhenius_Ea_eV"] = float(popt[1])
                scalars["arrhenius_A_err"] = float(perr[0])
                scalars["arrhenius_Ea_err_eV"] = float(perr[1])
                fit_success = True
        except (RuntimeError, ValueError):
            pass

    if not fit_success:
        scalars["arrhenius_fit_failed"] = 1.0

    scalars["temp_range_c"] = temp_range

    return AnalysisResult(
        name="dark_current_vs_temperature",
        scalar_summary=scalars,
        metadata={
            "temperatures_c": temps,
            "dark_rates_e_per_s": dark_rates,
            "fit_success": fit_success,
        },
    )


def dark_spatial_structure(
    master_darks: Dict[float, Frame],
    config: SensorConfig,
) -> AnalysisResult:
    """Summarize 2-D dark-signal structure at each exposure time.

    Parameters
    ----------
    master_darks : dict
        ``{exptime_s: master_dark_frame}`` (bias-subtracted).
    config : SensorConfig
        Detector configuration.
    """
    g = config.gain_e_per_adu
    scalars = {}
    maps = {}

    for exp, md in sorted(master_darks.items()):
        maps[f"dark_structure_{exp}s"] = md.astype(np.float32)
        scalars[f"mean_{exp}s_adu"] = float(np.mean(md))
        scalars[f"mean_{exp}s_e"] = float(np.mean(md)) * g
        scalars[f"std_{exp}s_adu"] = float(np.std(md))
        scalars[f"max_{exp}s_adu"] = float(np.max(md))

    return AnalysisResult(
        name="dark_spatial_structure",
        scalar_summary=scalars,
        maps=maps,
        metadata={"gain_e_per_adu": g},
    )


def warm_pixel_map(
    dark_cube: Dict[float, List[str]],
    bias: Frame,
    config: SensorConfig,
    threshold_sigma: float = 5.0,
    roi: Optional[ROI] = None,
) -> AnalysisResult:
    """Identify warm/hot pixels in dark frames.

    Warm pixels are those with dark current > threshold_sigma * MAD
    above the median dark level.

    Parameters
    ----------
    dark_cube : dict
        ``{exptime_s: [file_paths]}`` for dark frames.
    bias : Frame
        Master bias.
    config : SensorConfig
        Detector configuration.
    threshold_sigma : float
        Detection threshold in MAD-sigma units.
    roi : ROI, optional
        Region of interest.
    """
    g = config.gain_e_per_adu
    scalars = {}
    maps = {}

    for exp in sorted(dark_cube.keys()):
        paths = dark_cube[exp]
        md = master_dark(paths, bias, roi=roi)

        # Warm pixel mask
        warm = outlier_mask(md, threshold_sigma=threshold_sigma, use_mad=True)
        n_warm = int(np.sum(warm))
        frac = n_warm / md.size

        maps[f"warm_mask_{exp}s"] = warm
        maps[f"master_dark_{exp}s"] = md
        scalars[f"n_warm_{exp}s"] = float(n_warm)
        scalars[f"frac_warm_{exp}s"] = frac
        scalars[f"dark_median_{exp}s_adu"] = float(np.median(md))

    return AnalysisResult(
        name="warm_pixel_map",
        scalar_summary=scalars,
        maps=maps,
        metadata={
            "threshold_sigma": threshold_sigma,
            "gain_e_per_adu": g,
        },
    )
