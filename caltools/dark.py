"""
caltools.dark — Dark current, temperature dependence, and warm pixels.

Dark rate from exposure ramps; optional Arrhenius fit when the temperature
range is sufficient (typically poorly constrained on narrow CMOS sweeps).
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy.optimize import curve_fit    # type: ignore

from ._types import AnalysisResult, Frame, ROI
from .stats import outlier_mask
from .stacking import master_bias, master_dark
from .io import load_frame


def dark_current_vs_exposure(
    dark_groups: Dict[float, List[str]],
    bias: Frame,
    gain: float,
    roi: Optional[ROI] = None,
) -> AnalysisResult:
    """Measure dark current from linear fit of mean signal vs exposure time.

    Parameters
    ----------
    dark_groups : dict
        ``{exptime_s: [file_paths]}`` for each dark exposure level.
    bias : Frame
        Master bias.
    gain : float
        Sensor gain in e-/ADU.
    roi : ROI, optional
        Central ROI.

    Returns
    -------
    AnalysisResult with dark current in e-/pix/s.
    """
    g = gain
    exptimes = []
    mom_signals = []
    sdom_signals = []

    for exp in sorted(dark_groups.keys()):
        paths = dark_groups[exp]
        frame_means = [
            float(np.mean(load_frame(path, roi=roi) - bias))
            for path in paths
            ]
        mom_s = float(np.mean(frame_means))
        sdom_s = float(np.std(frame_means, ddof=1) / np.sqrt(len(frame_means)))
        exptimes.append(exp)
        mom_signals.append(mom_s)
        sdom_signals.append(sdom_s)

    exptimes = np.array(exptimes)
    mom_signals = np.array(mom_signals)
    sdom_signals = np.array(sdom_signals)

    coeffs, cov = np.polyfit(exptimes, mom_signals, 1, w=1./sdom_signals, cov=True)
    dark_rate_adu = float(coeffs[0])
    dark_rate_adu_err = np.sqrt(cov[0, 0]) if cov is not None else np.nan
    offset_adu = float(coeffs[1])
    offset_e = offset_adu * g
    dark_rate_e = dark_rate_adu * g
    dark_rate_e_err = np.sqrt(cov[0, 0]) * g if cov is not None else np.nan

    predicted = np.polyval(coeffs, exptimes)
    ss_res = np.sum((mom_signals - predicted) ** 2)
    ss_tot = np.sum((mom_signals - np.mean(mom_signals)) ** 2)
    r_squared = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0

    return AnalysisResult(
        name="dark_current_vs_exposure",
        scalar_summary={
            "dark_rate_adu_per_s": dark_rate_adu,
            "dark_rate_adu_err": dark_rate_adu_err,
            "dark_rate_e_per_s": dark_rate_e,
            "dark_rate_e_err": dark_rate_e_err,
            "offset_adu": offset_adu,
            "r_squared": r_squared,
        },
        metadata={
            "exptimes": exptimes,
            "mean_signals_adu": mom_signals,
            "mean_signals_e": mom_signals * g,
            "std_signals_adu": sdom_signals,
            "std_signals_e": sdom_signals * g,
            "dark_rate_adu_per_s": dark_rate_adu,
            "dark_rate_adu_err": dark_rate_adu_err,
            "dark_rate_e_per_s": dark_rate_e,
            "dark_rate_e_err": dark_rate_e_err,
            "offset_adu": offset_adu,
            "offset_e": offset_e,
            "r_squared": r_squared,
            "fit_coeffs": coeffs,
            "fit_cov": cov,
            "gain_e_per_adu": g,
        },
    )


def arrhenius(T_celsius: np.ndarray, A: float, Ea: float) -> np.ndarray:
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
    dark_groups: Dict[float, Dict[Tuple[str, float], List[str]]],
    gain: float,
    roi: Optional[ROI] = None,
    arrhenius_fit: bool = False,
    current_vs_exposure: bool = True,
) -> AnalysisResult:
    """Summarize fitted dark-current rates at each temperature.

    Parameters
    ----------
    dark_groups : dict
        ``{temperature_C: {(TYPE, exptime): file_paths}}`` — dark frames grouped by type and exposure
        should include bias files per temperature and multiple exposures per temperature.
        by sensor temperature.
    gain : float
        Sensor gain in e-/ADU.
    roi : ROI, optional
        Central ROI.
    arrhenius_fit : bool
        Optionally fit an Arrhenius relation to the fitted rate at each
        temperature. With only three temperatures this is a diagnostic, not a
        precise activation-energy measurement.
    current_vs_exposure : bool
        If True, also return dark current vs exposure AnalysisResult per temperature.

    Returns
    -------
    AnalysisResult or Tuple[AnalysisResult, List[AnalysisResult]] if current_vs_exposure

    Notes
    -----
    This function uses one dark-current value per temperature: the slope from
    :func:`dark_current_vs_exposure` rather than the average of ``master_dark / exposure`` values, 
    because tiny offsets from large exposures dominate short exposure offsets (this could be prevented
    with weighted averaging)
    """
    temperatures_c = []
    dark_rates_e_per_s = []
    dark_rate_errors_e_per_s = []
    current_vs_exposure_results = []

    for temp_c in sorted(dark_groups):
        groups = dark_groups[temp_c]
        bias_paths = []
        per_temp_dark_files = {}
        for (itype, exptime), paths in groups.items():
            if itype == "BIAS":
                bias_paths.extend(paths)
            elif itype == "DARK":
                per_temp_dark_files[float(exptime)] = paths

        if not bias_paths:
            raise ValueError(f"No bias frames supplied for {temp_c:g} C")
        if len(per_temp_dark_files) < 2:
            raise ValueError(
                f"At least two dark exposure times are required for {temp_c:g} C"
            )

        bias = master_bias(bias_paths, roi=roi)
        exposure_result = dark_current_vs_exposure(
            per_temp_dark_files, bias, gain=gain, roi=roi
        )
        current_vs_exposure_results.append(exposure_result)

        fit = exposure_result.scalar_summary
        temperatures_c.append(float(temp_c))
        dark_rates_e_per_s.append(float(fit["dark_rate_e_per_s"]))
        dark_rate_errors_e_per_s.append(float(fit["dark_rate_e_err"]))

    temperatures_c = np.asarray(temperatures_c, dtype=float)
    dark_rates_e_per_s = np.asarray(dark_rates_e_per_s, dtype=float)
    dark_rate_errors_e_per_s = np.asarray(dark_rate_errors_e_per_s, dtype=float)
    temp_range = float(np.ptp(temperatures_c))

    scalars = {
        "n_temperatures": float(len(temperatures_c)),
        "temp_range_c": temp_range,
    }

    if arrhenius_fit:
        fit_success = False

        valid = (   # simple checks to prevent failure
            np.isfinite(dark_rates_e_per_s)
            & (dark_rates_e_per_s > 0)
            & np.isfinite(dark_rate_errors_e_per_s)
            & (dark_rate_errors_e_per_s > 0)
        )
        if np.count_nonzero(valid) >= 3 and temp_range >= 1.0:
            try:
                popt, pcov = curve_fit(
                    arrhenius,
                    temperatures_c[valid],
                    dark_rates_e_per_s[valid],
                    sigma=dark_rate_errors_e_per_s[valid],
                    absolute_sigma=True,
                    bounds=([0.0, 0.0], [np.inf, np.inf]),
                    p0=[1e10, 0.6],
                    maxfev=10000,
                )
                perr = np.sqrt(np.diag(pcov))
                if np.all(np.isfinite(perr)):
                    scalars["arrhenius_A"] = float(popt[0])
                    scalars["arrhenius_Ea_eV"] = float(popt[1])
                    scalars["arrhenius_A_err"] = float(perr[0])
                    scalars["arrhenius_Ea_err_eV"] = float(perr[1])
                    scalars["arrhenius_cov"] = pcov
                    fit_success = True
            except (RuntimeError, ValueError):
                pass

        if not fit_success:
            scalars["arrhenius_fit_failed"] = 1.0

    current_vs_temp_results = AnalysisResult(
            name="dark_current_vs_temperature",
            scalar_summary=scalars,
            metadata={
                "temperatures_c": temperatures_c,
                "dark_rates_e_per_s": dark_rates_e_per_s,
                "dark_rate_errors_e_per_s": dark_rate_errors_e_per_s,
                "per_temperature_results": current_vs_exposure_results,
                "temp_range": temp_range,
                **(
                    {"fit_success": fit_success} if arrhenius_fit else {}
                    ), # unpacks if arrhenius_fit, otherwise empty
            },
        )

    if current_vs_exposure:
        return current_vs_temp_results, current_vs_exposure_results

    return current_vs_temp_results


def dark_spatial_structure(
    master_darks: Dict[float, Frame],
    gain: float,
) -> AnalysisResult:
    """2-D glow maps at each exposure time.

    Parameters
    ----------
    master_darks : dict
        ``{exptime_s: master_dark_frame}`` (bias-subtracted).
    gain : float
        Sensor gain in e-/ADU.
    """
    g = gain
    scalars = {}
    maps = {}

    for exp, md in sorted(master_darks.items()):
        maps[f"glow_{exp}s"] = md.astype(np.float32)
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
    gain: float,
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
    gain : float
        Sensor gain in e-/ADU.
    threshold_sigma : float
        Detection threshold in MAD-sigma units.
    roi : ROI, optional
        Central ROI.
    """
    g = gain
    scalars = {}
    maps = {}

    for exp in sorted(dark_cube.keys()):
        paths = dark_cube[exp]
        md = master_dark(paths, bias, roi=roi)

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
