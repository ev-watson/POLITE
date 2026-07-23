"""
caltools.gain — Photon transfer curve, conversion gain, and full-well capacity.

The photon transfer curve plots variance versus mean signal in flat-field
frames. Its slope gives conversion gain (electrons per ADU). Both a free-
intercept fit and a fit with read-noise variance held fixed are reported.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np

from ._types import AnalysisResult, Frame, ROI, SensorConfig
from .io import load_frame


def _linear_least_squares(
    x: np.ndarray,
    y: np.ndarray,
) -> Tuple[float, float, np.ndarray]:
    """Weighted OLS: y = m*x + c.  Returns (slope, intercept, 2x2 Cov)."""
    x = np.asarray(x, dtype=np.float32)
    y = np.asarray(y, dtype=np.float32)
    B = np.column_stack((x, np.ones_like(x)))
    BTB_inv = np.linalg.inv(B.T @ B)
    params = BTB_inv @ (B.T @ y)
    return float(params[0]), float(params[1]), BTB_inv


def _ptc_pairs_from_group(
    frames: List[Frame],
    exp_t: float,
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """Use adjacent pairs from pre-loaded, bias-subtracted flat frames.

    Returns
    -------
    (signal, variance, labels, mismatch) — arrays of mean signal (ADU),
    Var(diff)/2 (ADU²), group label per pair, and mismatch percentage per pair.
    """
    n = len(frames)
    n_pairs = n // 2
    signals = np.empty(n_pairs, dtype=np.float32)
    variances = np.empty(n_pairs, dtype=np.float32)
    mismatch = np.empty(n_pairs, dtype=np.float32)

    idx = 0
    pair_row = []
    for i in range(0, n, 2):
        f1m, f2m = np.mean(frames[i]), np.mean(frames[i + 1])
        scale = f1m/f2m
        mean_s = 0.5 * (f1m + f2m)
        var_s = np.var(frames[i] - scale * frames[i + 1], ddof=1) / (1 + scale**2)
        signals[idx] = mean_s
        variances[idx] = var_s
        mismatch[idx] = 100.0 * abs(f1m - f2m) / mean_s
        pair_row.append(
            {
                "exptime_s": exp_t,
                "pair": n//2 + 1,
                "signal_adu": mean_s,
                "variance_adu2": var_s,
                "mismatch": mismatch[idx],
            }
        )
        idx += 1

    return signals, variances, [exp_t] * n_pairs, mismatch, pair_row


def photon_transfer_curve(
    flat_groups: Dict[float, List[str]],
    bias: Frame,
    roi: Optional[ROI] = None,
) -> AnalysisResult:
    """Measure conversion gain via the Photon Transfer Curve.

    

    Parameters
    ----------
    flat_groups : dict
        ``{exposure_time: [file_paths]}`` — each group is a set of flat frames
        at one exposure level.
    bias : Frame
        Master bias (full frame or matching ROI).
    roi : ROI, optional
        Central ROI to use (avoids edges/vignetting).

    Returns
    -------
    AnalysisResult with gain, RON, fit parameters in scalar_summary,
    and PTC data arrays in metadata (for plotting).
    """
    bias_roi = bias
    if roi is not None:
        bias_roi = bias[roi[0], roi[1]]

    n_pairs_total = sum(len(paths) // 2 for paths in flat_groups.values())
    signal = np.empty(n_pairs_total, dtype=np.float32)
    variance = np.empty(n_pairs_total, dtype=np.float32)
    labels = []
    mismatch = np.empty(n_pairs_total, dtype=np.float32)

    idx = 0
    pair_rows = []
    for exp_t, paths in flat_groups.items():
        frames = []
        for p in paths:
            raw = load_frame(p, roi=roi)
            frames.append(raw - bias_roi)

        s, v, lb, mm, pr = _ptc_pairs_from_group(frames, exp_t)
        signal[idx:idx+len(s)] = s
        variance[idx:idx+len(v)] = v
        labels.extend(lb)
        mismatch[idx:idx+len(mm)] = mm
        pair_rows.extend(pr)
        idx += len(s)

    m, c, cov = _linear_least_squares(signal, variance)
    gain = 1.0 / m if m != 0 else np.inf
    sigma_gain = float(np.sqrt(cov[0, 0])) / (m ** 2) if m != 0 else np.inf
    ron_from_fit = float(np.sqrt(max(c, 0.0)))

    return AnalysisResult(
        name="photon_transfer_curve",
        scalar_summary={
            "gain_e_per_adu": gain,
            "gain_uncertainty": sigma_gain,
            "ron_fit_adu": ron_from_fit,
            "slope": m,
            "intercept": c,
            "n_pairs": len(signal),
            "signal_min_adu": float(signal.min()),
            "signal_max_adu": float(signal.max()),
        },
        metadata={
            "signal": signal,
            "variance": variance,
            "labels": labels,
            "mismatch": mismatch,
            "fit": {
                "slope": m,
                "intercept": c,
                "cov": cov,
                "gain": gain,
                "gain_sigma": sigma_gain,
            },
            "pair_rows": pair_rows
        },
    )


def photon_transfer_curve_with_ron(
    flat_groups: Dict[str, List[str]],
    bias: Frame,
    config: SensorConfig,
    ron_var_adu2: float,
    roi: Optional[ROI] = None,
) -> AnalysisResult:
    """PTC with an external RON² prior.

    Same as ``photon_transfer_curve`` but the fixed-intercept fit
    uses the supplied ``ron_var_adu2`` instead of estimating from the
    free-fit intercept.

    Parameters
    ----------
    ron_var_adu2 : float
        Read noise variance in ADU², usually from bias frame analysis.
    """
    bias_roi = bias
    if roi is not None:
        bias_roi = bias[roi[0], roi[1]]

    all_signals = []
    all_variances = []
    all_labels = []

    for label, paths in flat_groups.items():
        frames = []
        for p in paths:
            raw = load_frame(p, roi=roi)
            frames.append(raw - bias_roi)
        s, v, lb = _ptc_pairs_from_group(frames, label)
        all_signals.extend(s)
        all_variances.extend(v)
        all_labels.extend(lb)

    signal = np.array(all_signals)
    variance = np.array(all_variances)

    m_free, c_free, cov_free = _linear_least_squares(signal, variance)
    gain_free = 1.0 / m_free if m_free != 0 else np.inf
    sigma_gain_free = float(np.sqrt(cov_free[0, 0])) / (m_free ** 2) if m_free != 0 else np.inf
    ron_from_fit = float(np.sqrt(max(c_free, 0.0)))

    var_shot = variance - ron_var_adu2
    denom = float(np.dot(signal, signal))
    m_fixed = float(np.dot(signal, var_shot)) / denom if denom > 0 else m_free
    gain_fixed = 1.0 / m_fixed if m_fixed != 0 else np.inf

    return AnalysisResult(
        name="photon_transfer_curve",
        scalar_summary={
            "gain_free_e_per_adu": gain_free,
            "gain_free_uncertainty": sigma_gain_free,
            "gain_fixed_e_per_adu": gain_fixed,
            "ron_from_intercept_adu": ron_from_fit,
            "slope_free": m_free,
            "intercept_free": c_free,
            "slope_fixed": m_fixed,
            "ron_var_prior_adu2": ron_var_adu2,
            "n_pairs": len(signal),
            "signal_min_adu": float(signal.min()),
            "signal_max_adu": float(signal.max()),
        },
        metadata={
            "signal": signal,
            "variance": variance,
            "labels": all_labels,
            "fit_free": {
                "slope": m_free,
                "intercept": c_free,
                "gain": gain_free,
                "gain_sigma": sigma_gain_free,
            },
            "fit_fixed": {
                "slope": m_fixed,
                "intercept": ron_var_adu2,
                "gain": gain_fixed,
            },
            "covariance_free": cov_free,
        },
    )


def full_well_capacity(
    ptc_result: AnalysisResult,
    config: SensorConfig,
) -> AnalysisResult:
    """Estimate Full Well Capacity from PTC data.

    FWC is the signal level where the PTC turns over (variance starts
    decreasing). If data does not reach saturation, reports the maximum
    observed signal as a lower bound.

    Parameters
    ----------
    ptc_result : AnalysisResult
        Output of ``photon_transfer_curve()``.
    config : SensorConfig
        Sensor configuration.
    """
    signal = ptc_result.metadata["signal"]
    variance = ptc_result.metadata["variance"]
    gain = config.gain_e_per_adu

    max_signal_adu = float(signal.max())
    max_signal_e = max_signal_adu * gain

    order = np.argsort(signal)
    sig_sorted = signal[order]
    var_sorted = variance[order]

    window = max(5, len(sig_sorted) // 10)
    turnover_adu = None

    for i in range(window, len(sig_sorted)):
        chunk_s = sig_sorted[i - window:i]
        chunk_v = var_sorted[i - window:i]
        if len(chunk_s) > 1:
            slope = np.polyfit(chunk_s, chunk_v, 1)[0]
            if slope < 0:
                turnover_adu = float(sig_sorted[i - window // 2])
                break

    if turnover_adu is not None:
        fwc_adu = turnover_adu
        fwc_e = turnover_adu * gain
        is_lower_bound = False
    else:
        fwc_adu = max_signal_adu
        fwc_e = max_signal_e
        is_lower_bound = True

    return AnalysisResult(
        name="full_well_capacity",
        scalar_summary={
            "fwc_adu": fwc_adu,
            "fwc_e": fwc_e,
            "max_signal_adu": max_signal_adu,
            "max_signal_e": max_signal_e,
            "is_lower_bound": float(is_lower_bound),
        },
        metadata={
            "turnover_adu": turnover_adu,
            "gain_e_per_adu": gain,
        },
    )


def noise_decomposition(
    ptc_signal: np.ndarray,
    ptc_var: np.ndarray,
    config: SensorConfig,
) -> AnalysisResult:
    """Decompose total noise into read noise + shot noise components.

    At each signal level, the total variance is:
        Var_total = RON² + S/G

    This function computes the expected shot noise and read noise
    contributions and returns arrays for visualization.

    Parameters
    ----------
    ptc_signal : 1-D array
        Mean bias-subtracted signal (ADU).
    ptc_var : 1-D array
        Measured Var(diff)/2 (ADU²).
    config : SensorConfig
        Sensor configuration.

    Returns
    -------
    AnalysisResult with decomposition arrays in maps and crossover
    signal in scalar_summary.
    """
    g = config.gain_e_per_adu

    order = np.argsort(ptc_signal)
    sig = ptc_signal[order]
    var_total = ptc_var[order]

    var_shot = sig / g
    low_mask = sig < np.percentile(sig, 20)
    if np.sum(low_mask) > 5:
        ron2 = float(np.median(var_total[low_mask]))
    else:
        ron2 = float(var_total[0])

    var_read = np.full_like(sig, ron2)

    crossover_adu = ron2 * g
    crossover_e = crossover_adu * g

    total_noise_e = np.sqrt(var_total) * g
    shot_noise_e = np.sqrt(var_shot) * g
    read_noise_e = np.sqrt(var_read) * g

    return AnalysisResult(
        name="noise_decomposition",
        scalar_summary={
            "ron_adu": float(np.sqrt(ron2)),
            "ron_e": float(np.sqrt(ron2)) * g,
            "crossover_signal_adu": crossover_adu,
            "crossover_signal_e": crossover_e,
        },
        maps={
            "signal_adu": sig,
            "total_noise_e": total_noise_e,
            "shot_noise_e": shot_noise_e,
            "read_noise_e": read_noise_e,
        },
    )


def momsdom(flat_groups: Dict[float, List[str]],
    bias: Frame,
    roi: Optional[ROI] = None,
) -> AnalysisResult:
    """Report mean of means (MOM) and standard deviation of the means (SDOM) for each exposure group.
    
        
    
        Parameters
        ----------
        flat_groups : dict
            ``{exposure_time: [file_paths]}`` — each group is a set of flat frames
            at one exposure level.
        bias : Frame
            Master bias (full frame or matching ROI).
        roi : ROI, optional
            Central ROI to use (avoids edges/vignetting).
    
        Returns
        -------
        AnalysisResult with gain, RON, fit parameters in scalar_summary,
        and PTC data arrays in metadata (for plotting).
        """
    mom_rows = []

    for exptime, paths in sorted(flat_groups.items()):
        frame_means = []

        for path in paths:
            flat = load_frame(path, roi) - bias
            frame_mean = float(np.mean(flat))
            frame_means.append(frame_mean)

        mom_adu = np.mean(frame_means)
        sdom_adu = np.std(frame_means, ddof=1) / np.sqrt(len(frame_means))
        mom_rows.append(
            {
                "exptime_s": exptime,
                "n_frames": len(frame_means),
                "mom_adu": mom_adu,
                "sdom_adu": sdom_adu,
                "mom_rate_adu_s": mom_adu / exptime,
                "sdom_rate_adu_s": sdom_adu / exptime,
            }
        )

    exposures = np.array([row["exptime_s"] for row in mom_rows])
    nframes = np.array([row["n_frames"] for row in mom_rows])
    moms = np.array([row["mom_adu"] for row in mom_rows])
    sdoms = np.array([row["sdom_adu"] for row in mom_rows])
    mom_rates = np.array([row["mom_rate_adu_s"] for row in mom_rows])
    sdom_rates = np.array([row["sdom_rate_adu_s"] for row in mom_rows])

    return AnalysisResult(
        name="momsdom",
        scalar_summary={
            "n_groups": len(mom_rows),
            "MMOMs": np.mean(moms),
            "SDOSDOMs_adu": np.std(sdoms, ddof=1) / np.sqrt(len(sdoms)),
        },
        metadata={
            "exposures_s": exposures,
            "n_frames": nframes,
            "moms_adu": moms,
            "sdoms_adu": sdoms,
            "mom_rates_adu_s": mom_rates,
            "sdom_rates_adu_s": sdom_rates,
            "mom_rows": mom_rows
        },
    )