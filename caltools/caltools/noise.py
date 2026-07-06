"""
caltools.noise — Read noise, banding, DSNU, FPN, and unstable pixels.

Functions for read-noise maps, fixed-pattern structure, row/column
profiles, and temporally unstable pixels.
"""

from __future__ import annotations

from typing import Dict, Tuple

import numpy as np
from scipy.fft import rfft, rfftfreq

from ._types import AnalysisResult, Frame, FrameCube, SensorConfig
from .stats import WelfordAccumulator, mad_sigma


def read_noise_map(
    cube: FrameCube,
    method: str = "frame_diff",
) -> Tuple[Frame, Frame]:
    """Compute per-pixel read noise from a bias cube.

    Parameters
    ----------
    cube : FrameCube
        3-D array ``(n_frames, ny, nx)`` of bias frames.
    method : str
        ``"frame_diff"`` — consecutive-pair differencing (removes FPN).
        ``"temporal_std"`` — pixel-wise std across stack (includes FPN).

    Returns
    -------
    (read_noise_map, temporal_std_map) : tuple of Frame
        Both in the same units as the input (ADU).
        ``read_noise_map`` uses frame differencing (FPN-free).
        ``temporal_std_map`` is direct pixel-wise temporal std.
    """
    n = cube.shape[0]

    if method == "temporal_std":
        ts = np.std(cube, axis=0, dtype=np.float64)
        return ts, ts

    # Frame-differencing with Welford accumulator
    ny, nx = cube.shape[1], cube.shape[2]
    acc = WelfordAccumulator((ny, nx))

    for k in range(n - 1):
        diff = cube[k + 1].astype(np.float64) - cube[k].astype(np.float64)
        acc.update(diff)

    rms_diff = np.sqrt(acc.variance)
    rn_map = rms_diff / np.sqrt(2.0)

    # Also compute direct temporal std (includes FPN)
    temporal_std = np.std(cube, axis=0, dtype=np.float64)

    return rn_map.astype(np.float32), temporal_std.astype(np.float32)


def read_noise_spatial(
    rn_map: Frame,
    config: SensorConfig,
) -> AnalysisResult:
    """Summarize read noise map statistics and high-noise pixel counts.

    Parameters
    ----------
    rn_map : Frame
        Per-pixel read noise in ADU.
    config : SensorConfig
        Detector configuration (provides gain for electron conversion).

    Returns
    -------
    AnalysisResult with scalar_summary including:
        ron_median_adu, ron_median_e, ron_rms_adu, ron_rms_e,
        hot_3sig, hot_5sig, hot_10sig (high-noise pixel counts).
    """
    g = config.gain_e_per_adu
    med = float(np.median(rn_map))
    rms = float(np.sqrt(np.mean(rn_map ** 2)))
    sig = mad_sigma(rn_map)

    hot = {}
    for nsig in [3, 5, 10]:
        mask = rn_map > (med + nsig * sig)
        hot[f"hot_{nsig}sig"] = int(np.sum(mask))

    return AnalysisResult(
        name="read_noise_spatial",
        scalar_summary={
            "ron_median_adu": med,
            "ron_median_e": med * g,
            "ron_rms_adu": rms,
            "ron_rms_e": rms * g,
            "mad_sigma_adu": sig,
            **hot,
        },
        maps={
            "hot_5sig_mask": rn_map > (med + 5 * sig),
        },
        metadata={
            "gain_e_per_adu": g,
            "n_pixels": int(rn_map.size),
        },
    )


def row_column_noise(
    diff_image: Frame,
) -> AnalysisResult:
    """Characterize column and row banding via 1-D profiles and FFT.

    Parameters
    ----------
    diff_image : Frame
        A single representative bias-difference image.

    Returns
    -------
    AnalysisResult with maps containing 1-D profiles and FFT spectra,
    and scalar_summary with top-5 dominant frequencies per axis.
    """
    col_profile = np.median(diff_image, axis=0)
    row_profile = np.median(diff_image, axis=1)

    # FFT power spectra
    col_fft = np.abs(rfft(col_profile - np.mean(col_profile)))
    col_freq = rfftfreq(len(col_profile))
    row_fft = np.abs(rfft(row_profile - np.mean(row_profile)))
    row_freq = rfftfreq(len(row_profile))

    # Top-5 peaks
    scalars = {}
    for axis_name, spectrum, freq in [
        ("col", col_fft, col_freq),
        ("row", row_fft, row_freq),
    ]:
        power = spectrum[1:] ** 2
        freqs = freq[1:]
        top5 = np.argsort(power)[-5:][::-1]
        for rank, idx in enumerate(top5, 1):
            f = float(freqs[idx])
            period = 1.0 / f if f > 0 else np.inf
            scalars[f"{axis_name}_freq_{rank}"] = f
            scalars[f"{axis_name}_period_{rank}"] = period
            scalars[f"{axis_name}_power_{rank}"] = float(power[idx])

    return AnalysisResult(
        name="row_column_noise",
        scalar_summary=scalars,
        maps={
            "col_profile": col_profile,
            "row_profile": row_profile,
            "col_fft_power": col_fft[1:] ** 2,
            "col_fft_freq": col_freq[1:],
            "row_fft_power": row_fft[1:] ** 2,
            "row_fft_freq": row_freq[1:],
        },
    )


def dsnu(
    master_darks_by_exptime: Dict[float, Frame],
    bias: Frame,
    config: SensorConfig,
) -> AnalysisResult:
    """Compute Dark Signal Non-Uniformity decomposed into pixel/row/column.

    DSNU is computed from bias-subtracted master dark frames across the
    spatial dimensions.
    The returned summary separates total structure into row, column, and
    residual pixel components.

    Parameters
    ----------
    master_darks_by_exptime : dict
        ``{exptime: master_dark_frame}`` (bias-subtracted).
    bias : Frame
        Master bias, retained for API compatibility.
    config : SensorConfig
        Detector configuration.
    """
    g = config.gain_e_per_adu
    scalars = {}
    maps = {}

    for exp, md in sorted(master_darks_by_exptime.items()):
        dark_sub = md.astype(np.float64)

        # Total DSNU
        total_dsnu = float(np.std(dark_sub))
        scalars[f"dsnu_total_{exp}s_adu"] = total_dsnu
        scalars[f"dsnu_total_{exp}s_e"] = total_dsnu * g

        # Row structure
        row_med = np.median(dark_sub, axis=1)
        row_dsnu = float(np.std(row_med))
        scalars[f"dsnu_row_{exp}s_adu"] = row_dsnu

        # Column structure
        col_med = np.median(dark_sub, axis=0)
        col_dsnu = float(np.std(col_med))
        scalars[f"dsnu_col_{exp}s_adu"] = col_dsnu

        # Pixel DSNU (residual)
        struct = row_med[:, np.newaxis] + col_med[np.newaxis, :] - np.mean(dark_sub)
        residual = dark_sub - struct
        pix_dsnu = float(np.std(residual))
        scalars[f"dsnu_pixel_{exp}s_adu"] = pix_dsnu

        maps[f"dark_structure_{exp}s"] = dark_sub.astype(np.float32)

    return AnalysisResult(
        name="dsnu",
        scalar_summary=scalars,
        maps=maps,
        metadata={"gain_e_per_adu": g},
    )


def fpn(
    temporal_std_map: Frame,
    rn_map: Frame,
) -> AnalysisResult:
    """Isolate Fixed Pattern Noise by quadrature subtraction.

    FPN² = temporal_std² - read_noise² (frame-diff)

    The temporal std includes FPN while the frame-diff read noise does not.

    Parameters
    ----------
    temporal_std_map : Frame
        Pixel-wise temporal std across the bias stack (includes FPN).
    rn_map : Frame
        Frame-differencing read noise map (FPN-free).

    Returns
    -------
    AnalysisResult with the FPN map and summary statistics.
    """
    var_temporal = temporal_std_map.astype(np.float64) ** 2
    var_rn = rn_map.astype(np.float64) ** 2
    var_fpn = np.maximum(var_temporal - var_rn, 0.0)
    fpn_map = np.sqrt(var_fpn).astype(np.float32)

    return AnalysisResult(
        name="fpn",
        scalar_summary={
            "fpn_mean_adu": float(np.mean(fpn_map)),
            "fpn_median_adu": float(np.median(fpn_map)),
            "fpn_max_adu": float(np.max(fpn_map)),
            "temporal_std_mean_adu": float(np.mean(temporal_std_map)),
            "ron_mean_adu": float(np.mean(rn_map)),
            "fpn_fraction": float(np.mean(fpn_map) / np.mean(temporal_std_map))
            if np.mean(temporal_std_map) > 0
            else 0.0,
        },
        maps={"fpn_map": fpn_map},
    )


def detect_rtn_pixels(
    cube: FrameCube,
    config: SensorConfig,
    sigma_threshold: float = 3.0,
) -> AnalysisResult:
    """Detect temporally unstable pixels in a bias stack.

    Flags pixels with anomalously high temporal scatter while their mean
    bias level remains close to the frame population.

    Parameters
    ----------
    cube : FrameCube
        Bias cube ``(n_frames, ny, nx)``.
    config : SensorConfig
        Detector configuration.
    sigma_threshold : float
        Number of RMS read-noise units above which a pixel is flagged.

    Returns
    -------
    AnalysisResult with the unstable-pixel mask and pixel statistics.
    """
    pixel_mean = np.mean(cube, axis=0, dtype=np.float64)
    pixel_std = np.std(cube, axis=0, dtype=np.float64)

    # Read noise via frame differencing (for the threshold)
    rn_map, _ = read_noise_map(cube, method="frame_diff")
    rms_ron = float(np.sqrt(np.mean(rn_map ** 2)))

    # Bias level statistics
    med_bias = float(np.median(pixel_mean))
    mad_bias = mad_sigma(pixel_mean)

    # High temporal scatter with an otherwise typical mean bias level.
    sp_mask = (
        (pixel_std > sigma_threshold * rms_ron)
        & (np.abs(pixel_mean - med_bias) < 5 * mad_bias)
    )
    # High-bias pixels: anomalous mean signal
    high_mask = pixel_mean > (med_bias + 5 * mad_bias)

    n_sp = int(np.sum(sp_mask))
    n_high = int(np.sum(high_mask))
    n_total = int(pixel_mean.size)

    # Find example pixel locations for visualization
    sp_ys, sp_xs = np.where(sp_mask)
    examples = []
    if len(sp_ys) > 0:
        sp_stds = pixel_std[sp_mask]
        sorted_idx = np.argsort(sp_stds)
        for frac in [0.25, 0.5, 0.75, 0.9, 0.99]:
            idx = min(int(frac * len(sorted_idx)), len(sorted_idx) - 1)
            examples.append((int(sp_ys[sorted_idx[idx]]), int(sp_xs[sorted_idx[idx]])))

    return AnalysisResult(
        name="rtn_detection",
        scalar_summary={
            "n_sp": n_sp,
            "frac_sp": n_sp / n_total,
            "n_high_bias": n_high,
            "frac_high_bias": n_high / n_total,
            "rms_ron_adu": rms_ron,
            "sigma_threshold": sigma_threshold,
        },
        maps={
            "sp_mask": sp_mask,
            "high_mask": high_mask,
            "pixel_mean": pixel_mean.astype(np.float32),
            "pixel_std": pixel_std.astype(np.float32),
        },
        metadata={
            "example_pixels": examples,
            "n_total": n_total,
        },
    )
