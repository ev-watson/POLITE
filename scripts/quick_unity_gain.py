#!/usr/bin/env python3
"""Estimate QHY268M conversion gain from one equal-exposure flat pair.

This is the fast, preliminary Janesick single-point photon-transfer check for
the 2026-07-17 calibration night. It uses two equal-exposure flats and two bias
frames over a central detector region::

    e-/ADU = (mean(flat_a) + mean(flat_b) - mean(bias_a) - mean(bias_b))
             / variance(flat_a - flat_b)

It assumes the flats have the same illumination and are in the linear regime.
The complete PTC ladder remains the definitive conversion-gain measurement.
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

import numpy as np
from astropy.io import fits


def _central_region(data: np.ndarray, fraction: float) -> np.ndarray:
    """Return a centered rectangular region occupying ``fraction`` per axis."""
    if data.ndim != 2:
        raise ValueError(f"Expected a 2-D FITS image, got shape {data.shape}")
    if not 0 < fraction <= 1:
        raise ValueError("central fraction must be in (0, 1]")
    ny, nx = data.shape
    height = max(1, round(ny * fraction))
    width = max(1, round(nx * fraction))
    y0 = (ny - height) // 2
    x0 = (nx - width) // 2
    return data[y0:y0 + height, x0:x0 + width]


def _read_region(path: str | Path, fraction: float) -> np.ndarray:
    data = np.asarray(fits.getdata(path, memmap=False), dtype=np.float64)
    return _central_region(data, fraction)


def estimate_conversion_gain(
    flat_a: np.ndarray,
    flat_b: np.ndarray,
    bias_a: np.ndarray,
    bias_b: np.ndarray,
) -> dict[str, float]:
    """Return the single-point conversion-gain estimate and its input metrics."""
    shapes = {a.shape for a in (flat_a, flat_b, bias_a, bias_b)}
    if len(shapes) != 1:
        raise ValueError(f"All four regions must have the same shape, got {sorted(shapes)}")
    if any(not np.all(np.isfinite(a)) for a in (flat_a, flat_b, bias_a, bias_b)):
        raise ValueError("Input regions contain non-finite pixels")

    mean_a = float(np.mean(flat_a))
    mean_b = float(np.mean(flat_b))
    mean_bias_a = float(np.mean(bias_a))
    mean_bias_b = float(np.mean(bias_b))
    signal_adu = mean_a + mean_b - mean_bias_a - mean_bias_b
    difference_variance_adu2 = float(np.var(flat_a - flat_b, ddof=1))
    if signal_adu <= 0:
        raise ValueError(f"Bias-subtracted flat signal must be positive, got {signal_adu:.3f} ADU")
    if difference_variance_adu2 <= 0:
        raise ValueError("Variance of flat_a - flat_b must be positive")

    return {
        "mean_flat_a_adu": mean_a,
        "mean_flat_b_adu": mean_b,
        "mean_bias_a_adu": mean_bias_a,
        "mean_bias_b_adu": mean_bias_b,
        "flat_pair_change_fraction": abs(mean_a - mean_b) / ((mean_a + mean_b) / 2),
        "signal_sum_adu": signal_adu,
        "difference_variance_adu2": difference_variance_adu2,
        "conversion_gain_e_per_adu": signal_adu / difference_variance_adu2,
    }


def _format_result(metrics: dict[str, float]) -> Iterable[str]:
    yield "Preliminary Janesick single-point PTC (central region)"
    yield f"  bias-subtracted paired signal : {metrics['signal_sum_adu']:.3f} ADU"
    yield f"  variance(flat A - flat B)    : {metrics['difference_variance_adu2']:.3f} ADU^2"
    yield f"  flat-pair mean change         : {metrics['flat_pair_change_fraction'] * 100:.3f}%"
    yield f"  conversion gain               : {metrics['conversion_gain_e_per_adu']:.4f} e-/ADU"
    if metrics["flat_pair_change_fraction"] > 0.02:
        yield "  WARNING: flat pair differs by >2%; use the full PTC reduction before judging unity."
    yield "  This is preliminary: the full PTC ladder is the verification result."


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("flat_a", type=Path)
    parser.add_argument("flat_b", type=Path)
    parser.add_argument("bias_a", type=Path)
    parser.add_argument("bias_b", type=Path)
    parser.add_argument(
        "--central-fraction", type=float, default=0.5,
        help="Fraction of each image dimension to use, centered (default: 0.5)",
    )
    args = parser.parse_args()

    try:
        metrics = estimate_conversion_gain(
            _read_region(args.flat_a, args.central_fraction),
            _read_region(args.flat_b, args.central_fraction),
            _read_region(args.bias_a, args.central_fraction),
            _read_region(args.bias_b, args.central_fraction),
        )
    except (OSError, ValueError) as exc:
        parser.error(str(exc))

    print("\n".join(_format_result(metrics)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
