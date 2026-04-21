"""
caltools.stats — Statistical utilities for detector characterization.

Welford online accumulator, robust estimators, gaussianity tests,
and the per-pixel sigma-vs-mean preparation (Alarcon+2023 Fig. 3).
"""

from __future__ import annotations

import warnings
from typing import Dict, Optional, Tuple

import numpy as np
from scipy import stats as sp_stats

from ._types import Frame


class WelfordAccumulator:
    """Online (single-pass) mean and variance accumulator.

    Processes one frame at a time — avoids storing the full cube.
    Uses the numerically stable Welford algorithm.

    Usage
    -----
    >>> acc = WelfordAccumulator(shape=(ny, nx))
    >>> for frame in frames:
    ...     acc.update(frame)
    >>> mean, var = acc.mean, acc.variance
    """

    def __init__(self, shape: Tuple[int, ...]) -> None:
        self.n = 0
        self._mean = np.zeros(shape, dtype=np.float64)
        self._m2 = np.zeros(shape, dtype=np.float64)

    def update(self, frame: np.ndarray) -> None:
        """Incorporate one frame into the running statistics."""
        self.n += 1
        delta = frame.astype(np.float64) - self._mean
        self._mean += delta / self.n
        delta2 = frame.astype(np.float64) - self._mean
        self._m2 += delta * delta2

    @property
    def mean(self) -> np.ndarray:
        """Running mean."""
        return self._mean

    @property
    def variance(self) -> np.ndarray:
        """Population variance (ddof=0)."""
        if self.n < 2:
            return np.zeros_like(self._mean)
        return self._m2 / self.n

    @property
    def std(self) -> np.ndarray:
        """Population standard deviation."""
        return np.sqrt(self.variance)

    @property
    def sample_variance(self) -> np.ndarray:
        """Sample variance (ddof=1)."""
        if self.n < 2:
            return np.zeros_like(self._mean)
        return self._m2 / (self.n - 1)


def mad_sigma(data: np.ndarray) -> float:
    """Median absolute deviation scaled to Gaussian sigma.

    ``sigma_MAD = 1.4826 * median(|x - median(x)|)``
    """
    med = np.median(data)
    return 1.4826 * np.median(np.abs(data - med))


def outlier_mask(
    data: np.ndarray,
    threshold_sigma: float = 5.0,
    use_mad: bool = True,
) -> np.ndarray:
    """Boolean mask of outlier pixels.

    Parameters
    ----------
    data : ndarray
        Input array (any shape).
    threshold_sigma : float
        Number of sigma above the median to flag.
    use_mad : bool
        If True, use MAD-based sigma; otherwise use standard deviation.

    Returns
    -------
    mask : ndarray (bool)
        True where ``data > median + threshold * sigma``.
    """
    med = np.median(data)
    sigma = mad_sigma(data) if use_mad else np.std(data)
    return data > (med + threshold_sigma * sigma)


def gaussianity_tests(
    data: np.ndarray,
    subsample_size: int = 5000,
    seed: int = 42,
) -> Dict[str, object]:
    """Run D'Agostino-Pearson and Anderson-Darling normality tests.

    Parameters
    ----------
    data : ndarray
        1-D array of values (will be flattened and subsampled).
    subsample_size : int
        Number of points to subsample for the tests.
    seed : int
        RNG seed for reproducibility.

    Returns
    -------
    dict with keys:
        dagostino_stat, dagostino_p,
        anderson_stat, anderson_critical, anderson_significance,
        skewness, kurtosis
    """
    flat = data.ravel()
    rng = np.random.default_rng(seed)
    n = min(subsample_size, len(flat))
    sample = rng.choice(flat, size=n, replace=False)

    k2_stat, k2_p = sp_stats.normaltest(sample)
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore", category=FutureWarning,
            message=".*p-value calculation method.*",
        )
        ad = sp_stats.anderson(sample, dist="norm")

    return {
        "dagostino_stat": float(k2_stat),
        "dagostino_p": float(k2_p),
        "anderson_stat": float(ad.statistic),
        "anderson_critical": list(ad.critical_values),
        "anderson_significance": list(ad.significance_level),
        "skewness": float(sp_stats.skew(flat)),
        "kurtosis": float(sp_stats.kurtosis(flat)),
    }


def sigma_vs_mean_2d(
    pixel_mean: Frame,
    pixel_std: Frame,
    n_sub: int = 500_000,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray]:
    """Prepare subsampled (mean, std) arrays for Alarcon+2023 Fig. 3.

    Parameters
    ----------
    pixel_mean : Frame
        Per-pixel temporal mean across a frame stack.
    pixel_std : Frame
        Per-pixel temporal std across a frame stack.
    n_sub : int
        Number of pixels to subsample.
    seed : int
        RNG seed.

    Returns
    -------
    (mean_sub, std_sub) : tuple of 1-D arrays
    """
    pm = pixel_mean.ravel()
    ps = pixel_std.ravel()
    rng = np.random.default_rng(seed)
    n = min(n_sub, len(pm))
    idx = rng.choice(len(pm), size=n, replace=False)
    return pm[idx], ps[idx]
