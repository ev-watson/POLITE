"""
poltools.calibration — Standard-star calibration (Mueller/IP chain inversion).

Implements the publication calibration steps (research map §4; Masiero 2007 §3,
DUSTPol §3, Serkowski 1974 — Sources B/A):

1. **Instrumental polarization** (q0, u0) from *unpolarized* standards
   (true p≈0 ⇒ measured q,u = IP). Subtract in q–u space.
2. **Position-angle zero-point** Δθ from *polarized* standards with known θ_lit.
   Rotate q,u by 2Δθ to the equatorial (N→E) frame.
3. **Polarization efficiency** from highly-polarized standards: divide p by eff.

A :class:`PolCalibration` bundles the three and applies them in order.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence, Tuple

import numpy as np


def fit_instrumental_polarization(
    q: Sequence[float], u: Sequence[float],
    weights: Optional[Sequence[float]] = None,
) -> Tuple[float, float, np.ndarray]:
    """IP (q0, u0) as the (weighted) mean q,u of unpolarized standards.

    Returns ``(q0, u0, cov)`` with ``cov`` the 2x2 covariance of the means.
    """
    q = np.asarray(q, dtype=float)
    u = np.asarray(u, dtype=float)
    w = np.ones_like(q) if weights is None else np.asarray(weights, dtype=float)
    w = w / np.sum(w)
    q0 = float(np.sum(w * q))
    u0 = float(np.sum(w * u))
    n = len(q)
    if n > 1:
        cov = np.cov(np.vstack([q, u])) / n
    else:
        cov = np.zeros((2, 2))
    return q0, u0, cov


def apply_ip(q: float, u: float, q0: float, u0: float) -> Tuple[float, float]:
    """Subtract instrumental polarization in q–u space."""
    return q - q0, u - u0


def _wrap_pa(dtheta_deg: float) -> float:
    """Wrap a PA difference into (−90, 90] degrees."""
    return (dtheta_deg + 90.0) % 180.0 - 90.0


def fit_pa_zeropoint(
    theta_meas_deg: Sequence[float], theta_lit_deg: Sequence[float],
    weights: Optional[Sequence[float]] = None,
) -> Tuple[float, float]:
    """PA zero-point Δθ = <θ_lit − θ_meas> (wrapped) from polarized standards.

    Returns ``(dtheta_deg, sigma_deg)``.
    """
    tm = np.asarray(theta_meas_deg, dtype=float)
    tl = np.asarray(theta_lit_deg, dtype=float)
    diffs = np.array([_wrap_pa(a - b) for a, b in zip(tl, tm)])
    w = np.ones_like(diffs) if weights is None else np.asarray(weights, dtype=float)
    w = w / np.sum(w)
    dtheta = float(np.sum(w * diffs))
    n = len(diffs)
    sigma = float(np.std(diffs, ddof=1) / np.sqrt(n)) if n > 1 else 0.0
    return dtheta, sigma


def apply_pa_zeropoint(q: float, u: float, dtheta_deg: float) -> Tuple[float, float]:
    """Rotate (q, u) by 2Δθ to correct the PA zero-point."""
    a = 2.0 * np.deg2rad(dtheta_deg)
    c, s = np.cos(a), np.sin(a)
    return q * c - u * s, q * s + u * c


def fit_efficiency(
    p_meas: Sequence[float], p_lit: Sequence[float],
    weights: Optional[Sequence[float]] = None,
) -> float:
    """Polarization efficiency = <p_meas / p_lit> from polarized standards."""
    pm = np.asarray(p_meas, dtype=float)
    pl = np.asarray(p_lit, dtype=float)
    ratios = pm / pl
    w = np.ones_like(ratios) if weights is None else np.asarray(weights, dtype=float)
    w = w / np.sum(w)
    return float(np.sum(w * ratios))


def apply_efficiency(q: float, u: float, efficiency: float) -> Tuple[float, float]:
    """Divide the polarization vector by the efficiency."""
    if efficiency == 0:
        return q, u
    return q / efficiency, u / efficiency


@dataclass(frozen=True)
class PolCalibration:
    """Bundled calibration: IP subtraction, PA zero-point, efficiency.

    Apply order (research map §4): IP → efficiency → PA zero-point.
    """

    q0: float = 0.0
    u0: float = 0.0
    dtheta_deg: float = 0.0
    efficiency: float = 1.0

    def apply(self, q: float, u: float) -> Tuple[float, float]:
        q, u = apply_ip(q, u, self.q0, self.u0)
        q, u = apply_efficiency(q, u, self.efficiency)
        q, u = apply_pa_zeropoint(q, u, self.dtheta_deg)
        return q, u
