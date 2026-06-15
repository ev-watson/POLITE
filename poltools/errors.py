"""
poltools.errors — Polarization uncertainties and debiasing estimators.

Functions used when assembling :class:`~poltools._types.StokesResult`:

* **Residual polarization uncertainty** — from the modulation fit
  (Magalhães et al. 1984; Ramírez et al. 2017, SOLVEPOL).
* **Modified Asymptotic debiasing** — corrects low signal-to-noise bias in the
  polarization fraction (Plaszczynski et al. 2014).
* **Position-angle uncertainty** — Gaussian formula at high signal-to-noise;
  full interval from Naghizadeh-Khouei & Clarke (1993) at low signal-to-noise.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
from scipy import integrate
from scipy.special import erfcx

_RAD_HALF_DEG = 0.5 * np.rad2deg(1.0)


def residual_sigma_p(z: np.ndarray, q: float, u: float, n: int) -> float:
    """Residual polarization uncertainty from the modulation fit (SOLVEPOL).

    For equally spaced half-wave plate angles,

    ``σ_P = sqrt( ((2/N) Σ z_i² − q² − u²) / (N − 2) )``,

    where ``z_i`` are the measured beam ratios at each angle.
    """
    z = np.asarray(z, dtype=float)
    if n <= 2:
        return float("nan")
    val = ((2.0 / n) * np.sum(z ** 2) - q ** 2 - u ** 2) / (n - 2)
    return float(np.sqrt(max(val, 0.0)))


def debias_naive(p: float) -> float:
    """Return the measured polarization fraction unchanged (biased at low SNR)."""
    return float(p)


def debias_wardle_kronberg(p: float, sigma_p: float) -> float:
    """Wardle & Kronberg (1974) debiasing: ``sqrt(max(p² − σ², 0))``."""
    return float(np.sqrt(max(p ** 2 - sigma_p ** 2, 0.0)))


def debias_mas(p: float, sigma_p: float, b2: Optional[float] = None) -> float:
    """Modified Asymptotic estimator (Plaszczynski et al. 2014, eq. 20).

    ``p_MAS = p − b²(1 − e^{−p²/b²}) / (2p)`` with ``b² = σ_p²`` by default.
    """
    if b2 is None:
        b2 = sigma_p ** 2
    if p <= 0:
        return 0.0
    return float(p - b2 * (1.0 - np.exp(-(p ** 2) / b2)) / (2.0 * p))


def sigma_theta_highsnr(p: float, sigma_p: float) -> float:
    """Position-angle uncertainty [degrees] at high polarization signal-to-noise.

    Uses the Serkowski relation: ``28.65° × σ_P / P``.
    """
    if p <= 0:
        return float("nan")
    return float(_RAD_HALF_DEG * sigma_p / p)


def _nkc_pdf_unnorm(dtheta_rad: np.ndarray, snr: float) -> np.ndarray:
    """Naghizadeh-Khouei & Clarke (1993) position-angle offset PDF (unnormalized)."""
    eta0 = (snr / np.sqrt(2.0)) * np.cos(2.0 * dtheta_rad)
    inv_sqrt_pi = 1.0 / np.sqrt(np.pi)
    return inv_sqrt_pi + eta0 * erfcx(-eta0)


def sigma_theta_nkc(snr: float, conf: float = 0.6827) -> float:
    """Position-angle confidence half-width [degrees] at low or moderate SNR.

    Integrates the Naghizadeh-Khouei & Clarke (1993) distribution for the
    polarization angle rather than assuming a Gaussian.

    Parameters
    ----------
    snr : float
        Polarization signal-to-noise, ``P / σ_P``.
    conf : float
        Central probability contained in the interval (default 68.27%, one sigma).
    """
    if snr <= 0:
        return 45.0
    if snr > 25.0:
        return float(_RAD_HALF_DEG / snr)
    grid = np.linspace(-np.pi / 2, np.pi / 2, 4001)
    pdf = _nkc_pdf_unnorm(grid, snr)
    norm = integrate.trapezoid(pdf, grid)
    pdf = pdf / norm

    def covered(h):
        mask = np.abs(grid) <= h
        return integrate.trapezoid(pdf[mask], grid[mask])

    target = conf
    lo, hi = 0.0, np.pi / 2
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        if covered(mid) < target:
            lo = mid
        else:
            hi = mid
    return float(np.rad2deg(0.5 * (lo + hi)))


def sigma_r(f_o: float, f_e: float, sig_o: float, sig_e: float) -> float:
    """Uncertainty on the beam ratio ``R = (f_e − f_o)/(f_e + f_o)``."""
    s = f_e + f_o
    if s == 0:
        return float("nan")
    return float(np.sqrt(4.0 / s ** 4 * (f_o ** 2 * sig_e ** 2 + f_e ** 2 * sig_o ** 2)))
