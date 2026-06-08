"""
poltools.errors — Publication-grade polarization error metrics & debiasing.

All estimators are sourced from the Phase 1 research map (Sources A/B):

* Residual σ_P from the modulation fit — Magalhães, Benedetti & Roland (1984) /
  SOLVEPOL (Ramírez et al. 2017, Source B).
* Polarization-angle error: high-SNR ``28.65·σ_P/P`` (Serkowski; SOLVEPOL,
  Source B) and the low-SNR distribution of Naghizadeh-Khouei & Clarke
  (1993, A&A 274, 968, Source B).
* Polarization bias / debiasing: the Modified Asymptotic (MAS) estimator —
  Plaszczynski et al. (2014, MNRAS 439, 4048, Source B) eq. 20; restated in
  Montier et al. (2015 II, Source B) eq. 19; with the MAS variance (Montier II
  eq. 20). Classical comparators: Wardle & Kronberg (1974) and the naive
  estimator.

No analysis method outside Sources A/B is introduced.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np
from scipy import integrate, optimize
from scipy.special import erfcx

# 0.5 rad expressed in degrees: σ_θ[deg] = (1/2)(σ_P/P)·(180/π) = 28.6479·σ_P/P
_RAD_HALF_DEG = 0.5 * np.rad2deg(1.0)  # = 28.6478897565...


# --------------------------------------------------------------------------- #
# Residual-based polarization uncertainty (LSQ modulation)
# --------------------------------------------------------------------------- #
def residual_sigma_p(z: np.ndarray, q: float, u: float, n: int) -> float:
    """SOLVEPOL/Magalhães (1984) residual polarization error.

    For the canonical equally-spaced (22.5°) HWP set the modulation design is
    orthogonal with ``Σcos²4ψ = Σsin²4ψ = N/2``, and the residual-based σ on the
    fractional Stokes is

        ``σ_P = sqrt( ( (2/N)·Σ z_i² − q² − u² ) / (N − 2) )``.

    This is the form consistent with the LSQ parameter covariance
    (``σ_q = σ_u = sqrt(2/N)·σ_z``, ``σ_z² = Σresid²/(N−2)``): it vanishes for a
    noiseless perfect fit and equals the propagated σ_q. (Validated against
    Monte-Carlo in the test suite; matches SOLVEPOL, Source B.)
    """
    z = np.asarray(z, dtype=float)
    if n <= 2:
        return float("nan")
    val = ((2.0 / n) * np.sum(z ** 2) - q ** 2 - u ** 2) / (n - 2)
    return float(np.sqrt(max(val, 0.0)))


# --------------------------------------------------------------------------- #
# Debiasing estimators for P = sqrt(q² + u²)  (Rice bias at low SNR)
# --------------------------------------------------------------------------- #
def debias_naive(p: float) -> float:
    """Naive (biased) estimator — returned unchanged, for comparison tables."""
    return float(p)


def debias_wardle_kronberg(p: float, sigma_p: float) -> float:
    """Wardle & Kronberg (1974): ``p_db = sqrt(max(p² − σ², 0))``."""
    return float(np.sqrt(max(p ** 2 - sigma_p ** 2, 0.0)))


def debias_mas(p: float, sigma_p: float, b2: float = None) -> float:
    """Modified Asymptotic estimator (Plaszczynski 2014 eq. 20).

    ``p_MAS = p − b²·(1 − e^{−p²/b²}) / (2p)``.

    Parameters
    ----------
    p : float
        Measured polarization fraction.
    sigma_p : float
        Polarization uncertainty.
    b2 : float, optional
        Noise-bias parameter b². Defaults to the canonical equal-variance value
        ``b² = σ_p²`` (research map §3.4; Montier II eq. 15 canonical case).
    """
    if b2 is None:
        b2 = sigma_p ** 2
    if p <= 0:
        return 0.0
    return float(p - b2 * (1.0 - np.exp(-(p ** 2) / b2)) / (2.0 * p))


def sigma_p_mas(sigma_q: float, sigma_u: float, theta_deg: float,
                I0: float = 1.0, psi_deg: float = 0.0) -> float:
    """MAS polarization-fraction variance (Montier II eq. 20).

    ``σ²_{p,MAS} = [σ'_Q² cos²(2ψ−θ) + σ'_U² sin²(2ψ−θ)] / I₀²``.

    With ``ψ`` the analyzer reference and ``θ`` the polarization angle. For the
    common equal-variance, normalized case this reduces to ``σ_p ≈ σ_q ≈ σ_u``.
    """
    th = np.deg2rad(theta_deg)
    ps = np.deg2rad(psi_deg)
    arg = 2 * ps - 2 * th  # 2ψ − 2θ (θ already the half-angle in deg of PA)
    var = (sigma_q ** 2 * np.cos(arg) ** 2 + sigma_u ** 2 * np.sin(arg) ** 2) / (I0 ** 2)
    return float(np.sqrt(max(var, 0.0)))


# --------------------------------------------------------------------------- #
# Polarization-angle uncertainty
# --------------------------------------------------------------------------- #
def sigma_theta_highsnr(p: float, sigma_p: float) -> float:
    """High-SNR position-angle error in degrees: ``28.65·σ_P/P``."""
    if p <= 0:
        return float("nan")
    return float(_RAD_HALF_DEG * sigma_p / p)


def _nkc_pdf_unnorm(dtheta_rad: np.ndarray, snr: float) -> np.ndarray:
    """Unnormalized Naghizadeh-Khouei & Clarke (1993) PA-offset PDF.

    ``G(η) ∝ 1/√π + η₀·erfcx(−η₀)`` with ``η₀ = (P₀/σ/√2)·cos(2·dθ)``.
    Uses ``erfcx`` (scaled complementary error function) for numerical stability
    at all SNR (avoids the e^{η₀²} overflow in the textbook form).
    """
    eta0 = (snr / np.sqrt(2.0)) * np.cos(2.0 * dtheta_rad)
    inv_sqrt_pi = 1.0 / np.sqrt(np.pi)
    return inv_sqrt_pi + eta0 * erfcx(-eta0)


def sigma_theta_nkc(snr: float, conf: float = 0.6827) -> float:
    """Naghizadeh-Khouei & Clarke (1993) PA confidence half-width (deg).

    Returns the symmetric half-width ``Δθ`` of the central ``conf`` credible
    interval of the position-angle offset, valid at low/moderate SNR where the
    Gaussian ``28.65/SNR`` approximation fails. Converges to the high-SNR value
    as SNR grows.

    Parameters
    ----------
    snr : float
        Polarization signal-to-noise ``P₀/σ_P``.
    conf : float
        Central probability content (default 1σ-equivalent, 0.6827).
    """
    if snr <= 0:
        return 45.0  # fully indeterminate over (−90, 90] half-range
    if snr > 25.0:
        # erfcx overflows for η₀² > ~709; at this SNR the NK&C interval is
        # indistinguishable from the Gaussian asymptotic 28.65/SNR.
        return float(_RAD_HALF_DEG / snr)
    # PA offset lives in (−π/2, π/2]; PDF is symmetric about 0.
    grid = np.linspace(-np.pi / 2, np.pi / 2, 4001)
    pdf = _nkc_pdf_unnorm(grid, snr)
    norm = integrate.trapezoid(pdf, grid)
    pdf = pdf / norm

    def covered(h):
        mask = np.abs(grid) <= h
        return integrate.trapezoid(pdf[mask], grid[mask])

    target = conf
    # bisection for half-width h in (0, π/2]
    lo, hi = 0.0, np.pi / 2
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        if covered(mid) < target:
            lo = mid
        else:
            hi = mid
    return float(np.rad2deg(0.5 * (lo + hi)))


# --------------------------------------------------------------------------- #
# Analytic propagation through the double-difference ratio (Method A)
# --------------------------------------------------------------------------- #
def sigma_R(f_o: float, f_e: float, sig_o: float, sig_e: float) -> float:
    """Uncertainty on ``R = (f_e − f_o)/(f_e + f_o)`` by error propagation.

    ``σ_R² = 4/S⁴ · (f_o² σ_e² + f_e² σ_o²)`` with ``S = f_e + f_o``.
    """
    s = f_e + f_o
    if s == 0:
        return float("nan")
    return float(np.sqrt(4.0 / s ** 4 * (f_o ** 2 * sig_e ** 2 + f_e ** 2 * sig_o ** 2)))
