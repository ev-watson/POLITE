"""
poltools.stokes — Assemble normalized Stokes results.

Takes fitted ``q, u`` and their uncertainties, computes polarization fraction
and position angle, applies a debiasing estimator for the reported fraction,
and packages everything in a :class:`~poltools._types.StokesResult`.
"""

from __future__ import annotations

from typing import Dict, Optional

import numpy as np

from ._types import StokesResult
from . import errors

# Use the Gaussian position-angle formula above this polarization SNR;
# below it, use the Naghizadeh-Khouei & Clarke (1993) interval.
_SNR_GAUSSIAN_PA = 6.0


def polarization_fraction_angle(q: float, u: float):
    """Return polarization fraction ``p`` and position angle ``theta_deg`` [0, 180)."""
    p = float(np.hypot(q, u))
    theta = 0.5 * np.rad2deg(np.arctan2(u, q)) % 180.0
    return p, float(theta)


def sigma_p_from_qu(q: float, u: float, sigma_q: float, sigma_u: float,
                    cov_qu: float = 0.0) -> float:
    """Propagate uncertainties in q and u to uncertainty in ``p = sqrt(q² + u²)``."""
    p = np.hypot(q, u)
    if p < 1e-12:
        return float(np.sqrt(0.5 * (sigma_q ** 2 + sigma_u ** 2)))
    var = (q ** 2 * sigma_q ** 2 + u ** 2 * sigma_u ** 2 + 2 * q * u * cov_qu) / p ** 2
    return float(np.sqrt(max(var, 0.0)))


def assemble_stokes(qu: Dict[str, float], I0: float = 1.0,
                    name: str = "source", method: str = "lsq",
                    estimator: str = "mas",
                    provenance: Optional[Dict[str, str]] = None,
                    extra_metadata: Optional[Dict[str, object]] = None) -> StokesResult:
    """Build a :class:`StokesResult` from a modulation fit dictionary.

    Parameters
    ----------
    qu : dict
        Must contain ``q, u, sigma_q, sigma_u``; ``cov_qu`` optional.
    I0 : float
        Total intensity reference stored in the scalar summary.
    estimator : {"mas", "wk", "naive"}
        Which debiased polarization fraction to report as ``p_report``:

        * ``mas`` — Modified Asymptotic (recommended).
        * ``wk`` — Wardle & Kronberg square-root debiasing.
        * ``naive`` — raw measured fraction (biased at low SNR).
    """
    q = float(qu["q"])
    u = float(qu["u"])
    sigma_q = float(qu["sigma_q"])
    sigma_u = float(qu["sigma_u"])
    cov_qu = float(qu.get("cov_qu", 0.0))

    p, theta = polarization_fraction_angle(q, u)
    sigma_p = sigma_p_from_qu(q, u, sigma_q, sigma_u, cov_qu)
    snr = p / sigma_p if sigma_p > 0 else float("inf")

    p_mas = errors.debias_mas(p, sigma_p)
    p_wk = errors.debias_wardle_kronberg(p, sigma_p)

    sig_th_hi = errors.sigma_theta_highsnr(p, sigma_p)
    sig_th_nkc = errors.sigma_theta_nkc(snr) if np.isfinite(snr) else float("nan")
    sigma_theta = sig_th_hi if snr >= _SNR_GAUSSIAN_PA else sig_th_nkc

    if estimator == "wk":
        p_report = p_wk
    elif estimator == "naive":
        p_report = p
    else:
        p_report = p_mas

    scalar = {
        "I": float(I0),
        "q": q, "u": u,
        "sigma_q": sigma_q, "sigma_u": sigma_u,
        "p": p, "p_mas": p_mas, "p_wk": p_wk, "p_report": p_report,
        "theta_deg": theta,
        "sigma_p": sigma_p,
        "sigma_theta_deg": float(sigma_theta),
        "sigma_theta_highsnr_deg": float(sig_th_hi),
        "sigma_theta_nkc_deg": float(sig_th_nkc),
        "snr": float(snr),
    }
    if "sigma_p_resid" in qu:
        scalar["sigma_p_resid"] = float(qu["sigma_p_resid"])
    if "chi2" in qu and np.isfinite(qu.get("chi2", np.nan)):
        scalar["chi2"] = float(qu["chi2"])
        scalar["dof"] = int(qu.get("dof", 0))

    meta: Dict[str, object] = {
        "method": method,
        "estimator_default": estimator,
        "pa_form": "highsnr" if snr >= _SNR_GAUSSIAN_PA else "nkc1993",
        "provenance": provenance or {},
    }
    if extra_metadata:
        meta.update(extra_metadata)

    return StokesResult(name=name, scalar_summary=scalar, maps={}, metadata=meta)
