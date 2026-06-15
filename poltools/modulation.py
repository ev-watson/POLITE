"""
poltools.modulation — Extract normalized Stokes parameters from HWP modulation.

After aperture photometry, each source has ordinary and extraordinary fluxes at
several half-wave plate angles. These functions fit the modulation curve to
recover ``q = Q/I`` and ``u = U/I``.

Three methods are provided:

* :func:`lsq_modulation` — weighted least-squares fit of
  ``R(θ) = q cos4θ + u sin4θ`` (Magalhães et al. 1984; Ramírez et al. 2017).
  Default when frames are flat-fielded; returns covariance, chi-squared, and
  residual polarization uncertainty.

* :func:`double_ratio` — ratio-of-ratios (Tinbergen 1996; Masiero et al. 2007).
  Cancels flat-field errors; requires half-wave plate angles
  {0°, 22.5°, 45°, 67.5°}.

* :func:`double_difference` — compares beam ratios at paired angles; useful as
  a cross-check.
"""

from __future__ import annotations

from typing import Dict, List

import numpy as np

from ._types import BeamFlux
from .errors import residual_sigma_p, sigma_r


def ratio_r(f_o: float, f_e: float) -> float:
    """Beam polarization ratio ``R = (f_extraordinary − f_ordinary) / sum``."""
    s = f_e + f_o
    return float((f_e - f_o) / s) if s != 0 else float("nan")


def _validate_beam_fluxes(beam_fluxes: List[BeamFlux]) -> None:
    """Require finite, positive ordinary and extraordinary fluxes at every angle."""
    bad = [
        (bf.hwp_deg, bf.f_o, bf.f_e)
        for bf in beam_fluxes
        if not (np.isfinite(bf.f_o) and np.isfinite(bf.f_e)
                and bf.f_o > 0.0 and bf.f_e > 0.0)
    ]
    if bad:
        detail = ", ".join(f"{a:g} deg (ordinary={fo:g}, extraordinary={fe:g})"
                           for a, fo, fe in bad)
        raise ValueError(
            "modulation requires finite, positive beam fluxes at every "
            f"half-wave plate angle; bad values at: {detail}"
        )


def _index_by_angle(beam_fluxes: List[BeamFlux]) -> Dict[float, BeamFlux]:
    out: Dict[float, BeamFlux] = {}
    for bf in beam_fluxes:
        out[round(bf.hwp_deg % 180.0, 3)] = bf
    return out


def double_difference(beam_fluxes: List[BeamFlux]) -> Dict[str, float]:
    """Double-difference reduction at half-wave plate angles {0, 22.5, 45, 67.5}°.

    Returns ``{q, u, sigma_q, sigma_u, R0, R45, R22, R67}``.
    """
    idx = _index_by_angle(beam_fluxes)
    needed = [0.0, 45.0, 22.5, 67.5]
    for ang in needed:
        if round(ang, 3) not in idx:
            raise ValueError(
                f"double_difference requires half-wave plate angle {ang} deg; "
                f"have {sorted(idx.keys())}"
            )
    _validate_beam_fluxes([idx[round(ang, 3)] for ang in needed])

    def R_and_sig(ang):
        bf = idx[round(ang, 3)]
        return ratio_r(bf.f_o, bf.f_e), sigma_r(bf.f_o, bf.f_e, bf.sig_o, bf.sig_e)

    R0, s0 = R_and_sig(0.0)
    R45, s45 = R_and_sig(45.0)
    R22, s22 = R_and_sig(22.5)
    R67, s67 = R_and_sig(67.5)

    q = 0.5 * (R0 - R45)
    u = 0.5 * (R22 - R67)
    sigma_q = 0.5 * np.sqrt(s0 ** 2 + s45 ** 2)
    sigma_u = 0.5 * np.sqrt(s22 ** 2 + s67 ** 2)

    return {
        "q": float(q), "u": float(u),
        "sigma_q": float(sigma_q), "sigma_u": float(sigma_u),
        "R0": R0, "R45": R45, "R22": R22, "R67": R67,
    }


def _beam_ratio_pair(idx, tha, thb):
    """Return ``(sqrt(RR), sigma_lnRR)`` for ``RR = r(tha)/r(thb)``, ``r = f_e/f_o``."""
    a = idx[round(tha, 3)]
    b = idx[round(thb, 3)]
    rr = (a.f_e / a.f_o) / (b.f_e / b.f_o)
    s = np.sqrt(rr)

    def rel2(f, sig):
        return (sig / f) ** 2 if f != 0 else 0.0

    var_lnrr = rel2(a.f_e, a.sig_e) + rel2(a.f_o, a.sig_o) \
        + rel2(b.f_e, b.sig_e) + rel2(b.f_o, b.sig_o)
    return float(s), float(np.sqrt(var_lnrr))


def double_ratio(beam_fluxes: List[BeamFlux]) -> Dict[str, float]:
    """Flat-field-independent double-ratio reduction.

    Per-beam throughput cancels when comparing ratios of extraordinary-to-
    ordinary flux at paired half-wave plate angles:

    ``RR = r(0°)/r(45°)`` and ``r(22.5°)/r(67.5°)`` with ``r = F_e/F_o``.

    Returns ``{q, u, sigma_q, sigma_u, s_q, s_u}``.
    """
    idx = _index_by_angle(beam_fluxes)
    needed = (0.0, 45.0, 22.5, 67.5)
    for ang in needed:
        if round(ang, 3) not in idx:
            raise ValueError(
                f"double_ratio requires half-wave plate angle {ang} deg; "
                f"have {sorted(idx.keys())}"
            )
    _validate_beam_fluxes([idx[round(ang, 3)] for ang in needed])
    s_q, sig_lnrr_q = _beam_ratio_pair(idx, 0.0, 45.0)
    s_u, sig_lnrr_u = _beam_ratio_pair(idx, 22.5, 67.5)
    q = (s_q - 1.0) / (s_q + 1.0)
    u = (s_u - 1.0) / (s_u + 1.0)
    sigma_q = s_q * sig_lnrr_q / (s_q + 1.0) ** 2
    sigma_u = s_u * sig_lnrr_u / (s_u + 1.0) ** 2
    return {"q": float(q), "u": float(u),
            "sigma_q": float(sigma_q), "sigma_u": float(sigma_u),
            "s_q": s_q, "s_u": s_u}


def lsq_modulation(beam_fluxes: List[BeamFlux]) -> Dict[str, object]:
    """Least-squares fit of the modulation curve (needs at least four angles).

    Solves ``R_i = q cos4ψ_i + u sin4ψ_i`` by weighted linear least squares.

    .. warning::
       Assumes flat-fielded data or matched beam throughput. Pixel-to-pixel
       sensitivity variation (photo-response non-uniformity) does **not**
       cancel between the two aperture positions. Use :func:`double_ratio` when
       flat fields are missing or unreliable.
    """
    bfs = sorted(beam_fluxes, key=lambda b: b.hwp_deg)
    n = len(bfs)
    if n < 4:
        raise ValueError(f"lsq_modulation needs at least four angles; got {n}")
    _validate_beam_fluxes(bfs)

    psi = np.array([np.deg2rad(b.hwp_deg) for b in bfs])
    z = np.array([ratio_r(b.f_o, b.f_e) for b in bfs])
    sig_z = np.array([sigma_r(b.f_o, b.f_e, b.sig_o, b.sig_e) for b in bfs])

    A = np.column_stack([np.cos(4 * psi), np.sin(4 * psi)])

    have_w = np.all(np.isfinite(sig_z)) and np.all(sig_z > 0)
    if have_w:
        w = 1.0 / sig_z
        Aw = A * w[:, None]
        zw = z * w
        coef, *_ = np.linalg.lstsq(Aw, zw, rcond=None)
        cov = np.linalg.inv(Aw.T @ Aw)
    else:
        coef, *_ = np.linalg.lstsq(A, z, rcond=None)
        resid = z - A @ coef
        dof = max(n - 2, 1)
        s2 = float(resid @ resid) / dof
        cov = s2 * np.linalg.inv(A.T @ A)

    q, u = float(coef[0]), float(coef[1])
    model = A @ coef
    resid = z - model
    chi2 = float(np.sum((resid / sig_z) ** 2)) if have_w else float("nan")

    sigma_q = float(np.sqrt(cov[0, 0]))
    sigma_u = float(np.sqrt(cov[1, 1]))
    sig_p_resid = residual_sigma_p(z, q, u, n)

    return {
        "q": q, "u": u,
        "sigma_q": sigma_q, "sigma_u": sigma_u,
        "cov_qu": float(cov[0, 1]),
        "sigma_p_resid": sig_p_resid,
        "chi2": chi2, "dof": n - 2, "n_angles": n,
        "z": z, "model": model, "resid": resid, "psi_deg": np.rad2deg(psi),
    }
