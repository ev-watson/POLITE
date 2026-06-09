"""
poltools.modulation — HWP modulation → normalized Stokes (q, u).

Three literature-grounded reductions of a half-wave-plate sequence:

* :func:`double_ratio` (point sources; *primary*). Tinbergen (1996);
  DBIP / Masiero 2007 (Source B); SOLVEPOL (Source B). With the per-angle beam
  ratio ``r(θ) = F_e/F_o``,
  ``RR_q = r(0°)/r(45°)``, ``q = (√RR_q − 1)/(√RR_q + 1)`` and the analogous
  ``u`` from ``r(22.5°)/r(67.5°)``. The per-beam throughput ``t_e/t_o`` appears
  as a common factor in every ``r`` and **cancels exactly** in the ratio, so the
  result is rigorously **flat-field- and seeing-independent**.

* :func:`double_difference` (first-order comparator). With
  ``R(θ) = (F_e − F_o)/(F_e + F_o)``,
  ``q = ½[R(0°) − R(45°)]``, ``u = ½[R(22.5°) − R(67.5°)]``. Cancels seeing /
  transparency and beam throughput only to first order (kept for cross-checks).

* :func:`lsq_modulation` (least-squares modulation fit, N≥4 angles;
  cross-check). SOLVEPOL (Ramírez et al. 2017, Source B) / Magalhães et al.
  (1984). ``z_i = R(ψ_i) = q cos4ψ_i + u sin4ψ_i`` solved by linear least
  squares (assumes matched beams or pre-normalized o/e total fluxes).

All return fractional Stokes ``q = Q/I``, ``u = U/I`` (and their covariances).
"""

from __future__ import annotations

from typing import Dict, List

import numpy as np

from ._types import BeamFlux
from .errors import residual_sigma_p, sigma_r


def ratio_r(f_o: float, f_e: float) -> float:
    """Normalized polarization ratio ``R = (f_e − f_o)/(f_e + f_o)``."""
    s = f_e + f_o
    return float((f_e - f_o) / s) if s != 0 else float("nan")


def _validate_beam_fluxes(beam_fluxes: List[BeamFlux]) -> None:
    """Enforce the reduction input invariant: finite, positive o/e fluxes.

    A non-positive or non-finite beam flux signals a non-detection, blended,
    saturated, or over-sky-subtracted source — it cannot yield a meaningful
    polarization ratio. All three reductions reject such input identically and
    loudly (fail-closed) instead of returning silent garbage
    (``double_difference``, ``lsq_modulation``) or crashing with an opaque
    ``ZeroDivisionError`` (``double_ratio``). This is input validation only; no
    analysis method outside CLAUDE.md Sources A/B is introduced.
    """
    bad = [
        (bf.hwp_deg, bf.f_o, bf.f_e)
        for bf in beam_fluxes
        if not (np.isfinite(bf.f_o) and np.isfinite(bf.f_e)
                and bf.f_o > 0.0 and bf.f_e > 0.0)
    ]
    if bad:
        detail = ", ".join(f"{a:g} deg (f_o={fo:g}, f_e={fe:g})"
                           for a, fo, fe in bad)
        raise ValueError(
            "modulation requires finite, positive o/e fluxes at every HWP "
            f"angle; got non-positive/non-finite flux at: {detail}"
        )


def _index_by_angle(beam_fluxes: List[BeamFlux]) -> Dict[float, BeamFlux]:
    """Map HWP angle (mod 180, rounded to 1e-3 deg) → BeamFlux for lookup."""
    out: Dict[float, BeamFlux] = {}
    for bf in beam_fluxes:
        out[round(bf.hwp_deg % 180.0, 3)] = bf
    return out


def double_difference(beam_fluxes: List[BeamFlux]) -> Dict[str, float]:
    """Double-difference reduction over HWP angles {0, 22.5, 45, 67.5}.

    Returns ``{q, u, sigma_q, sigma_u, R0, R45, R22, R67}``.
    """
    idx = _index_by_angle(beam_fluxes)
    needed = [0.0, 45.0, 22.5, 67.5]
    for ang in needed:
        if round(ang, 3) not in idx:
            raise ValueError(
                f"double_difference requires HWP angle {ang} deg; have "
                f"{sorted(idx.keys())}"
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
    """Return (sqrt(RR), sigma_lnRR) for the angle pair (tha, thb).

    ``RR = r(tha)/r(thb)`` with ``r = f_e/f_o``; ``s = sqrt(RR)``.
    ``Var(lnRR) = Σ (σ/f)²`` over the four involved fluxes.

    Precondition: all four fluxes are finite and positive (enforced by
    :func:`_validate_beam_fluxes` at the public boundary), so the divisions and
    ``sqrt`` are well-defined.
    """
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
    """Flat-field-independent double-ratio reduction (HWP {0,22.5,45,67.5}).

    ``q = (√RR_q − 1)/(√RR_q + 1)`` with ``RR_q = r(0°)/r(45°)``, ``r=F_e/F_o``;
    ``u`` analogously from ``r(22.5°)/r(67.5°)``.

    Error propagation: with ``s = √RR``, ``q = (s−1)/(s+1)`` gives
    ``σ_q = s·σ_{lnRR} / (s+1)²`` (and likewise for u).

    Returns ``{q, u, sigma_q, sigma_u, s_q, s_u}``.
    """
    idx = _index_by_angle(beam_fluxes)
    needed = (0.0, 45.0, 22.5, 67.5)
    for ang in needed:
        if round(ang, 3) not in idx:
            raise ValueError(
                f"double_ratio requires HWP angle {ang} deg; have "
                f"{sorted(idx.keys())}"
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
    """Least-squares modulation fit over all provided HWP angles (N≥4).

    Solves ``z_i = q cos4ψ_i + u sin4ψ_i`` by (optionally weighted) linear
    least squares. Returns ``q, u``, their covariance, the per-fit χ², the
    SOLVEPOL residual σ_P, and the model/residual arrays.

    .. warning::
       Unlike :func:`double_ratio` (the flat-field-independent **primary**
       method, where per-beam throughput cancels in the ratio-of-ratios), this
       fit assumes the o/e beams are **matched / pre-normalized**: it does *not*
       cancel pixel-to-pixel response (PRNU/flat-field) between the o and e
       aperture positions (finding C-7). On real CMOS frames, flat-field first
       (the ``reduction.md`` PRNU map) before using ``lsq`` as a headline result.
    """
    bfs = sorted(beam_fluxes, key=lambda b: b.hwp_deg)
    n = len(bfs)
    if n < 4:
        raise ValueError(f"lsq_modulation needs N>=4 angles; got {n}")
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
    if have_w:
        chi2 = float(np.sum((resid / sig_z) ** 2))
    else:
        chi2 = float(np.nan)

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
