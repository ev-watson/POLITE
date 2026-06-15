"""
poltools.mueller — Mueller-matrix forward model for the optical train.

Builds Stokes 4-vectors and Mueller matrices for each optical element, then
computes how much light reaches the ordinary and extraordinary beams at a
given half-wave plate angle.

For an ideal half-wave plate the extraordinary and ordinary intensities are
``I_e = ½[I + Q cos4θ + U sin4θ]`` and ``I_o = ½[I − Q cos4θ − U sin4θ]``
(Masiero et al. 2007; DUSTPol design). Retardance δ below 180° reduces the
modulation amplitude.
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np


def stokes_vector(I: float, q: float = 0.0, u: float = 0.0, v: float = 0.0,
                  normalized: bool = True) -> np.ndarray:
    """Build Stokes 4-vector ``(I, Q, U, V)``.

    If ``normalized``, ``q,u,v`` are fractional (Q = q·I, …).
    """
    if normalized:
        return np.array([I, q * I, u * I, v * I], dtype=float)
    return np.array([I, q, u, v], dtype=float)


def M_rotator(theta_deg: float) -> np.ndarray:
    """Mueller rotation of the reference frame by θ [deg]."""
    t = np.deg2rad(theta_deg)
    c, s = np.cos(2 * t), np.sin(2 * t)
    return np.array([
        [1, 0, 0, 0],
        [0, c, s, 0],
        [0, -s, c, 0],
        [0, 0, 0, 1],
    ], dtype=float)


def M_retarder(delta_deg: float, fast_axis_deg: float = 0.0) -> np.ndarray:
    """Mueller matrix of a linear retarder.

    Built as ``R(−φ) · M_ret(δ) · R(φ)``. For δ = 180° this is the HWP matrix.
    """
    d = np.deg2rad(delta_deg)
    cd, sd = np.cos(d), np.sin(d)
    base = np.array([
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, cd, sd],
        [0, 0, -sd, cd],
    ], dtype=float)
    R = M_rotator(fast_axis_deg)
    Rinv = M_rotator(-fast_axis_deg)
    return Rinv @ base @ R


def M_hwp(theta_deg: float, retardance_deg: float = 180.0) -> np.ndarray:
    """Half-wave plate Mueller matrix at angle θ [degrees]."""
    return M_retarder(retardance_deg, theta_deg)


def M_linear_polarizer(transmission_axis_deg: float = 0.0) -> np.ndarray:
    """Mueller matrix of an ideal linear polarizer."""
    a = np.deg2rad(transmission_axis_deg)
    c, s = np.cos(2 * a), np.sin(2 * a)
    return 0.5 * np.array([
        [1, c, s, 0],
        [c, c * c, c * s, 0],
        [s, c * s, s * s, 0],
        [0, 0, 0, 0],
    ], dtype=float)


def system_mueller(hwp_deg: float, rotator_deg: float = 0.0,
                   retardance_deg: float = 180.0,
                   analyzer_deg: float = 0.0,
                   M_tel: Optional[np.ndarray] = None) -> np.ndarray:
    """Chain Mueller matrix: analyzer · half-wave plate · rotator · telescope.

    Ideal filters are omitted (they commute and act as scalars).
    """
    M = M_linear_polarizer(analyzer_deg) @ M_hwp(hwp_deg, retardance_deg) @ M_rotator(rotator_deg)
    if M_tel is not None:
        M = M @ M_tel
    return M


def oe_intensities(stokes: np.ndarray, hwp_deg: float,
                   retardance_deg: float = 180.0,
                   efficiency: float = 1.0,
                   ip: Tuple[float, float] = (0.0, 0.0),
                   rotator_deg: float = 0.0) -> Tuple[float, float]:
    """Ordinary and extraordinary analyzed intensities at one HWP angle.

    Parameters
    ----------
    stokes : array_like
        Incident Stokes ``(I, Q, U, V)``.
    efficiency : float
        Modulation efficiency in [0, 1].
    ip : (q0, u0)
        Fractional instrumental polarization on the sky beam before the
        field rotator. Downstream telescope polarization is removed by
        standard-star calibration.
    rotator_deg : float
        Field-rotator angle [degrees].

    Returns
    -------
    (I_o, I_e) : tuple of float
        Flux-conserving split (``I_o + I_e = I``). The e-beam carries the +Q′
        analyzer axis; ``(I_e − I_o)/(I_e + I_o) = q cos4θ + u sin4θ`` ideally.
    """
    S = np.asarray(stokes, dtype=float).copy()
    I0 = S[0]
    if I0 != 0:
        S[1] += ip[0] * I0
        S[2] += ip[1] * I0

    Sr = M_rotator(rotator_deg) @ S
    Sp = M_hwp(hwp_deg, retardance_deg) @ Sr
    ac = efficiency * Sp[1]
    I_e = 0.5 * (Sp[0] + ac)
    I_o = 0.5 * (Sp[0] - ac)
    return float(I_o), float(I_e)
