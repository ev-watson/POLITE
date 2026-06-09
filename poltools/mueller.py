"""
poltools.mueller — Mueller-calculus forward model for the POLITE chain.

Provides the Stokes 4-vector helpers and the Mueller matrices for the optical
train (frame rotation, general retarder, ideal linear analyzer) plus the
convenience ``oe_intensities`` that maps an incident Stokes vector to the two
analyzed (ordinary/extraordinary) beam intensities for a given HWP angle.

Provenance (CLAUDE.md Sources A/B)
----------------------------------
* Mueller calculus is the standard formalism (Source B textbooks; used by
  DBIP/Masiero 2007 and DUSTPol).
* The ideal-HWP modulation
  ``I'_e = ½[I + Q cos4θ + U sin4θ]``, ``I'_o = ½[I − Q cos4θ − U sin4θ]``
  reproduced by :func:`oe_intensities` is exactly the ideal-HWP modulation
  (Masiero 2007, Source B; DUSTPol §2.1, Source A).
* The real-retarder depolarization scaling ``∝ (1 − cos δ)`` (Masiero 2007,
  Source B) is captured by using a general retardance ``δ`` rather than the
  ideal δ=180°.

Everything is full 4x4 / 4-vector (V-ready). The linear pipeline does not solve
for V; ``M_retarder(90, ...)`` is correct for a future QWP Stokes-V mode.
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np


def stokes_vector(I: float, q: float = 0.0, u: float = 0.0, v: float = 0.0,
                  normalized: bool = True) -> np.ndarray:
    """Build a Stokes 4-vector ``(I, Q, U, V)``.

    If ``normalized`` is True, ``q,u,v`` are fractional (Q = q·I, ...).
    Otherwise they are absolute Q,U,V.
    """
    if normalized:
        return np.array([I, q * I, u * I, v * I], dtype=float)
    return np.array([I, q, u, v], dtype=float)


def M_rotator(theta_deg: float) -> np.ndarray:
    """Mueller rotation of the reference frame by ``theta`` (deg)."""
    t = np.deg2rad(theta_deg)
    c, s = np.cos(2 * t), np.sin(2 * t)
    return np.array([
        [1, 0, 0, 0],
        [0, c, s, 0],
        [0, -s, c, 0],
        [0, 0, 0, 1],
    ], dtype=float)


def M_retarder(delta_deg: float, fast_axis_deg: float = 0.0) -> np.ndarray:
    """Mueller matrix of a general linear retarder.

    Parameters
    ----------
    delta_deg : float
        Retardance (HWP = 180, QWP = 90).
    fast_axis_deg : float
        Fast-axis orientation (deg).

    Notes
    -----
    Built as ``R(-φ) · M_ret(δ) · R(φ)``. For δ=180 this reduces to the
    half-wave-plate matrix with the ``cos4φ / sin4φ`` signature.
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
    """Half-wave-plate Mueller matrix at angle ``theta`` (deg).

    Convenience wrapper around :func:`M_retarder` with δ=``retardance_deg``.
    For the ideal HWP (δ=180) this equals
    ``[[1,0,0,0],[0,cos4θ,sin4θ,0],[0,sin4θ,−cos4θ,0],[0,0,0,−1]]``.
    """
    return M_retarder(retardance_deg, theta_deg)


def M_linear_polarizer(transmission_axis_deg: float = 0.0) -> np.ndarray:
    """Mueller matrix of an ideal linear polarizer/analyzer."""
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
    """Full chain Mueller matrix up to (and including) one analyzer beam.

    ``M = M_analyzer · M_HWP(θ,δ) · R_PWI4(α) · M_telescope``.

    Ordering follows the real POLITE light path: CDK20 (``M_telescope``) →
    **PWI4 field rotator** ``R_PWI4(α)`` → L3 cut filter → HWP(θ,δ) → EFW filter
    → α-BBO Savart analyzer. The L3 and EFW filters are ideal (scalar × identity)
    and commute, so they are omitted here (they do not alter q,u). The telescope
    Mueller (instrumental polarization) defaults to identity.
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
    """Analyzed ordinary/extraordinary intensities for one HWP angle.

    Parameters
    ----------
    stokes : array_like
        Incident Stokes 4-vector ``(I, Q, U, V)``.
    hwp_deg : float
        HWP angle θ (deg).
    retardance_deg : float
        Retarder retardance δ (180 = ideal HWP). Non-ideal δ introduces the
        ``(1 − cos δ)`` depolarization of the linear signal (Masiero 2007).
    efficiency : float
        Polarization (modulation) efficiency in [0, 1]; scales the AC term.
    ip : (q0, u0)
        Additive instrumental polarization (fractional), injected on the incident
        beam **upstream of the PWI4 rotator** — i.e. it models the **CDK20
        (telescope) IP**, which therefore rotates with α. IP introduced *downstream*
        of the rotator (L3 / EFW / Savart, fixed in the instrument frame) is not
        separated here; it is absorbed by the standard-star IP calibration. Used by
        the simulator to create a known IP to calibrate.
    rotator_deg : float
        **PWI4 field-rotator** angle α (deg); rotates the incident frame ahead of
        the HWP.

    Returns
    -------
    (I_o, I_e) : tuple of float
        Ordinary and extraordinary analyzed intensities. ``I_o + I_e == I``
        (flux-conserving split). The **e-beam carries the +Q' analyzer axis**
        (see _types convention), so
        ``(I_e − I_o)/(I_e + I_o) = q cos4θ + u sin4θ`` in the ideal limit.
    """
    S = np.asarray(stokes, dtype=float).copy()
    I0 = S[0]
    if I0 != 0:
        # inject instrumental polarization (fractional) on the incident beam
        S[1] += ip[0] * I0
        S[2] += ip[1] * I0

    Sr = M_rotator(rotator_deg) @ S
    Sp = M_hwp(hwp_deg, retardance_deg) @ Sr  # after HWP
    # ideal analyzer along Q': e = +Q' arm, o = -Q' arm
    ac = efficiency * Sp[1]
    I_e = 0.5 * (Sp[0] + ac)
    I_o = 0.5 * (Sp[0] - ac)
    return float(I_o), float(I_e)
