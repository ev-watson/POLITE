"""
poltools.plotting — Publication figures for polarimetry results.

Modulation curves, the q–u plane with error ellipses, recovered-vs-injected
diagnostics, the MC pull histogram, and polarization-vector overlays. Images use
``origin='upper'`` (CLAUDE.md). All functions accept an optional ``ax`` and
return the Axes for composition.
"""

from __future__ import annotations

from typing import List, Optional, Sequence

import numpy as np

import matplotlib
matplotlib.use("Agg")  # headless / scriptable
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse

from ._types import BeamFlux
from .modulation import ratio_R


def plot_modulation_curve(beam_fluxes: List[BeamFlux], q: float, u: float,
                          ax=None, title: str = "HWP modulation"):
    """Plot R(θ) data points and the fitted ``q cos4θ + u sin4θ`` model."""
    if ax is None:
        _, ax = plt.subplots(figsize=(5, 3.2))
    ang = np.array([b.hwp_deg for b in beam_fluxes])
    R = np.array([ratio_R(b.f_o, b.f_e) for b in beam_fluxes])
    sR = np.array([
        np.hypot(b.sig_o, b.sig_e) / max(b.f_o + b.f_e, 1e-30) for b in beam_fluxes
    ])
    order = np.argsort(ang)
    ax.errorbar(ang[order], R[order], yerr=sR[order], fmt="o", color="k",
                ms=4, capsize=2, label="data")
    th = np.linspace(0, max(ang.max(), 67.5), 400)
    model = q * np.cos(4 * np.deg2rad(th)) + u * np.sin(4 * np.deg2rad(th))
    ax.plot(th, model, "-", color="C3", lw=1.5, label="model")
    ax.axhline(0, color="0.7", lw=0.8)
    ax.set_xlabel("HWP angle θ [deg]")
    ax.set_ylabel("R = (F_e − F_o)/(F_e + F_o)")
    ax.set_title(title)
    ax.legend(fontsize=8)
    return ax


def plot_qu_plane(results, ax=None, title: str = "q–u plane",
                  injected=None):
    """Scatter calibrated (q, u) with 1σ error ellipses.

    ``results`` is a sequence of StokesResult; ``injected`` optional list of
    (q_true, u_true) to overplot as crosses.
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(4.6, 4.4))
    for r in results:
        s = r.scalar_summary
        q, u = s["q"], s["u"]
        sq, su = s.get("sigma_q", 0.0), s.get("sigma_u", 0.0)
        ax.add_patch(Ellipse((q, u), 2 * sq, 2 * su, fill=False,
                             edgecolor="C0", lw=0.8, alpha=0.7))
        ax.plot(q, u, "o", color="C0", ms=3)
    if injected is not None:
        inj = np.atleast_2d(injected)
        ax.plot(inj[:, 0], inj[:, 1], "x", color="C3", ms=7, label="injected")
        ax.legend(fontsize=8)
    ax.axhline(0, color="0.8", lw=0.7)
    ax.axvline(0, color="0.8", lw=0.7)
    ax.set_xlabel("q = Q/I")
    ax.set_ylabel("u = U/I")
    ax.set_aspect("equal", adjustable="datalim")
    ax.set_title(title)
    return ax


def plot_recovered_vs_injected(injected_p, recovered_p, sigma_p=None, ax=None,
                               title: str = "Recovered vs injected p"):
    """Recovered p (with error bars) against injected p, with the 1:1 line."""
    if ax is None:
        _, ax = plt.subplots(figsize=(4.6, 4.4))
    injected_p = np.asarray(injected_p, dtype=float)
    recovered_p = np.asarray(recovered_p, dtype=float)
    yerr = None if sigma_p is None else np.asarray(sigma_p, dtype=float)
    ax.errorbar(injected_p * 100, recovered_p * 100, yerr=None if yerr is None else yerr * 100,
                fmt="o", color="C0", ms=4, capsize=2)
    lim = [0, max(injected_p.max(), recovered_p.max()) * 100 * 1.1]
    ax.plot(lim, lim, "--", color="0.5", lw=1, label="1:1")
    ax.set_xlabel("injected p [%]")
    ax.set_ylabel("recovered p [%]")
    ax.set_title(title)
    ax.legend(fontsize=8)
    return ax


def plot_pull_histogram(pulls, ax=None, title: str = "Pull distribution"):
    """Histogram of pulls (x̂−x_true)/σ with a unit Gaussian overlay."""
    if ax is None:
        _, ax = plt.subplots(figsize=(5, 3.4))
    pulls = np.asarray(pulls, dtype=float)
    pulls = pulls[np.isfinite(pulls)]
    ax.hist(pulls, bins=30, density=True, color="C0", alpha=0.6,
            label=f"μ={pulls.mean():.2f}, σ={pulls.std(ddof=1):.2f}")
    x = np.linspace(-4, 4, 200)
    ax.plot(x, np.exp(-0.5 * x ** 2) / np.sqrt(2 * np.pi), "C3-", lw=1.5,
            label="N(0,1)")
    ax.set_xlabel("pull = (p̂ − p_true)/σ_p")
    ax.set_ylabel("density")
    ax.set_title(title)
    ax.legend(fontsize=8)
    return ax


def plot_pol_vectors(image: np.ndarray, results, positions, scale: float = 200.0,
                     ax=None, title: str = "Polarization vectors"):
    """Overlay polarization vectors (length ∝ p, orientation = θ) on an image."""
    if ax is None:
        _, ax = plt.subplots(figsize=(5.5, 5))
    vmin, vmax = np.percentile(image, [5, 99])
    ax.imshow(image, origin="upper", cmap="gray", vmin=vmin, vmax=vmax)
    for r, (x, y) in zip(results, positions):
        s = r.scalar_summary
        p = s.get("p_report", s.get("p", 0.0))
        th = np.deg2rad(s["theta_deg"])
        L = p * scale
        dx = L * np.cos(th)
        dy = L * np.sin(th)
        ax.plot([x - dx, x + dx], [y - dy, y + dy], "-", color="C1", lw=1.5)
    ax.set_title(title)
    ax.set_xlabel("x [px]")
    ax.set_ylabel("y [px]")
    return ax
