"""
caltools.plotting — Diagnostic figures for detector characterization.

Images use ``origin='upper'`` (astronomical convention).
"""

from __future__ import annotations

from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.axes_grid1 import make_axes_locatable

from ._types import AnalysisResult


def image_with_colorbar(
    ax: plt.Axes,
    data: np.ndarray,
    label: str = "",
    cmap: str = "viridis",
    percentile_clip: Tuple[float, float] = (0.5, 99.5),
    origin: str = "upper",
    **imshow_kw,
) -> plt.cm.ScalarMappable:
    """Display a 2-D image with a matched colorbar.

    Parameters
    ----------
    ax : Axes
        Target axes.
    data : 2-D array
        Image data.
    label : str
        Colorbar label.
    cmap : str
        Colormap name.
    percentile_clip : tuple of float
        ``(low, high)`` percentiles for vmin/vmax clipping.
    origin : str
        Image origin (default ``"upper"``).

    Returns
    -------
    ScalarMappable (the imshow return value).
    """
    if "vmin" not in imshow_kw:
        imshow_kw["vmin"] = np.percentile(data, percentile_clip[0])
    if "vmax" not in imshow_kw:
        imshow_kw["vmax"] = np.percentile(data, percentile_clip[1])

    im = ax.imshow(data, origin=origin, cmap=cmap, **imshow_kw)
    ax.set_xlabel("x (px)")
    ax.set_ylabel("y (px)")

    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="4%", pad=0.05)
    ax.figure.colorbar(im, cax=cax, label=label)
    return im


def quick_view(file_path: str) -> None:
    """Quickly view a FITS file with matplotlib.

    Parameters
    ----------
    file_path : str
        Path to FITS file.
    """
    from astropy.io import fits

    data = fits.getdata(file_path, memmap=False)
    fig, ax = plt.subplots()
    image_with_colorbar(ax, data, label="ADU")
    ax.set_title(file_path)
    plt.show()
    return None


def ptc_plot(
    ax: plt.Axes,
    ptc_result: AnalysisResult,
    log_scale: bool = False,
) -> None:
    """Plot PTC data points with fit lines.

    Expects ``ptc_result.metadata`` to contain:
        ``signal``, ``variance``, ``labels`` (per-point group label),
        ``fit_free`` dict with ``slope``, ``intercept``, ``gain``,
        ``fit_fixed`` dict with ``slope``, ``intercept``, ``gain``.
    """
    meta = ptc_result.metadata
    signal = meta["signal"]
    variance = meta["variance"]
    labels = meta["labels"]
    fit_free = meta["fit"]

    # Color each group
    unique_labels = list(dict.fromkeys(labels))
    cmap = plt.get_cmap("tab10")
    label_color = {lb: cmap(i % 10) for i, lb in enumerate(unique_labels)}

    for lb in unique_labels:
        mask = np.array([l == lb for l in labels])
        ax.scatter(
            signal[mask], variance[mask],
            s=55, color=label_color[lb], label=lb,
        )

    s_plot = np.linspace(0, signal.max() * 1.05, 300)
    fit_line = fit_free["slope"] * s_plot + fit_free["intercept"]

    fit = fit_free["slope"] * signal + fit_free["intercept"]
    r_squared = 1.0 - np.sum((variance - fit)**2) / \
    np.sum((variance - np.mean(variance))**2)

    ax.plot(
        s_plot, fit_line,
        "k-",
        label=f"gain = {fit_free['gain']:.3f} e-/ADU; R² = {r_squared:.5f}",
    )

    ax.set(
    xlabel="Mean bias-subtracted signal (ADU)",
    ylabel="Pair-difference variance (ADU²)",
    title="Mean variance versus mean — cover PTC only",
    )
    ax.grid()
    ax.legend()
    plt.tight_layout()
    ax.legend()

    if log_scale:
        ax.set_xscale("log")
        ax.set_yscale("log")


def momsdom_plot(
    ax: plt.Axes,
    momsdom_result: AnalysisResult,
    log_scale: bool = False,
) -> None:
    """Plot mean of means and standard deviation of the means versus frame number.

    Expects ``momsdom_result.metadata`` to contain:
        ``frame_numbers``, ``mean``, ``std``.
    """
    md = momsdom_result.metadata
    exposures = md["exposures_s"]
    moms = md["moms_adu"]
    sdoms = md["sdoms_adu"]
    mom_rates = md["mom_rates_adu_s"]
    sdom_rates = md["sdom_rates_adu_s"]

    ax.errorbar(exposures, moms, yerr=sdoms, fmt="o-", capsize=4, label="MOM ± SDOM")
    ax.set(xlabel="Exposure time (s)", ylabel="Bias-subtracted signal (ADU)", title="Flat MOMs and SDOMs")
    ax.grid()
    ax.legend()

    ax.errorbar(exposures, mom_rates, yerr=sdom_rates, fmt="o-", capsize=4, label="MOM/exptime ± SDOM")
    ax.set(xlabel="Exposure time (s)", ylabel="Signal rate (ADU/s)", title="Twilight-rate check")
    ax.grid()
    ax.legend()
    plt.tight_layout()

    if log_scale:
            ax.set_xscale("log")
            ax.set_yscale("log")


def histogram_gaussian_overlay(
    ax: plt.Axes,
    data: np.ndarray,
    n_sigma: float = 8.0,
    log_scale: bool = True,
    bins_per_adu: float = 1.0,
    color: str = "steelblue",
    label_data: str = "Data",
) -> None:
    """Histogram with Gaussian overlay, optionally on log scale.

    Parameters
    ----------
    ax : Axes
        Target axes.
    data : 1-D array
        Values to histogram.
    n_sigma : float
        Range in sigma around the mean.
    log_scale : bool
        Use log y-axis.
    bins_per_adu : float
        Bin width in ADU.
    color : str
        Histogram bar color.
    label_data : str
        Legend label for the histogram.
    """
    from scipy import stats as sp_stats

    flat = data.ravel()
    mu, sig = np.mean(flat), np.std(flat)

    bins = np.arange(mu - n_sigma * sig, mu + n_sigma * sig, 1.0 / bins_per_adu)
    ax.hist(flat, bins=bins, density=True, color=color, alpha=0.7, label=label_data)

    x = np.linspace(bins[0], bins[-1], 500)
    ax.plot(
        x, sp_stats.norm.pdf(x, mu, sig),
        "r-", lw=2,
        label=f"Gaussian ({chr(956)}={mu:.2f}, {chr(963)}={sig:.2f})",
    )
    ax.set_xlabel("ADU")
    ax.set_ylabel("Probability density")
    ax.legend()
    if log_scale:
        ax.set_yscale("log")
        ax.set_ylim(bottom=1e-7)


def noise_map_with_histogram(
    fig: plt.Figure,
    noise_map: np.ndarray,
    label: str = "Read noise",
    unit: str = "e⁻",
    cmap: str = "viridis",
) -> Tuple[plt.Axes, plt.Axes]:
    """Side-by-side noise map + histogram.

    Parameters
    ----------
    fig : Figure
        Matplotlib figure (should be empty or have room for 1x2 subplots).
    noise_map : 2-D array
        The noise map to display.
    label : str
        Label for the quantity.
    unit : str
        Physical unit string.
    cmap : str
        Colormap.

    Returns
    -------
    (ax_map, ax_hist) : tuple of Axes
    """
    ax_map, ax_hist = fig.subplots(1, 2)

    image_with_colorbar(
        ax_map, noise_map, label=f"{unit} rms", cmap=cmap,
    )
    med = np.median(noise_map)
    ax_map.set_title(f"{label} map [{unit}]\nmedian = {med:.2f} {unit}")

    flat = noise_map.ravel()
    ax_hist.hist(flat, bins=200, color="steelblue", alpha=0.7)
    ax_hist.axvline(med, ls="--", color="red", lw=1.5, label=f"median = {med:.2f}")
    ax_hist.set_xlabel(f"{label} ({unit})")
    ax_hist.set_xlim(auto=True)
    ax_hist.set_ylabel("Pixel count")
    ax_hist.set_title(f"{label} distribution")
    ax_hist.legend()

    fig.tight_layout()
    return ax_map, ax_hist


def summary_table(
    results: List[AnalysisResult],
    title: str = "Analysis Summary",
) -> str:
    """Build a Markdown summary table from a list of AnalysisResults.

    Returns a string suitable for ``IPython.display.Markdown``.
    """
    lines = [f"## {title}", "", "| Analysis | Parameter | Value |", "|---|---|---|"]
    for r in results:
        for k, v in r.scalar_summary.items():
            if isinstance(v, float):
                lines.append(f"| {r.name} | {k} | {v:.4f} |")
            else:
                lines.append(f"| {r.name} | {k} | {v} |")
    return "\n".join(lines)
