"""
poltools.photometry — Source detection, o/e pairing, aperture photometry.

Uses the Astropy-affiliated ``photutils`` (per CLAUDE.md "prefer Astropy/AAS
conventions"): ``DAOStarFinder`` for detection and
``CircularAperture``/``CircularAnnulus``/``ApertureStats`` for concentric-
aperture photometry with a sky annulus — the standard DBIP/SOLVEPOL operation
(Source B). Photometric uncertainties use the CCD equation (source shot noise +
sky + read noise + sky-estimation error).

Pixel convention: photutils positions are ``(x, y)`` with ``data[y, x]``,
identical to the rest of poltools (origin upper-left is a display choice only).
Fluxes are returned in **electrons**.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import photutils
from astropy.stats import sigma_clipped_stats
from photutils.aperture import (ApertureStats, CircularAnnulus,
                                CircularAperture, aperture_photometry)
from photutils.detection import DAOStarFinder

# Opt into the non-deprecated detection column names (x_centroid/y_centroid);
# photutils 3.0 deprecated the legacy xcentroid/ycentroid aliases (CLAUDE.md:
# avoid deprecated APIs). pair_oe also falls back to the legacy names.
try:
    photutils.future_column_names = True
except Exception:  # pragma: no cover - older photutils without the flag
    pass

from ._types import BeamFlux, BeamGeometry, PolConfig


def estimate_background(image: np.ndarray, sigma: float = 3.0) -> Tuple[float, float]:
    """Sigma-clipped ``(median, std)`` background of an image (ADU)."""
    _, median, std = sigma_clipped_stats(image, sigma=sigma)
    return float(median), float(std)


def detect_sources(image: np.ndarray, fwhm_px: float,
                   threshold_sigma: float = 5.0):
    """Detect point sources with DAOStarFinder.

    Returns the photutils detection table (with ``xcentroid``, ``ycentroid``)
    or ``None`` if nothing is found.
    """
    median, std = estimate_background(image)
    finder = DAOStarFinder(fwhm=fwhm_px, threshold=threshold_sigma * std)
    tbl = finder(image - median)
    return tbl


def pair_oe(detections, beam: BeamGeometry, tol_px: float = 2.0
            ) -> List[Tuple[Tuple[float, float], Tuple[float, float]]]:
    """Pair detections into (o, e) using the known beam offset vector.

    For each detection treated as an ordinary beam, look for a partner near
    ``o + offset``. Returns a list of ``((x_o, y_o), (x_e, y_e))`` pairs.
    """
    if detections is None or len(detections) == 0:
        return []
    xcol = "x_centroid" if "x_centroid" in detections.colnames else "xcentroid"
    ycol = "y_centroid" if "y_centroid" in detections.colnames else "ycentroid"
    xs = np.asarray(detections[xcol], dtype=float)
    ys = np.asarray(detections[ycol], dtype=float)
    dx, dy = beam.offset_xy()
    coords = np.column_stack([xs, ys])
    pairs = []
    used = set()
    for i, (xo, yo) in enumerate(coords):
        if i in used:
            continue
        target = np.array([xo + dx, yo + dy])
        d = np.hypot(coords[:, 0] - target[0], coords[:, 1] - target[1])
        d[i] = np.inf
        j = int(np.argmin(d))
        if d[j] <= tol_px and j not in used:
            pairs.append(((float(xo), float(yo)), (float(coords[j, 0]), float(coords[j, 1]))))
            used.add(i)
            used.add(j)
    return pairs


def measure_fluxes(image: np.ndarray, positions: Sequence[Tuple[float, float]],
                   cfg: PolConfig, *, r_ap: float, r_in: float, r_out: float,
                   bias_adu: float = 1000.0) -> Tuple[np.ndarray, np.ndarray]:
    """Aperture photometry at given positions. Returns (flux_e, sigma_e).

    Net flux = aperture sum − (annulus-median sky) × aperture area, converted to
    electrons. Uncertainty via the CCD equation:
    ``σ² = F_src + n_ap·(bg + RON²) + (n_ap²/n_sky)·(bg + RON²)`` (electrons).
    """
    positions = list(positions)
    gain = cfg.sensor.gain_e_per_adu
    ron = cfg.read_noise_e

    aper = CircularAperture(positions, r=r_ap)
    annulus = CircularAnnulus(positions, r_in=r_in, r_out=r_out)
    phot = aperture_photometry(image, aper)
    ap_sum_adu = np.asarray(phot["aperture_sum"], dtype=float)

    sky_stats = ApertureStats(image, annulus)
    sky_med_adu = np.atleast_1d(np.asarray(sky_stats.median, dtype=float))

    n_ap = float(aper.area)
    n_sky = float(annulus.area)

    net_adu = ap_sum_adu - sky_med_adu * n_ap
    flux_e = net_adu * gain

    bg_e = np.clip((sky_med_adu - bias_adu) * gain, 0.0, None)
    var = np.clip(flux_e, 0.0, None) + n_ap * (bg_e + ron ** 2) \
        + (n_ap ** 2 / max(n_sky, 1.0)) * (bg_e + ron ** 2)
    sigma_e = np.sqrt(np.clip(var, 0.0, None))
    return flux_e, sigma_e


def measure_pair(image: np.ndarray, o_xy: Tuple[float, float],
                 e_xy: Tuple[float, float], cfg: PolConfig, hwp_deg: float,
                 *, r_ap: float, r_in: float, r_out: float,
                 bias_adu: float = 1000.0) -> BeamFlux:
    """Aperture-photometer one o/e pair → :class:`BeamFlux` (electrons)."""
    flux, sig = measure_fluxes(image, [o_xy, e_xy], cfg, r_ap=r_ap, r_in=r_in,
                               r_out=r_out, bias_adu=bias_adu)
    return BeamFlux(hwp_deg=hwp_deg, f_o=float(flux[0]), f_e=float(flux[1]),
                    sig_o=float(sig[0]), sig_e=float(sig[1]))


def photometer_sequence(
    frames_by_angle: Dict[float, np.ndarray],
    cfg: PolConfig,
    o_positions: Sequence[Tuple[float, float]],
    names: Optional[Sequence[str]] = None,
    *,
    r_ap: float = 6.0, r_in: float = 10.0, r_out: float = 16.0,
    bias_adu: float = 1000.0,
) -> Dict[str, List[BeamFlux]]:
    """Measure each source's o/e fluxes across all HWP angles.

    Parameters
    ----------
    frames_by_angle : dict
        ``{hwp_deg: image_array}`` (physical ADU).
    o_positions : list of (x, y)
        Ordinary-beam positions of each source (e found via the beam offset).
    Returns ``{source_name: [BeamFlux, ...]}`` ordered by HWP angle.
    """
    o_positions = list(o_positions)
    names = list(names) if names is not None else [f"src{i}" for i in range(len(o_positions))]
    if len(names) != len(o_positions):
        raise ValueError(
            f"names ({len(names)}) must match o_positions ({len(o_positions)})"
        )
    if len(set(names)) != len(names):
        # duplicate names would collapse the per-source result dict below
        raise ValueError(f"source names must be unique; got {names}")
    dx, dy = cfg.beam.offset_xy()
    out: Dict[str, List[BeamFlux]] = {nm: [] for nm in names}
    for ang in sorted(frames_by_angle.keys()):
        img = frames_by_angle[ang]
        o_pos = list(o_positions)
        e_pos = [(x + dx, y + dy) for (x, y) in o_pos]
        fo, so = measure_fluxes(img, o_pos, cfg, r_ap=r_ap, r_in=r_in,
                                r_out=r_out, bias_adu=bias_adu)
        fe, se = measure_fluxes(img, e_pos, cfg, r_ap=r_ap, r_in=r_in,
                                r_out=r_out, bias_adu=bias_adu)
        for k, nm in enumerate(names):
            out[nm].append(BeamFlux(hwp_deg=float(ang), f_o=float(fo[k]),
                                    f_e=float(fe[k]), sig_o=float(so[k]),
                                    sig_e=float(se[k])))
    return out
