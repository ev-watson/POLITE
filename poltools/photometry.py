"""
poltools.photometry — Source detection, beam pairing, and aperture photometry.

Each science source appears twice on the detector (ordinary and extraordinary
beams). This module finds sources, pairs the two images using the known Savart
offset, and measures aperture fluxes in electrons.

Uncertainties follow the standard CCD noise model: photon shot noise from the
source and sky, read noise, and uncertainty in the sky estimate (Stockmans
et al., eq. 16 for combined frames).

CMOS-specific handling:

* optional per-pixel read-noise map (captures random telegraph noise tails);
* bad-pixel mask with local interpolation before summing apertures;
* variance scaling when input frames were median-combined (effective frame
  count is reduced by a factor π/2 for Gaussian noise);
* flagging when an aperture peak hits saturation or the linearity limit.

Positions are ``(x, y)`` in pixels; array indexing is ``data[y, x]``.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import photutils
from astropy.convolution import Gaussian2DKernel, interpolate_replace_nans
from astropy.stats import sigma_clipped_stats
from photutils.aperture import (ApertureStats, CircularAnnulus,
                                CircularAperture, aperture_photometry)
from photutils.detection import DAOStarFinder

# Median combination inflates per-pixel variance by π/2 (Kenney & Keeping 1962).
_MEDIAN_VAR_INFLATION = float(np.pi / 2.0)
_BAD_PIXEL_KERNEL = Gaussian2DKernel(x_stddev=1.0)

try:
    photutils.future_column_names = True
except Exception:  # pragma: no cover
    pass

from ._types import BeamFlux, BeamGeometry, PolConfig


def _effective_n(n_combined: int, combine: str = "median") -> float:
    """Effective number of frames for variance scaling after stack combination."""
    n = max(int(n_combined), 1)
    if combine == "mean" or n <= 2:
        return float(n)
    if combine == "median":
        return n / _MEDIAN_VAR_INFLATION
    raise ValueError(f"combine must be 'median' or 'mean'; got {combine!r}")


def _repair_bad_pixels(image: np.ndarray, bad_pixel_mask: np.ndarray) -> np.ndarray:
    """Fill masked pixels by PSF-weighted local interpolation."""
    bad = np.asarray(bad_pixel_mask, dtype=bool)
    if bad.shape != image.shape:
        raise ValueError(
            f"bad_pixel_mask shape {bad.shape} != image shape {image.shape}"
        )
    if not bad.any():
        return np.asarray(image, dtype=float)
    work = np.array(image, dtype=float)
    work[bad] = np.nan
    work = interpolate_replace_nans(work, _BAD_PIXEL_KERNEL)
    if not np.all(np.isfinite(work)):
        work = np.where(np.isfinite(work), work, np.nanmedian(work))
    return work


def aperture_peaks(image: np.ndarray, positions: Sequence[Tuple[float, float]],
                   r_ap: float,
                   bad_pixel_mask: Optional[np.ndarray] = None) -> np.ndarray:
    """Peak pixel value [ADU] inside each circular aperture."""
    positions = list(positions)
    aper = CircularAperture(positions, r=r_ap)
    masks = aper.to_mask(method="center")
    bad = None if bad_pixel_mask is None else np.asarray(bad_pixel_mask, dtype=bool)
    peaks = []
    for m in masks:
        cut = m.cutout(image, fill_value=np.nan)
        if cut is None:
            peaks.append(float("nan"))
            continue
        sel = m.data > 0
        if bad is not None:
            bad_cut = m.cutout(bad, fill_value=False)
            sel = sel & ~np.asarray(bad_cut, dtype=bool)
        vals = cut[sel]
        vals = vals[np.isfinite(vals)]
        peaks.append(float(np.max(vals)) if vals.size else float("nan"))
    return np.asarray(peaks, dtype=float)


def estimate_background(image: np.ndarray, sigma: float = 3.0) -> Tuple[float, float]:
    """Sigma-clipped ``(median, std)`` background [ADU]."""
    _, median, std = sigma_clipped_stats(image, sigma=sigma)
    return float(median), float(std)


def detect_sources(image: np.ndarray, fwhm_px: float,
                   threshold_sigma: float = 5.0):
    """Detect point sources with DAOStarFinder; returns table or ``None``."""
    median, std = estimate_background(image)
    finder = DAOStarFinder(fwhm=fwhm_px, threshold=threshold_sigma * std)
    return finder(image - median)


Region = Tuple[float, float, float, float]


def point_in_regions(x: float, y: float,
                     regions: Optional[Sequence[Region]]) -> bool:
    """True if ``(x, y)`` lies inside any ``(x0, y0, x1, y1)`` rectangle."""
    if not regions:
        return False
    for (x0, y0, x1, y1) in regions:
        if min(x0, x1) <= x <= max(x0, x1) and min(y0, y1) <= y <= max(y0, y1):
            return True
    return False


def pair_oe(detections, beam: BeamGeometry, tol_px: float = 2.0,
            exclude_regions: Optional[Sequence[Region]] = None
            ) -> List[Tuple[Tuple[float, float], Tuple[float, float]]]:
    """Pair detections into (ordinary, extraordinary) positions.

    Uses the known Savart offset from ``beam``. ``exclude_regions`` drops pairs
    whose ordinary or extraordinary centroid falls in a masked area of the
    detector.
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
            xe, ye = float(coords[j, 0]), float(coords[j, 1])
            if (point_in_regions(float(xo), float(yo), exclude_regions)
                    or point_in_regions(xe, ye, exclude_regions)):
                used.add(i)
                used.add(j)
                continue
            pairs.append(((float(xo), float(yo)), (xe, ye)))
            used.add(i)
            used.add(j)
    return pairs


def measure_fluxes(image: np.ndarray, positions: Sequence[Tuple[float, float]],
                   cfg: PolConfig, *, r_ap: float, r_in: float, r_out: float,
                   bias_adu: float = 1000.0,
                   n_combined: int = 1, combine: str = "median",
                   ron_map: Optional[np.ndarray] = None,
                   bad_pixel_mask: Optional[np.ndarray] = None
                   ) -> Tuple[np.ndarray, np.ndarray]:
    """Aperture photometry at given positions. Returns ``(flux_e, sigma_e)``.

    Dark shot noise is not added separately: the annulus background already
    includes dark accumulation.
    """
    positions = list(positions)
    gain = cfg.sensor.gain_e_per_adu
    ron = cfg.read_noise_e

    # Gain and read noise are per-night characterization values, not header state.
    # They are not defaulted anywhere; reduction requires the analyst to supply them.
    if gain is None:
        raise ValueError(
            "measure_fluxes: conversion gain (e-/ADU) is unknown. It is a per-night "
            "characterization value — set cfg.sensor.gain_e_per_adu before photometry."
        )
    if ron is None and ron_map is None:
        raise ValueError(
            "measure_fluxes: read noise (e-) is unknown. Supply cfg.read_noise_e or a "
            "ron_map before photometry — it is a per-night characterization value."
        )

    work = (image if bad_pixel_mask is None
            else _repair_bad_pixels(image, bad_pixel_mask))

    aper = CircularAperture(positions, r=r_ap)
    annulus = CircularAnnulus(positions, r_in=r_in, r_out=r_out)
    phot = aperture_photometry(work, aper)
    ap_sum_adu = np.asarray(phot["aperture_sum"], dtype=float)

    sky_stats = ApertureStats(work, annulus)
    sky_med_adu = np.atleast_1d(np.asarray(sky_stats.median, dtype=float))

    n_ap = float(aper.area)
    n_sky = float(annulus.area)

    net_adu = ap_sum_adu - sky_med_adu * n_ap
    flux_e = net_adu * gain

    if ron_map is not None:
        ron2 = np.asarray(ron_map, dtype=float) ** 2
        if ron2.shape != np.asarray(image).shape:
            raise ValueError(
                f"ron_map shape {ron2.shape} != image shape {np.asarray(image).shape}"
            )
        ron2_ap = np.asarray(aperture_photometry(ron2, aper)["aperture_sum"],
                             dtype=float)
        ron2_sky = np.atleast_1d(np.asarray(ApertureStats(ron2, annulus).mean,
                                            dtype=float))
    else:
        ron2_ap = n_ap * ron ** 2
        ron2_sky = ron ** 2

    bg_e = np.clip((sky_med_adu - bias_adu) * gain, 0.0, None)
    var = (np.clip(flux_e, 0.0, None)
           + (n_ap * bg_e + ron2_ap)
           + (n_ap ** 2 / max(n_sky, 1.0)) * (bg_e + ron2_sky))
    var = var / _effective_n(n_combined, combine)
    sigma_e = np.sqrt(np.clip(var, 0.0, None))
    return flux_e, sigma_e


def _aperture_saturated(image: np.ndarray,
                        positions: Sequence[Tuple[float, float]],
                        cfg: PolConfig, r_ap: float, bias_adu: float,
                        bad_pixel_mask: Optional[np.ndarray] = None
                        ) -> np.ndarray:
    peaks_adu = aperture_peaks(image, positions, r_ap, bad_pixel_mask=bad_pixel_mask)
    peaks_e = (peaks_adu - bias_adu) * cfg.sensor.gain_e_per_adu
    return np.asarray(peaks_e >= cfg.sat_limit_e(), dtype=bool)


def measure_pair(image: np.ndarray, o_xy: Tuple[float, float],
                 e_xy: Tuple[float, float], cfg: PolConfig, hwp_deg: float,
                 *, r_ap: float, r_in: float, r_out: float,
                 bias_adu: float = 1000.0,
                 n_combined: int = 1, combine: str = "median",
                 ron_map: Optional[np.ndarray] = None,
                 bad_pixel_mask: Optional[np.ndarray] = None) -> BeamFlux:
    """Aperture-photometer one ordinary/extraordinary pair → :class:`BeamFlux`."""
    flux, sig = measure_fluxes(image, [o_xy, e_xy], cfg, r_ap=r_ap, r_in=r_in,
                               r_out=r_out, bias_adu=bias_adu,
                               n_combined=n_combined, combine=combine,
                               ron_map=ron_map, bad_pixel_mask=bad_pixel_mask)
    sat = _aperture_saturated(image, [o_xy, e_xy], cfg, r_ap, bias_adu,
                              bad_pixel_mask=bad_pixel_mask)
    return BeamFlux(hwp_deg=hwp_deg, f_o=float(flux[0]), f_e=float(flux[1]),
                    sig_o=float(sig[0]), sig_e=float(sig[1]),
                    saturated=bool(sat.any()))


def photometer_sequence(
    frames_by_angle: Dict[float, np.ndarray],
    cfg: PolConfig,
    o_positions: Sequence[Tuple[float, float]],
    names: Optional[Sequence[str]] = None,
    *,
    r_ap: float = 6.0, r_in: float = 10.0, r_out: float = 16.0,
    bias_adu: float = 1000.0,
    n_by_angle: Optional[Dict[float, int]] = None,
    combine: str = "median",
    ron_map: Optional[np.ndarray] = None,
    bad_pixel_mask: Optional[np.ndarray] = None,
) -> Dict[str, List[BeamFlux]]:
    """Measure each source's ordinary and extraordinary fluxes at every HWP angle.

    Returns ``{source_name: [BeamFlux, ...]}`` ordered by half-wave plate angle.
    """
    o_positions = list(o_positions)
    names = list(names) if names is not None else [f"src{i}" for i in range(len(o_positions))]
    if len(names) != len(o_positions):
        raise ValueError(
            f"names ({len(names)}) must match o_positions ({len(o_positions)})"
        )
    if len(set(names)) != len(names):
        raise ValueError(f"source names must be unique; got {names}")
    dx, dy = cfg.beam.offset_xy()
    out: Dict[str, List[BeamFlux]] = {nm: [] for nm in names}
    for ang in sorted(frames_by_angle.keys()):
        img = frames_by_angle[ang]
        n_comb = 1 if n_by_angle is None else int(n_by_angle.get(ang, 1))
        o_pos = list(o_positions)
        e_pos = [(x + dx, y + dy) for (x, y) in o_pos]
        fo, so = measure_fluxes(img, o_pos, cfg, r_ap=r_ap, r_in=r_in,
                                r_out=r_out, bias_adu=bias_adu,
                                n_combined=n_comb, combine=combine,
                                ron_map=ron_map, bad_pixel_mask=bad_pixel_mask)
        fe, se = measure_fluxes(img, e_pos, cfg, r_ap=r_ap, r_in=r_in,
                                r_out=r_out, bias_adu=bias_adu,
                                n_combined=n_comb, combine=combine,
                                ron_map=ron_map, bad_pixel_mask=bad_pixel_mask)
        sat_o = _aperture_saturated(img, o_pos, cfg, r_ap, bias_adu,
                                    bad_pixel_mask=bad_pixel_mask)
        sat_e = _aperture_saturated(img, e_pos, cfg, r_ap, bias_adu,
                                    bad_pixel_mask=bad_pixel_mask)
        for k, nm in enumerate(names):
            out[nm].append(BeamFlux(hwp_deg=float(ang), f_o=float(fo[k]),
                                    f_e=float(fe[k]), sig_o=float(so[k]),
                                    sig_e=float(se[k]),
                                    saturated=bool(sat_o[k] or sat_e[k])))
    return out
