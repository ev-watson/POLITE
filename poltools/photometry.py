"""
poltools.photometry — Source detection, o/e pairing, aperture photometry.

Uses the Astropy-affiliated ``photutils`` (per CLAUDE.md "prefer Astropy/AAS
conventions"): ``DAOStarFinder`` for detection and
``CircularAperture``/``CircularAnnulus``/``ApertureStats`` for concentric-
aperture photometry with a sky annulus — the standard DBIP/SOLVEPOL operation
(Source B). Photometric uncertainties use the CCD equation (source shot noise +
sky + read noise + sky-estimation error).

CMOS-detector handling (QHY268M / IMX571; ``reduction.md``, Sources A/B):
* the read-noise term accepts a **per-pixel read-noise map** (RTN/hot-pixel
  tail), not just the scalar Gaussian core (finding C-1);
* a **bad/hot-pixel mask** can be supplied — flagged pixels are repaired by
  local interpolation (Astropy) before summing so they neither inflate the
  aperture nor bias the sky median (finding C-2);
* the reported σ is **suppressed by the effective number of combined frames**
  (Stockmans et al. eq. 16, Source B: "...suppressed by √N for N exposures to
  average over"); ``N`` for a mean, ``N/(π/2)`` for a median (finding C-3);
* aperture peaks are checked against the saturation/linearity limit (C-6).
Dark shot noise needs no separate term: the background is measured empirically
from the annulus, whose pixels accumulate dark too, so ``bg_e`` already includes
it (an explicit ``n_ap·D·t`` term would double-count — finding C-4).

Pixel convention: photutils positions are ``(x, y)`` with ``data[y, x]``,
identical to the rest of poltools (origin upper-left is a display choice only).
Fluxes are returned in **electrons**.
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

# Variance inflation of a per-pixel median vs a mean for Gaussian noise: in the
# large-N limit Var(median) -> (π/2)·σ²/N (Kenney & Keeping 1962), while the
# mean attains σ²/N. For N <= 2 the median equals the mean (and N=1 is the frame
# itself), so no penalty applies there. Used to turn a frame count into an
# effective N for the σ suppression of finding C-3 (Stockmans et al., Source B).
_MEDIAN_VAR_INFLATION = float(np.pi / 2.0)

# Compact Gaussian kernel for repairing isolated bad/hot pixels by PSF-weighted
# local interpolation (finding C-2). Small enough that a flagged pixel is filled
# from its immediate neighbourhood (≈ local PSF/sky value).
_BAD_PIXEL_KERNEL = Gaussian2DKernel(x_stddev=1.0)


def _effective_n(n_combined: int, combine: str = "median") -> float:
    """Effective number of independent frames for variance scaling (C-3).

    Combining ``N`` equal-exposure frames suppresses the per-pixel (hence
    aperture-summed) variance by ``N`` for a mean and by ``N/(π/2)`` for a
    median (Stockmans et al. eq. 16, Source B). ``N <= 2`` is exact for the
    median (median == frame for N=1, median == mean for N=2), so no median
    penalty is applied there.
    """
    n = max(int(n_combined), 1)
    if combine == "mean" or n <= 2:
        return float(n)
    if combine == "median":
        return n / _MEDIAN_VAR_INFLATION
    raise ValueError(f"combine must be 'median' or 'mean'; got {combine!r}")


def _repair_bad_pixels(image: np.ndarray, bad_pixel_mask: np.ndarray) -> np.ndarray:
    """Return a copy of ``image`` with masked pixels filled by interpolation.

    Flagged pixels are set to NaN and replaced with a PSF-weighted local
    estimate via ``astropy.convolution.interpolate_replace_nans`` (finding C-2);
    any pixel with no finite neighbours falls back to the global median.
    """
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
    """Peak pixel value (ADU) inside each circular aperture (saturation check).

    Masked pixels are ignored (a flagged hot pixel is a defect, not saturation).
    """
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


Region = Tuple[float, float, float, float]


def point_in_regions(x: float, y: float,
                     regions: Optional[Sequence[Region]]) -> bool:
    """True if ``(x, y)`` lies inside any ``(x0, y0, x1, y1)`` rectangle.

    Rectangles are inclusive and corner-order-independent (min/max normalized);
    coordinates are detector ``(x, y)`` with origin upper-left. Used to drop
    sources landing on a known bad FOV region — e.g. the α-BBO Savart's chipped
    corner, once it is characterized from real flats (see
    :func:`reduce_to_stokes`'s ``exclude_regions``). ``None`` → never excluded.
    """
    if not regions:
        return False
    for (x0, y0, x1, y1) in regions:
        if min(x0, x1) <= x <= max(x0, x1) and min(y0, y1) <= y <= max(y0, y1):
            return True
    return False


def pair_oe(detections, beam: BeamGeometry, tol_px: float = 2.0,
            exclude_regions: Optional[Sequence[Region]] = None
            ) -> List[Tuple[Tuple[float, float], Tuple[float, float]]]:
    """Pair detections into (o, e) using the known beam offset vector.

    For each detection treated as an ordinary beam, look for a partner near
    ``o + offset``. Returns a list of ``((x_o, y_o), (x_e, y_e))`` pairs.

    ``exclude_regions`` (list of ``(x0,y0,x1,y1)`` rectangles) drops any pair
    whose o- or e-centroid falls inside a bad FOV region (e.g. the Savart chipped
    corner). ``offset_xy`` comes from ``beam``, which is the **active filter's**
    geometry (the α-BBO split is dispersive — pass ``cfg.for_filter(name).beam``).
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
    """Aperture photometry at given positions. Returns (flux_e, sigma_e).

    Net flux = aperture sum − (annulus-median sky) × aperture area, converted to
    electrons. Uncertainty via the CCD equation
    ``σ² = F_src + (n_ap·bg + Σ_ap RON²) + (n_ap²/n_sky)·(bg + ⟨RON²⟩)``
    (electrons), divided by the effective number of combined frames.

    Parameters
    ----------
    n_combined : int
        How many equal-exposure frames were combined into ``image`` (per HWP
        angle). The variance is suppressed by the effective N (finding C-3;
        Stockmans et al. eq. 16, Source B). Default 1 → single-exposure σ.
    combine : {"median", "mean"}
        Combination used to build ``image`` — sets the effective-N penalty
        (median: ``N/(π/2)``; mean: ``N``).
    ron_map : ndarray, optional
        Per-pixel read noise (**electrons**), same shape as ``image`` (finding
        C-1). When given, ``Σ_ap RON_i²`` is summed over aperture pixels and
        ``⟨RON²⟩`` averaged over the annulus, instead of the scalar
        ``cfg.read_noise_e``.
    bad_pixel_mask : ndarray of bool, optional
        True where a pixel is bad/hot; repaired by local interpolation before
        photometry (finding C-2).
    """
    positions = list(positions)
    gain = cfg.sensor.gain_e_per_adu
    ron = cfg.read_noise_e

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

    # read-noise variance (electrons²): scalar core, or the per-pixel map summed
    # over the aperture (Σ RON²) and averaged over the annulus (⟨RON²⟩).
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
    """Boolean per-position flag: aperture peak ≥ saturation/linearity limit (C-6)."""
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
    """Aperture-photometer one o/e pair → :class:`BeamFlux` (electrons)."""
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
    """Measure each source's o/e fluxes across all HWP angles.

    Parameters
    ----------
    frames_by_angle : dict
        ``{hwp_deg: image_array}`` (physical ADU).
    o_positions : list of (x, y)
        Ordinary-beam positions of each source (e found via the beam offset).
    n_by_angle : dict, optional
        ``{hwp_deg: n_frames_combined}`` for the per-angle σ suppression (finding
        C-3). Missing/absent angles default to 1.
    combine : {"median", "mean"}
        Combination used per angle (sets the effective-N penalty).
    ron_map, bad_pixel_mask : ndarray, optional
        Per-pixel read-noise map (C-1) and bad-pixel mask (C-2), threaded to
        :func:`measure_fluxes`.

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
