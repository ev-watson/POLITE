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

from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import photutils
from astropy.convolution import Gaussian2DKernel, interpolate_replace_nans
from astropy.stats import sigma_clipped_stats
from astropy.table import Table
from photutils.aperture import (ApertureStats, CircularAnnulus,
                                CircularAperture, aperture_photometry)
from photutils.detection import DAOStarFinder
from scipy.ndimage import gaussian_filter, maximum_filter

# Median combination inflates per-pixel variance by π/2 (Kenney & Keeping 1962).
_MEDIAN_VAR_INFLATION = float(np.pi / 2.0)
_BAD_PIXEL_KERNEL = Gaussian2DKernel(x_stddev=1.0)

try:
    photutils.future_column_names = True
except Exception:  # pragma: no cover
    pass

from ._types import (BeamFlux, BeamGeometry, PolConfig,
                     nominal_beam_separation_px)


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


def brightest_point_source(
    image: np.ndarray,
    *,
    source_scale_px: float = 25.0,
    centroid_radius_px: Optional[float] = None,
    threshold_sigma: float = 5.0,
    bad_pixel_mask: Optional[np.ndarray] = None,
    iterations: int = 4,
) -> Tuple[float, float]:
    """Return the centroid of the brightest compact source candidate.

    This is a quick anchor finder, not a replacement for a reviewed source
    catalogue.  It ranks sources by a Gaussian-smoothed, background-subtracted
    image, rather than by one raw peak pixel.  Consequently an isolated hot or
    saturated pixel is strongly suppressed, while a resolved or defocused star
    still contributes its local integrated flux.

    ``source_scale_px`` is the approximate radius of the stellar image, not a
    measured FWHM and not a DAOStarFinder kernel.  The default is suitable for
    POLITE's deliberately defocused acquisition frames.  Set it larger only
    after confirming that its centroid region cannot include the other Savart
    beam.  ``bad_pixel_mask`` excludes known bad pixels from both the
    background estimate and centroid.

    Raises
    ------
    ValueError
        If the input is not a 2-D image, the mask shape differs, or no source
        is significant at the requested threshold.
    """
    arr = np.asarray(image, dtype=float)
    if arr.ndim != 2:
        raise ValueError("image must be a 2-D array")
    if source_scale_px <= 0:
        raise ValueError("source_scale_px must be positive")

    invalid = ~np.isfinite(arr)
    if bad_pixel_mask is not None:
        bad = np.asarray(bad_pixel_mask, dtype=bool)
        if bad.shape != arr.shape:
            raise ValueError("bad_pixel_mask shape must match image")
        invalid |= bad
    if invalid.all():
        raise ValueError("image has no finite, unmasked pixels")

    _mean, background, noise = sigma_clipped_stats(
        arr, mask=invalid, sigma=3.0, maxiters=5
    )
    background, noise = float(background), float(noise)
    if not np.isfinite(noise) or noise <= 0:
        raise ValueError("could not estimate a positive background noise")

    # A background-filled working image keeps defects from becoming maxima and
    # lets Gaussian smoothing rank local integrated source flux.
    work = np.where(invalid, background, arr)
    # Retain negative fluctuations here so the clipped standard deviation below
    # is the noise of the *smoothed* field, in the same units as ``smooth``.
    # Clipping before convolution would give it a positive pedestal and compare
    # a smoothed signal against the (much larger) per-pixel noise.
    smooth = gaussian_filter(
        work - background, sigma=float(source_scale_px) / 2.0
    )
    _mean, _median, smooth_noise = sigma_clipped_stats(
        smooth, sigma=3.0, maxiters=5
    )
    smooth_noise = float(smooth_noise)
    if not np.isfinite(smooth_noise) or smooth_noise <= 0:
        raise ValueError("could not estimate a positive smoothed-image noise")

    radius = (2.0 * source_scale_px if centroid_radius_px is None
              else float(centroid_radius_px))
    if radius <= 0:
        raise ValueError("centroid_radius_px must be positive")
    margin = int(np.ceil(radius))
    if arr.shape[0] <= 2 * margin or arr.shape[1] <= 2 * margin:
        raise ValueError("image is too small for the requested centroid radius")
    smooth[:margin, :] = -np.inf
    smooth[-margin:, :] = -np.inf
    smooth[:, :margin] = -np.inf
    smooth[:, -margin:] = -np.inf
    y, x = np.unravel_index(int(np.argmax(smooth)), smooth.shape)
    if smooth[y, x] < threshold_sigma * smooth_noise:
        raise ValueError(
            f"no source reaches {threshold_sigma:g}-sigma above background"
        )

    cx, cy = float(x), float(y)
    for _ in range(max(1, int(iterations))):
        x0 = max(0, int(np.floor(cx - radius)))
        x1 = min(arr.shape[1], int(np.ceil(cx + radius)) + 1)
        y0 = max(0, int(np.floor(cy - radius)))
        y1 = min(arr.shape[0], int(np.ceil(cy + radius)) + 1)
        yy, xx = np.mgrid[y0:y1, x0:x1]
        inside = (xx - cx) ** 2 + (yy - cy) ** 2 <= radius ** 2
        weights = np.where(
            inside & ~invalid[y0:y1, x0:x1],
            np.clip(arr[y0:y1, x0:x1] - background, 0.0, None),
            0.0,
        )
        total = float(weights.sum())
        if total <= 0:
            raise ValueError("no usable source flux in centroid region")
        next_x = float((xx * weights).sum() / total)
        next_y = float((yy * weights).sum() / total)
        if max(abs(next_x - cx), abs(next_y - cy)) < 0.01:
            cx, cy = next_x, next_y
            break
        cx, cy = next_x, next_y
    return cx, cy


@dataclass(frozen=True)
class AnchorPairProposal:
    """Brightest valid dual-beam pair proposed from one image.

    The o/e order follows a deterministic detector convention: the measured
    split is canonicalized to positive detector y (then positive x).  It is a
    working reduction convention, not a physical ordinary-ray identification.
    """

    beam_a_xy: Tuple[float, float]
    beam_b_xy: Tuple[float, float]
    separation_px: float
    axis_angle_deg: float
    supporting_pair_count: int
    offset_rms_px: float
    pair_score: float
    candidate_count: int
    matched_pairs: Tuple[
        Tuple[Tuple[float, float], Tuple[float, float]], ...
    ]

    @property
    def beam_geometry(self) -> BeamGeometry:
        """Canonical directed geometry matching :meth:`as_pair_anchors`."""
        dx = self.beam_b_xy[0] - self.beam_a_xy[0]
        dy = self.beam_b_xy[1] - self.beam_a_xy[1]
        return BeamGeometry(
            separation_px=float(np.hypot(dx, dy)),
            position_angle_deg=float(np.degrees(np.arctan2(dx, dy)) % 360.0),
        )

    def as_pair_anchors(self) -> Dict[str, Tuple[float, float]]:
        """Return canonical detector A/B coordinates, not physical ray labels."""
        return {"a": self.beam_a_xy, "b": self.beam_b_xy}

    def overlay(self, ax, *, majorcolor="tab:blue", minorcolor="tab:orange",
                markersize=12.):
        """Draw all matched pairs, highlighting the default brightest pair."""
        shrink = markersize / 2

        for first, second in self.matched_pairs:
            ax.annotate("", second, first, arrowprops=dict(
                arrowstyle="-", color=minorcolor, linewidth=0.8,
                shrinkA=shrink, shrinkB=shrink,
            ))
            ax.plot(*zip(first, second), linestyle="none", marker="o",
                    color=minorcolor, markerfacecolor="none",
                    markersize=markersize)

        ax.annotate("", self.beam_b_xy, self.beam_a_xy, arrowprops=dict(
            arrowstyle="-", color=majorcolor, linewidth=1.5,
            shrinkA=shrink, shrinkB=shrink,
        ))
        ax.plot(*zip(self.beam_a_xy, self.beam_b_xy), linestyle="none",
                marker="o", color=majorcolor, markerfacecolor="none",
                markersize=markersize)

        ax.annotate("a", self.beam_a_xy, color=majorcolor,
                    xytext=(8, -8), textcoords="offset points")
        ax.annotate("b", self.beam_b_xy, color=majorcolor,
                    xytext=(8, -8), textcoords="offset points")
        return ax


@dataclass(frozen=True)
class _SourceCandidate:
    xy: Tuple[float, float]
    score: float
    source_scale_px: float


_AUTO_SOURCE_SCALES_PX = (8.0, 16.0, 25.0, 40.0)
_MAX_CANDIDATES_PER_SCALE = 64


def _prepare_anchor_image(
    image: np.ndarray,
    bad_pixel_mask: Optional[np.ndarray],
) -> Tuple[np.ndarray, np.ndarray, float]:
    """Validate an anchor image and return ``(work, invalid, background)``."""
    arr = np.asarray(image, dtype=float)
    if arr.ndim != 2:
        raise ValueError("image must be a 2-D array")
    invalid = ~np.isfinite(arr)
    if bad_pixel_mask is not None:
        bad = np.asarray(bad_pixel_mask, dtype=bool)
        if bad.shape != arr.shape:
            raise ValueError("bad_pixel_mask shape must match image")
        invalid |= bad
    if invalid.all():
        raise ValueError("image has no finite, unmasked pixels")
    _mean, background, noise = sigma_clipped_stats(
        arr, mask=invalid, sigma=3.0, maxiters=5
    )
    if not np.isfinite(background) or not np.isfinite(noise) or noise <= 0:
        raise ValueError("could not estimate a positive background noise")
    return np.where(invalid, float(background), arr), invalid, float(background)


def _centroid_candidate(
    work: np.ndarray,
    invalid: np.ndarray,
    background: float,
    xy: Tuple[float, float],
    radius: float,
    iterations: int = 4,
) -> Tuple[float, float]:
    """Iteratively centre a candidate using only unmasked positive flux."""
    cx, cy = map(float, xy)
    for _ in range(max(1, int(iterations))):
        x0 = max(0, int(np.floor(cx - radius)))
        x1 = min(work.shape[1], int(np.ceil(cx + radius)) + 1)
        y0 = max(0, int(np.floor(cy - radius)))
        y1 = min(work.shape[0], int(np.ceil(cy + radius)) + 1)
        yy, xx = np.mgrid[y0:y1, x0:x1]
        inside = (xx - cx) ** 2 + (yy - cy) ** 2 <= radius ** 2
        weights = np.where(
            inside & ~invalid[y0:y1, x0:x1],
            np.clip(work[y0:y1, x0:x1] - background, 0.0, None),
            0.0,
        )
        total = float(weights.sum())
        if total <= 0:
            raise ValueError("no usable source flux in centroid region")
        next_x = float((xx * weights).sum() / total)
        next_y = float((yy * weights).sum() / total)
        if max(abs(next_x - cx), abs(next_y - cy)) < 0.01:
            return next_x, next_y
        cx, cy = next_x, next_y
    return cx, cy


def _multiscale_candidates(
    work: np.ndarray,
    invalid: np.ndarray,
    background: float,
    *,
    source_scales_px: Sequence[float],
    threshold_sigma: float,
) -> List[_SourceCandidate]:
    """Find and de-duplicate compact-source candidates without an FWHM input."""
    raw: List[_SourceCandidate] = []
    residual = work - background
    for scale in source_scales_px:
        scale = float(scale)
        if scale <= 0:
            raise ValueError("source_scales_px values must be positive")
        smooth = gaussian_filter(residual, sigma=scale / 2.0)
        _mean, _median, smooth_noise = sigma_clipped_stats(
            smooth, sigma=3.0, maxiters=5
        )
        smooth_noise = float(smooth_noise)
        if not np.isfinite(smooth_noise) or smooth_noise <= 0:
            continue
        spacing = max(3, int(np.ceil(scale)))
        maxima = smooth == maximum_filter(smooth, size=spacing, mode="nearest")
        ys, xs = np.nonzero(maxima & (smooth >= threshold_sigma * smooth_noise))
        if len(xs) > _MAX_CANDIDATES_PER_SCALE:
            keep = np.argpartition(
                smooth[ys, xs], -_MAX_CANDIDATES_PER_SCALE
            )[-_MAX_CANDIDATES_PER_SCALE:]
            ys, xs = ys[keep], xs[keep]
        for y, x in zip(ys, xs):
            if x < 2 * scale or y < 2 * scale:
                continue
            if x >= work.shape[1] - 2 * scale or y >= work.shape[0] - 2 * scale:
                continue
            try:
                xy = _centroid_candidate(
                    work, invalid, background, (float(x), float(y)), 2.0 * scale
                )
            except ValueError:
                continue
            raw.append(_SourceCandidate(xy, float(smooth[y, x]), scale))

    selected: List[_SourceCandidate] = []
    for candidate in sorted(raw, key=lambda c: c.score, reverse=True):
        if any(
            np.hypot(candidate.xy[0] - prior.xy[0], candidate.xy[1] - prior.xy[1])
            < max(candidate.source_scale_px, prior.source_scale_px)
            for prior in selected
        ):
            continue
        selected.append(candidate)
    return selected


def _canonical_axis_offset(dx: float, dy: float) -> Tuple[float, float]:
    """Choose one sign for an unoriented detector axis (positive y first)."""
    if dy < 0 or (dy == 0 and dx < 0):
        return -dx, -dy
    return dx, dy


def _infer_split_axis(
    candidates: Sequence[_SourceCandidate],
    *,
    nominal_separation_px: float,
    separation_tolerance_frac: float,
    cluster_tolerance_px: float,
    min_supporting_pairs: int,
) -> Tuple[float, float, int, float]:
    """Measure a recurring Savart displacement axis from candidate offsets."""
    if not 0 < separation_tolerance_frac < 1:
        raise ValueError("separation_tolerance_frac must lie between 0 and 1")
    offsets: List[Tuple[float, float]] = []
    lo = nominal_separation_px * (1.0 - separation_tolerance_frac)
    hi = nominal_separation_px * (1.0 + separation_tolerance_frac)
    for i, first in enumerate(candidates):
        for second in candidates[i + 1:]:
            dx = second.xy[0] - first.xy[0]
            dy = second.xy[1] - first.xy[1]
            if lo <= np.hypot(dx, dy) <= hi:
                offsets.append(_canonical_axis_offset(dx, dy))
    if not offsets:
        raise ValueError("no candidate offsets are consistent with the Savart separation")

    clusters: List[List[Tuple[float, float]]] = []
    for offset in offsets:
        for cluster in clusters:
            centre = np.mean(cluster, axis=0)
            if np.hypot(offset[0] - centre[0], offset[1] - centre[1]) <= cluster_tolerance_px:
                cluster.append(offset)
                break
        else:
            clusters.append([offset])
    best = max(clusters, key=len)
    if len(best) < min_supporting_pairs:
        raise ValueError(
            "Savart split is not supported by enough independent candidate pairs"
        )
    centre = np.mean(best, axis=0)
    residuals = [np.hypot(dx - centre[0], dy - centre[1]) for dx, dy in best]
    rms = float(np.sqrt(np.mean(np.square(residuals))))
    return float(centre[0]), float(centre[1]), len(best), rms


def propose_anchor_pair(
    image: np.ndarray,
    *,
    bad_pixel_mask: Optional[np.ndarray] = None,
    source_scales_px: Sequence[float] = _AUTO_SOURCE_SCALES_PX,
    threshold_sigma: float = 5.0,
    separation_tolerance_frac: float = 0.10,
    cluster_tolerance_px: float = 6.0,
    min_supporting_pairs: int = 2,
) -> AnchorPairProposal:
    """Propose the brightest unlabelled Savart pair in a corrected image.

    The helper searches several source scales internally, measures a common
    *unoriented* split axis from multiple candidate pairs, then calls
    :func:`pair_oe` for final matching.  It requires no supplied FWHM.  The
    manufacturer separation is a search-range sanity check only; the returned
    separation and axis are measured from this image.

    The A/B order is a deterministic detector convention, not a physical
    ordinary/extraordinary identification.  Its coordinates can be used
    directly through :meth:`AnchorPairProposal.as_pair_anchors`.
    """
    if threshold_sigma <= 0:
        raise ValueError("threshold_sigma must be positive")
    if cluster_tolerance_px <= 0:
        raise ValueError("cluster_tolerance_px must be positive")
    if min_supporting_pairs < 2:
        raise ValueError("min_supporting_pairs must be at least 2")

    work, invalid, background = _prepare_anchor_image(image, bad_pixel_mask)
    candidates = _multiscale_candidates(
        work, invalid, background,
        source_scales_px=source_scales_px,
        threshold_sigma=threshold_sigma,
    )
    if len(candidates) < 4:
        raise ValueError("fewer than four viable source candidates were found")

    dx, dy, support, rms = _infer_split_axis(
        candidates,
        nominal_separation_px=nominal_beam_separation_px(),
        separation_tolerance_frac=separation_tolerance_frac,
        cluster_tolerance_px=cluster_tolerance_px,
        min_supporting_pairs=min_supporting_pairs,
    )
    separation = float(np.hypot(dx, dy))
    axis_angle = float(np.degrees(np.arctan2(dx, dy)) % 180.0)
    canonical_geometry = BeamGeometry(
        separation_px=separation,
        position_angle_deg=float(np.degrees(np.arctan2(dx, dy)) % 360.0),
    )
    detections = Table({
        "xcentroid": [c.xy[0] for c in candidates],
        "ycentroid": [c.xy[1] for c in candidates],
    })
    pairs = pair_oe(detections, canonical_geometry, tol_px=cluster_tolerance_px)
    if not pairs:
        raise ValueError("inferred Savart axis did not pair any source candidates")

    def candidate_score(xy: Tuple[float, float]) -> float:
        return max(
            c.score for c in candidates
            if np.hypot(c.xy[0] - xy[0], c.xy[1] - xy[1]) <= cluster_tolerance_px
        )

    first_xy, second_xy = max(
        pairs, key=lambda pair: candidate_score(pair[0]) + candidate_score(pair[1])
    )
    pair_dx, pair_dy = second_xy[0] - first_xy[0], second_xy[1] - first_xy[1]
    if _canonical_axis_offset(pair_dx, pair_dy) != (pair_dx, pair_dy):
        first_xy, second_xy = second_xy, first_xy
    return AnchorPairProposal(
        beam_a_xy=first_xy,
        beam_b_xy=second_xy,
        separation_px=separation,
        axis_angle_deg=axis_angle,
        supporting_pair_count=support,
        offset_rms_px=rms,
        pair_score=float(candidate_score(first_xy) + candidate_score(second_xy)),
        candidate_count=len(candidates),
        matched_pairs=tuple(pairs),
    )


def track_matched_pair(
    previous_pair: Tuple[Tuple[float, float], Tuple[float, float]],
    previous_matches: Sequence[Tuple[Tuple[float, float], Tuple[float, float]]],
    current_matches: Sequence[Tuple[Tuple[float, float], Tuple[float, float]]],
    *,
    translation_tolerance_px: float = 12.0,
    match_tolerance_px: float = 15.0,
    min_translation_support: int = 2,
) -> Tuple[
    Tuple[Tuple[float, float], Tuple[float, float]],
    Tuple[float, float],
    float,
]:
    """Continue one canonical A/B pair from one frame into the next.

    All detected Savart-pair midpoints vote for detector translations between
    adjacent frames.  A translation must have independent pair support *and*
    predict a current match for ``previous_pair``; the closest such match is
    returned.  This prevents an unrelated, more-populated source pattern from
    overriding the requested pair.  It also lets a pair move by hundreds of
    pixels between exposures without enlarging a local centroid window enough
    to confuse the two beams.

    The returned tuple is ``(pair, translation_xy, prediction_rms_px)``.  The
    input and output pair order is the canonical A/B detector convention used
    by :func:`propose_anchor_pair`, not a physical o/e identification.  The
    function is deliberately fail-closed when the other pairs do not support
    one translation, or when no candidate matches the predicted pair closely.
    """
    if translation_tolerance_px <= 0:
        raise ValueError("translation_tolerance_px must be positive")
    if match_tolerance_px <= 0:
        raise ValueError("match_tolerance_px must be positive")
    if min_translation_support < 2:
        raise ValueError("min_translation_support must be at least 2")

    def as_pair_array(pair, label: str) -> np.ndarray:
        array = np.asarray(pair, dtype=float)
        if array.shape != (2, 2) or not np.all(np.isfinite(array)):
            raise ValueError(f"{label} must contain two finite (x, y) coordinates")
        return array

    prior = as_pair_array(previous_pair, "previous_pair")
    previous = [
        as_pair_array(pair, "previous_matches entry") for pair in previous_matches
    ]
    current = [
        as_pair_array(pair, "current_matches entry") for pair in current_matches
    ]
    if len(previous) < min_translation_support:
        raise ValueError("previous_matches lacks enough pairs to infer a translation")
    if not current:
        raise ValueError("current_matches is empty")

    shifts = [
        current_pair.mean(axis=0) - prior_match.mean(axis=0)
        for prior_match in previous
        for current_pair in current
    ]
    clusters: List[List[np.ndarray]] = []
    for shift in shifts:
        for cluster in clusters:
            centre = np.mean(cluster, axis=0)
            if np.hypot(*(shift - centre)) <= translation_tolerance_px:
                cluster.append(shift)
                break
        else:
            clusters.append([shift])

    supported_clusters = [
        cluster for cluster in clusters if len(cluster) >= min_translation_support
    ]
    if not supported_clusters:
        raise ValueError("matched pairs do not support a common translation")

    continuations = []
    for cluster in supported_clusters:
        translation = np.mean(cluster, axis=0)
        prediction = prior + translation
        residuals = [
            float(np.sqrt(np.mean(np.sum((candidate - prediction) ** 2, axis=1))))
            for candidate in current
        ]
        selected_index = int(np.argmin(residuals))
        residual = residuals[selected_index]
        if residual <= match_tolerance_px:
            continuations.append(
                (residual, len(cluster), translation, selected_index)
            )
    if not continuations:
        modal_cluster = max(supported_clusters, key=len)
        modal_translation = np.mean(modal_cluster, axis=0)
        modal_prediction = prior + modal_translation
        modal_residual = min(
            float(np.sqrt(np.mean(np.sum((candidate - modal_prediction) ** 2, axis=1))))
            for candidate in current
        )
        raise ValueError(
            "no supported translation continues the selected pair "
            f"(closest RMS {modal_residual:.1f} px)"
        )

    # A close continuation is more important than a slightly more-populated
    # but unrelated translation cluster.  Support breaks near-equal matches.
    continuations.sort(key=lambda result: (result[0], -result[1]))
    residual, _support, translation, selected_index = continuations[0]
    selected = current[selected_index]
    return (
        (
            (float(selected[0, 0]), float(selected[0, 1])),
            (float(selected[1, 0]), float(selected[1, 1])),
        ),
        (float(translation[0]), float(translation[1])),
        residual,
    )


def select_trackable_pair(
    previous_matches: Sequence[Tuple[Tuple[float, float], Tuple[float, float]]],
    current_matches: Sequence[Tuple[Tuple[float, float], Tuple[float, float]]],
    preferred_current_pair: Tuple[Tuple[float, float], Tuple[float, float]],
    *,
    translation_tolerance_px: float = 12.0,
    match_tolerance_px: float = 15.0,
    min_translation_support: int = 2,
) -> Tuple[
    Tuple[Tuple[float, float], Tuple[float, float]],
    Tuple[Tuple[float, float], Tuple[float, float]],
    Tuple[float, float],
    float,
]:
    """Choose an initial pair that can be continued into the next frame.

    ``preferred_current_pair`` is normally the next frame's default bright
    proposal.  Each pair from the previous frame is tried as a predecessor;
    only candidates with a supported continuation are retained, and the one
    whose continuation is closest to the preferred pair wins.  This avoids
    starting a sequence on a bright pair that immediately leaves the detector.

    Returns ``(previous_pair, current_pair, translation_xy,
    prediction_rms_px)``.  A pair that cannot be continued is not silently
    substituted; if no predecessor works, the function raises ``ValueError``.
    """
    preferred = np.asarray(preferred_current_pair, dtype=float)
    if preferred.shape != (2, 2) or not np.all(np.isfinite(preferred)):
        raise ValueError(
            "preferred_current_pair must contain two finite (x, y) coordinates"
        )

    candidates = []
    for previous_pair in previous_matches:
        try:
            current_pair, translation, prediction_rms = track_matched_pair(
                previous_pair, previous_matches, current_matches,
                translation_tolerance_px=translation_tolerance_px,
                match_tolerance_px=match_tolerance_px,
                min_translation_support=min_translation_support,
            )
        except ValueError:
            continue
        current = np.asarray(current_pair, dtype=float)
        preference_rms = float(
            np.sqrt(np.mean(np.sum((current - preferred) ** 2, axis=1)))
        )
        candidates.append(
            (preference_rms, prediction_rms, previous_pair,
             current_pair, translation)
        )
    if not candidates:
        raise ValueError(
            "no previous matched pair has a supported current continuation"
        )

    _preference_rms, prediction_rms, previous_pair, current_pair, translation = min(
        candidates, key=lambda result: (result[0], result[1])
    )
    previous = np.asarray(previous_pair, dtype=float)
    return (
        (
            (float(previous[0, 0]), float(previous[0, 1])),
            (float(previous[1, 0]), float(previous[1, 1])),
        ),
        current_pair,
        translation,
        prediction_rms,
    )


FrameDataLoader = Callable[[Any], Tuple[np.ndarray, Optional[np.ndarray]]]


@dataclass(frozen=True)
class TrackedPairFrame:
    """One canonical A/B pair continued into a raw camera frame.

    ``frame`` is caller-owned frame metadata (for example, a manifest row or a
    FITS path).  It is intentionally opaque to ``poltools``: the caller's
    ``load_data(frame)`` function owns FITS I/O and detector calibration.
    Coordinates follow the deterministic detector A/B convention, not a
    physical ordinary/extraordinary-ray identification.
    """

    frame: Any
    beam_a_xy: Tuple[float, float]
    beam_b_xy: Tuple[float, float]
    shift_xy: Optional[Tuple[float, float]]
    prediction_rms_px: Optional[float]

    @property
    def pair_xy(self) -> Tuple[Tuple[float, float], Tuple[float, float]]:
        """Return the tracked canonical A/B coordinate pair."""
        return self.beam_a_xy, self.beam_b_xy


def _load_tracking_data(
    frame: Any,
    load_data: FrameDataLoader,
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """Load one frame through the caller's calibration-aware adapter."""
    result = load_data(frame)
    if not isinstance(result, tuple) or len(result) != 2:
        raise ValueError("load_data(frame) must return (data, bad_pixel_mask)")
    data, bad_pixel_mask = result
    return np.asarray(data), bad_pixel_mask


def track_pair_sequence(
    frames: Sequence[Any],
    *,
    load_data: FrameDataLoader,
    initial_pair: Optional[Tuple[Tuple[float, float], Tuple[float, float]]] = None,
    translation_tolerance_px: float = 12.0,
    match_tolerance_px: float = 15.0,
    min_translation_support: int = 2,
) -> Tuple[TrackedPairFrame, ...]:
    """Continue one canonical A/B pair through an ordered camera sequence.

    ``frames`` is an ordered sequence of raw camera-frame metadata.  For each
    entry, ``load_data`` must return a calibrated ``(data, bad_pixel_mask)``
    pair.  The first detected A/B pair is used unless ``initial_pair`` is
    supplied; subsequent coordinates are always continued by
    :func:`track_matched_pair` and never replaced by a brighter source.

    This helper deliberately has no FITS I/O, notebook state, plotting, or HWP
    semantics.  It fails closed if the selected pair cannot be continued.
    """
    frames = tuple(frames)
    if not frames:
        raise ValueError("cannot track an empty frame sequence")

    first_frame = frames[0]
    first_data, first_mask = _load_tracking_data(first_frame, load_data)
    first_proposal = propose_anchor_pair(
        first_data, bad_pixel_mask=first_mask,
    )
    pair = (
        (first_proposal.beam_a_xy, first_proposal.beam_b_xy)
        if initial_pair is None else initial_pair
    )
    pair_array = np.asarray(pair, dtype=float)
    if pair_array.shape != (2, 2) or not np.all(np.isfinite(pair_array)):
        raise ValueError("initial_pair must contain two finite (x, y) coordinates")
    pair = (
        (float(pair_array[0, 0]), float(pair_array[0, 1])),
        (float(pair_array[1, 0]), float(pair_array[1, 1])),
    )
    tracked = [
        TrackedPairFrame(
            frame=first_frame,
            beam_a_xy=pair[0],
            beam_b_xy=pair[1],
            shift_xy=None,
            prediction_rms_px=None,
        )
    ]
    previous_matches = first_proposal.matched_pairs

    for frame in frames[1:]:
        data, bad_pixel_mask = _load_tracking_data(frame, load_data)
        proposal = propose_anchor_pair(data, bad_pixel_mask=bad_pixel_mask)
        pair, shift_xy, prediction_rms_px = track_matched_pair(
            pair,
            previous_matches,
            proposal.matched_pairs,
            translation_tolerance_px=translation_tolerance_px,
            match_tolerance_px=match_tolerance_px,
            min_translation_support=min_translation_support,
        )
        tracked.append(
            TrackedPairFrame(
                frame=frame,
                beam_a_xy=pair[0],
                beam_b_xy=pair[1],
                shift_xy=shift_xy,
                prediction_rms_px=prediction_rms_px,
            )
        )
        previous_matches = proposal.matched_pairs

    return tuple(tracked)


def _frame_label(frame: Any) -> str:
    """Use a camera filename when supplied, otherwise a useful frame label."""
    if isinstance(frame, Mapping):
        filename = frame.get("filename")
        if filename:
            return str(filename)
    filename = getattr(frame, "filename", None)
    return str(filename) if filename else str(frame)


def show_tracked_sequence(
    tracked_frames: Sequence[TrackedPairFrame],
    *,
    load_data: FrameDataLoader,
    half_size_px: float = 350.0,
    cmap: str = "gray",
    pair_color: str = "tab:green",
    percentile_clip: Tuple[float, float] = (5.0, 99.8),
):
    """Display a reviewed A/B track across an ordered camera sequence.

    The display reloads each frame through ``load_data`` rather than retaining
    multiple full detector arrays.  Bad or non-finite pixels are replaced only
    in a display copy; they are not repaired for photometry.  Green hollow
    rings and a shortened green connector identify the continued A/B pair.
    """
    tracked_frames = tuple(tracked_frames)
    if not tracked_frames:
        raise ValueError("cannot display an empty tracked-frame sequence")
    if half_size_px <= 0:
        raise ValueError("half_size_px must be positive")
    lo_pct, hi_pct = map(float, percentile_clip)
    if not 0.0 <= lo_pct < hi_pct <= 100.0:
        raise ValueError("percentile_clip must satisfy 0 <= low < high <= 100")

    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(
        1, len(tracked_frames), figsize=(4 * len(tracked_frames), 5), squeeze=False,
    )
    for ax, tracked in zip(axes[0], tracked_frames):
        data, bad_pixel_mask = _load_tracking_data(tracked.frame, load_data)
        image = np.asarray(data, dtype=float)
        if image.ndim != 2:
            raise ValueError("load_data(frame) must return a 2-D data array")
        invalid = ~np.isfinite(image)
        if bad_pixel_mask is not None:
            mask = np.asarray(bad_pixel_mask, dtype=bool)
            if mask.shape != image.shape:
                raise ValueError("bad_pixel_mask shape must match data")
            invalid |= mask
        good = ~invalid
        if not np.any(good):
            raise ValueError("frame has no finite, unmasked pixels for display")
        display_data = np.array(image, copy=True)
        display_data[invalid] = float(np.median(image[good]))
        vmin, vmax = np.percentile(display_data[good], (lo_pct, hi_pct))
        first, second = tracked.pair_xy
        midpoint = ((first[0] + second[0]) / 2.0, (first[1] + second[1]) / 2.0)

        ax.imshow(display_data, origin="upper", cmap=cmap, vmin=vmin, vmax=vmax)
        ax.annotate(
            "", second, first,
            arrowprops=dict(arrowstyle="-", color=pair_color, linewidth=1.3,
                            shrinkA=6, shrinkB=6),
        )
        ax.plot(
            *zip(first, second), linestyle="none", marker="o", color=pair_color,
            markerfacecolor="none", markersize=10,
        )
        ax.set_xlim(midpoint[0] - half_size_px, midpoint[0] + half_size_px)
        ax.set_ylim(midpoint[1] + half_size_px, midpoint[1] - half_size_px)
        if tracked.shift_xy is None:
            detail = "start"
        else:
            dx, dy = tracked.shift_xy
            detail = f"Δ=({dx:.1f}, {dy:.1f}) px\\nRMS={tracked.prediction_rms_px:.2f} px"
        ax.set_title(f"{_frame_label(tracked.frame)}\\n{detail}")

    fig.suptitle("Green rings are the automatically continued A/B pair")
    fig.tight_layout()
    plt.show()
    return fig


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
