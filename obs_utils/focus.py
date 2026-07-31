from __future__ import annotations

"""Star-image size metrics for focusing, measured on a frame already in hand.

Pure measurement: nothing here moves the focuser, fits a curve, or picks a best
position. A focus run is a human loop -- step the focuser, take a frame, measure
it, plot the sequence, read the minimum yourself. The plotting half lives in
:func:`obs_utils.live.focus_curve`; the stepping half is a notebook cell.

Two metrics, because they fail in different places:

``FWHM`` -- full width at half maximum of the star image. The number every
observer already has intuition for, and it is what "seeing" is quoted in. It
degrades badly when the star is far from focus: the profile flattens, the peak
becomes noise, and the half-maximum crossing jumps around.

``HFD`` -- half-flux diameter: twice the radius of the circle containing half the
star's background-subtracted flux. It stays monotonic and well-behaved far
outside focus, where FWHM has stopped meaning anything, which is what makes it
the useful quantity for the *coarse* end of a focus sweep.

Provenance, per the project's Sources A/B rule:

* Flux-weighted first and second moments (the centroid and the ``method="moments"``
  width) are SExtractor's ``X``/``Y`` and ``X2``/``Y2``, defined in
  **Bertin & Arnouts (1996), A&AS 117, 393** (doi:10.1051/aas:1996164) --
  verified in OpenAlex 2026-07-30. **ESTABLISHED.**
* ``FWHM = 2*sqrt(2*ln 2)*sigma`` for a Gaussian is arithmetic, not a method
  choice. **ESTABLISHED.**
* A radius enclosing a stated fraction of a source's flux is ordinary
  photometry (the half-light radius).  **ESTABLISHED.**
* **FLAGGED -- outside Sources A/B:** using half-flux diameter *as the focus
  metric* is standard practice in telescope control software, but a search of
  Semantic Scholar, OpenAlex and arXiv on 2026-07-30 turned up no peer-reviewed
  paper defining it for that purpose. It is used here because it is directly
  measurable, model-free, and monotonic in defocus -- not on anyone's authority.
  Nothing downstream of a focus decision enters a science result, so this does
  not touch a published number. If it ever does, replace this with a cited
  method rather than promoting the practice.

**Savart warning.** Every star in a POLITE science frame is a *pair* -- the
ordinary and extraordinary beams, split by ~239 px (see
``poltools.nominal_beam_separation_px``). An aperture wide enough to swallow both
measures the separation, not the focus, and reports a stable, meaningless number
that gets larger the better you focus. :data:`DEFAULT_APERTURE_PX` is sized well
inside one beam, and :func:`brightest_star` returns a single peak, so the default
path measures one beam. Do not widen the aperture past
:data:`MAX_SAFE_APERTURE_PX` without checking which beams you enclosed.
"""

import logging
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
from astropy.stats import sigma_clipped_stats

logger = logging.getLogger(__name__)

__all__ = [
    "StarProfile",
    "DEFAULT_APERTURE_PX",
    "MAX_SAFE_APERTURE_PX",
    "brightest_star",
    "measure_fwhm",
    "measure_hfd",
    "star_profile",
]

# Same clipping as obs_utils/qa_lib.py and live.py, so a background level printed
# by a focus cell is the number the QA gates would report.
_CLIP_SIGMA = 5.0
_CLIP_ITERS = 5

# Radius of the measurement box, in pixels. 25 px comfortably contains a focused
# QHY268M star image and a good deal of defocus, while staying far inside the
# ~239 px Savart beam separation (see the module docstring).
DEFAULT_APERTURE_PX = 25.0

# Half the nominal beam separation. Past this an aperture centred on one beam
# starts to reach the other one.
MAX_SAFE_APERTURE_PX = 119.0

_FWHM_PER_SIGMA = 2.0 * np.sqrt(2.0 * np.log(2.0))  # 2.3548...


@dataclass
class StarProfile:
    """One star measured on one frame.

    ``fwhm_px`` is ``nan`` when the profile never falls to half its peak inside
    the aperture -- badly defocused, or the aperture is too small. That is a real
    measurement outcome, not an error, and it is exactly the regime where you
    should be reading ``hfd_px`` instead.
    """

    x: float
    y: float
    fwhm_px: float
    hfd_px: float
    peak: float
    flux: float
    background: float
    aperture_px: float
    saturated: bool

    def line(self) -> str:
        fwhm = "  n/a" if not np.isfinite(self.fwhm_px) else f"{self.fwhm_px:5.2f}"
        sat = "  SATURATED" if self.saturated else ""
        return (
            f"({self.x:7.1f},{self.y:7.1f})  FWHM {fwhm} px  "
            f"HFD {self.hfd_px:5.2f} px  peak {self.peak:8.1f}  "
            f"bkg {self.background:7.1f}{sat}"
        )


def _cutout(
    data: np.ndarray, center: Tuple[float, float], aperture_px: float
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (values, dx, dy) for pixels within ``aperture_px`` of ``center``."""
    cx, cy = float(center[0]), float(center[1])
    half = int(np.ceil(aperture_px)) + 1
    y0 = max(0, int(round(cy)) - half)
    y1 = min(data.shape[0], int(round(cy)) + half + 1)
    x0 = max(0, int(round(cx)) - half)
    x1 = min(data.shape[1], int(round(cx)) + half + 1)
    if y1 <= y0 or x1 <= x0:
        raise ValueError(f"Aperture centred at ({cx:.1f}, {cy:.1f}) falls outside the frame")

    box = np.asarray(data[y0:y1, x0:x1], dtype=float)
    yy, xx = np.mgrid[y0:y1, x0:x1]
    dx = xx - cx
    dy = yy - cy
    inside = (dx * dx + dy * dy) <= aperture_px * aperture_px
    return box[inside], dx[inside], dy[inside]


def _background(data: np.ndarray, background: Optional[float]) -> float:
    if background is not None:
        return float(background)
    _mean, median, _std = sigma_clipped_stats(
        np.asarray(data, dtype=float), sigma=_CLIP_SIGMA, maxiters=_CLIP_ITERS
    )
    return float(median)


def brightest_star(
    data: np.ndarray,
    *,
    edge_margin_px: int = 30,
    aperture_px: float = DEFAULT_APERTURE_PX,
    background: Optional[float] = None,
    iterations: int = 4,
) -> Tuple[float, float]:
    """Locate the brightest star as a refined centroid, and return ``(x, y)``.

    The brightest pixel away from the edges, then repeated flux-weighted
    centroid passes over the aperture (Bertin & Arnouts 1996 first moments),
    each re-centred on the previous result. This is a *pointing aid for a focus
    run*, not source detection -- it finds one star, and it finds the brightest
    one, which on a Savart frame means one beam of the brightest pair.

    The iteration is not decoration. The CDK20 has a central obstruction, so a
    defocused star is a *donut*: its brightest pixel sits on the ring, not at
    the centre, and a single centroid pass seeded there stays stuck near the rim
    and reports a width for the wrong place. That is precisely the coarse-defocus
    regime HFD exists for, so it has to work. Iterating walks the centre in to
    the flux centroid of the whole ring.
    """
    arr = np.asarray(data, dtype=float)
    m = int(edge_margin_px)
    if arr.shape[0] <= 2 * m or arr.shape[1] <= 2 * m:
        m = 0
    inner = arr[m : arr.shape[0] - m, m : arr.shape[1] - m] if m else arr
    iy, ix = np.unravel_index(int(np.argmax(inner)), inner.shape)
    cx, cy = float(ix + m), float(iy + m)

    bkg = _background(arr, background)
    for _ in range(max(1, int(iterations))):
        values, dx, dy = _cutout(arr, (cx, cy), aperture_px)
        weights = np.clip(values - bkg, 0.0, None)
        total = float(weights.sum())
        if total <= 0:
            logger.warning("No flux above background at the peak pixel; using it uncentroided")
            return cx, cy
        shift_x = float((weights * dx).sum()) / total
        shift_y = float((weights * dy).sum()) / total
        cx, cy = cx + shift_x, cy + shift_y
        if abs(shift_x) < 0.01 and abs(shift_y) < 0.01:
            break
    return cx, cy


def star_profile(
    data: np.ndarray,
    *,
    center: Optional[Tuple[float, float]] = None,
    aperture_px: float = DEFAULT_APERTURE_PX,
    background: Optional[float] = None,
    full_scale_adu: float = 65535.0,
) -> StarProfile:
    """Measure one star: centroid, FWHM, HFD, peak, flux and background.

    ``center`` defaults to :func:`brightest_star`. ``background`` defaults to the
    sigma-clipped median of the whole frame, which is right for a sparse field
    and wrong inside a nebula -- pass your own value there.
    """
    arr = np.asarray(data, dtype=float)
    if aperture_px > MAX_SAFE_APERTURE_PX:
        logger.warning(
            "aperture_px=%.1f exceeds %.1f px (half the nominal Savart beam "
            "separation): this aperture can contain both beams of a pair, and "
            "the result would measure the beam split, not the focus.",
            aperture_px,
            MAX_SAFE_APERTURE_PX,
        )

    bkg = _background(arr, background)
    if center is None:
        cx, cy = brightest_star(arr, aperture_px=aperture_px, background=bkg)
    else:
        cx, cy = center
    values, dx, dy = _cutout(arr, (cx, cy), aperture_px)
    radii = np.hypot(dx, dy)
    net = values - bkg

    peak = float(net.max()) if net.size else float("nan")
    return StarProfile(
        x=float(cx),
        y=float(cy),
        fwhm_px=_fwhm_from_profile(radii, net, peak),
        hfd_px=_hfd_from_profile(radii, net),
        peak=peak,
        flux=float(np.clip(net, 0.0, None).sum()),
        background=bkg,
        aperture_px=float(aperture_px),
        saturated=bool(values.size and values.max() >= full_scale_adu),
    )


def _fwhm_from_profile(radii: np.ndarray, net: np.ndarray, peak: float) -> float:
    """Radial profile crossing of half the peak, doubled. No model assumed.

    Bins the background-subtracted pixels by radius, then linearly interpolates
    the radius where the mean profile drops through ``peak / 2``. Returns ``nan``
    when the profile never gets there inside the aperture -- see
    :class:`StarProfile`.
    """
    if not np.isfinite(peak) or peak <= 0 or radii.size == 0:
        return float("nan")

    edges = np.arange(0.0, float(radii.max()) + 1.0, 1.0)
    if edges.size < 2:
        return float("nan")
    index = np.digitize(radii, edges) - 1
    valid = (index >= 0) & (index < edges.size - 1)
    counts = np.bincount(index[valid], minlength=edges.size - 1)
    sums = np.bincount(index[valid], weights=net[valid], minlength=edges.size - 1)
    occupied = counts > 0
    if not occupied.any():
        return float("nan")
    centers = (0.5 * (edges[:-1] + edges[1:]))[occupied]
    profile = sums[occupied] / counts[occupied]

    half = peak / 2.0
    below = np.nonzero(profile < half)[0]
    if below.size == 0 or below[0] == 0:
        return float("nan")
    i = int(below[0])
    r_hi, r_lo = centers[i], centers[i - 1]
    p_hi, p_lo = profile[i], profile[i - 1]
    if p_lo == p_hi:
        return float(2.0 * r_lo)
    return float(2.0 * (r_lo + (p_lo - half) * (r_hi - r_lo) / (p_lo - p_hi)))


def _hfd_from_profile(radii: np.ndarray, net: np.ndarray) -> float:
    """Twice the radius enclosing half the background-subtracted flux.

    Negative pixels are clipped to zero first, so sky noise cannot cancel star
    flux and shrink the reported diameter.
    """
    flux = np.clip(net, 0.0, None)
    total = float(flux.sum())
    if total <= 0 or radii.size == 0:
        return float("nan")

    order = np.argsort(radii)
    r_sorted = radii[order]
    cumulative = np.cumsum(flux[order])
    half = total / 2.0
    i = int(np.searchsorted(cumulative, half))
    if i == 0:
        return float(2.0 * r_sorted[0])
    if i >= r_sorted.size:
        return float(2.0 * r_sorted[-1])
    c_lo, c_hi = cumulative[i - 1], cumulative[i]
    r_lo, r_hi = r_sorted[i - 1], r_sorted[i]
    if c_hi == c_lo:
        return float(2.0 * r_hi)
    return float(2.0 * (r_lo + (half - c_lo) * (r_hi - r_lo) / (c_hi - c_lo)))


def measure_fwhm(
    data: np.ndarray,
    *,
    center: Optional[Tuple[float, float]] = None,
    aperture_px: float = DEFAULT_APERTURE_PX,
    background: Optional[float] = None,
    method: str = "profile",
) -> float:
    """Full width at half maximum of one star image, in pixels.

    ``method="profile"`` (default) interpolates the radius at which the measured
    radial profile falls to half its peak and doubles it -- no model assumed, and
    ``nan`` when the star never gets there inside the aperture.

    ``method="moments"`` instead reports ``2*sqrt(2*ln 2)`` times the
    flux-weighted second moment (Bertin & Arnouts 1996 ``X2``/``Y2``), i.e. the
    FWHM the star *would* have if it were Gaussian. It always returns a number,
    including for profiles that are nothing like a Gaussian, so read it as a
    width statistic rather than as a true FWHM.
    """
    if method == "profile":
        return star_profile(
            data, center=center, aperture_px=aperture_px, background=background
        ).fwhm_px
    if method != "moments":
        raise ValueError(f"method must be 'profile' or 'moments', got {method!r}")

    arr = np.asarray(data, dtype=float)
    bkg = _background(arr, background)
    if center is None:
        cx, cy = brightest_star(arr, aperture_px=aperture_px, background=bkg)
    else:
        cx, cy = center
    values, dx, dy = _cutout(arr, (cx, cy), aperture_px)
    weights = np.clip(values - bkg, 0.0, None)
    total = float(weights.sum())
    if total <= 0:
        return float("nan")
    var_x = float((weights * dx * dx).sum()) / total
    var_y = float((weights * dy * dy).sum()) / total
    return float(_FWHM_PER_SIGMA * np.sqrt(0.5 * (var_x + var_y)))


def measure_hfd(
    data: np.ndarray,
    *,
    center: Optional[Tuple[float, float]] = None,
    aperture_px: float = DEFAULT_APERTURE_PX,
    background: Optional[float] = None,
) -> float:
    """Half-flux diameter of one star image, in pixels.

    Twice the radius of the circle containing half the star's
    background-subtracted flux within ``aperture_px``. Monotonic in defocus well
    past the point where FWHM stops being measurable, which is why a focus sweep
    should be read on this first and refined on FWHM.

    Note the aperture matters: HFD is a *fraction* of the flux you enclosed, so
    widening the aperture on a defocused star raises the number. Keep it fixed
    across a sweep, and see the module docstring about the Savart beam pair.
    """
    return star_profile(
        data, center=center, aperture_px=aperture_px, background=background
    ).hfd_px
