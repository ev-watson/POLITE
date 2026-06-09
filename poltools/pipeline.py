"""
poltools.pipeline — End-to-end reduction: raw frames → calibrated StokesResult.

Orchestrates the full chain:

    frames → group by FILTER → group by HWP → (detect + pair o/e)
          → aperture photometry
          → modulation (double_ratio / double_difference / lsq) → calibration
          → Stokes assembly + errors

Frames are grouped by EFW band **first**: the α-BBO Savart split is dispersive,
so each band is reduced with its own :class:`~poltools._types.BeamGeometry` via
``cfg.for_filter(name)``. One :class:`StokesResult` is produced per source per
band (the band is recorded in ``metadata['filter']``). Source positions may be
provided (recommended when known, e.g. the simulator showcase) or detected with
``photutils`` (``detect=True``).
"""

from __future__ import annotations

import warnings
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from ._types import PolConfig, StokesResult
from . import io as pol_io
from . import photometry as phot
from . import modulation as mod
from . import stokes as stk
from .calibration import PolCalibration


def _calibrate_qu(qu: Dict[str, float], calib: PolCalibration) -> Dict[str, float]:
    """Apply a PolCalibration to a modulation-result dict (q,u + sigmas)."""
    q_c, u_c = calib.apply(qu["q"], qu["u"])
    out = dict(qu)
    out["q"], out["u"] = q_c, u_c
    # efficiency rescales the uncertainties; IP/PA-rotation leave |σ| ~unchanged
    if calib.efficiency not in (0.0, 1.0):
        out["sigma_q"] = qu["sigma_q"] / calib.efficiency
        out["sigma_u"] = qu["sigma_u"] / calib.efficiency
    return out


def _safe_float(v) -> Optional[float]:
    """``float(v)`` or ``None`` (missing/unparseable FITS card)."""
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def _config_for_filter(cfg: PolConfig, band: str) -> PolConfig:
    """Per-band config: ``cfg.for_filter(band)`` if registered, else ``cfg``.

    Empty registry (single-filter / legacy use) → ``cfg`` unchanged. Non-empty
    registry but an unregistered band → warn and fall back to ``cfg.beam`` so the
    reduction still runs.
    """
    if not cfg.filters:
        return cfg
    try:
        return cfg.for_filter(band)
    except KeyError:
        warnings.warn(
            f"filter {band!r} not in cfg.filters registry "
            f"{[f.name for f in cfg.filters]!r}; using cfg.beam as-is",
            stacklevel=3,
        )
        return cfg


def _warn_if_instrot_varies(instrot_vals, band: str, tol_deg: float = 0.05) -> None:
    """Warn if the PWI4 field-rotator angle drifts across a sequence (WS2).

    The PA-on-sky zero-point assumes a **constant** alpha per sequence —
    standard-star PA calibration absorbs only a constant offset — so a varying
    INSTROT would smear recovered angles.
    """
    vals = [v for v in instrot_vals if v is not None and np.isfinite(v)]
    if len(vals) >= 2 and (max(vals) - min(vals)) > tol_deg:
        warnings.warn(
            f"filter {band!r}: PWI4 field-rotator angle (INSTROT) varies across "
            f"the sequence (range {min(vals):.3f}..{max(vals):.3f} deg). The "
            f"PA-on-sky zero-point assumes constant alpha per sequence; recovered "
            f"position angles may be smeared.",
            stacklevel=3,
        )


def reduce_to_stokes(
    frame_paths: Sequence[str],
    cfg: PolConfig,
    *,
    o_positions: Optional[Sequence[Tuple[float, float]]] = None,
    names: Optional[Sequence[str]] = None,
    method: str = "double_ratio",
    estimator: str = "mas",
    calibration: Optional[PolCalibration] = None,
    r_ap: float = 6.0, r_in: float = 10.0, r_out: float = 16.0,
    bias_adu: float = 1000.0,
    ron_map: Optional[np.ndarray] = None,
    bad_pixel_mask: Optional[np.ndarray] = None,
    detect: bool = False,
    fwhm_px: Optional[float] = None,
    threshold_sigma: float = 5.0,
    seeing_arcsec: float = 2.0,
    exclude_regions: Optional[Sequence[Tuple[float, float, float, float]]] = None,
) -> List[StokesResult]:
    """Reduce a HWP sequence to per-source calibrated Stokes results.

    Parameters
    ----------
    frame_paths : sequence of str
        FITS frames of one polarimetry sequence (any HWP order).
    cfg : PolConfig
        Instrument configuration (must match acquisition).
    o_positions : list of (x, y), optional
        Ordinary-beam positions. If omitted and ``detect`` is True, sources are
        found with DAOStarFinder and paired via the beam offset.
    method : {"double_ratio", "double_difference", "lsq"}
        "double_ratio" (primary, flat-field-independent; needs angles
        {0,22.5,45,67.5}); "double_difference" (first-order comparator, same
        angles); "lsq" (least-squares modulation fit, N>=4 angles).
    estimator : {"mas","wk","naive"}
        Debiasing estimator reported as ``p_report``.
    calibration : PolCalibration, optional
        Applied to (q,u) before Stokes assembly.
    ron_map : ndarray, optional
        Per-pixel read-noise map (**electrons**) for the CCD-equation read term
        (finding C-1; e.g. ``caltools.read_noise_map`` × gain). Defaults to the
        scalar ``cfg.read_noise_e``.
    bad_pixel_mask : ndarray of bool, optional
        Hot/bad-pixel mask (True = bad) repaired before photometry (finding
        C-2; e.g. the ``reduction.md`` hot-pixel census). Most important for the
        N=1 case the per-angle median cannot reject.
    exclude_regions : list of (x0, y0, x1, y1), optional
        Bad FOV rectangles (detector coords, origin upper-left). Any source whose
        o- or e-aperture centre falls inside is dropped — the readiness hook for
        the α-BBO Savart's chipped corner, to be characterized from real flats
        and applied only if it lands on-field. Default ``None`` → no exclusion.
    """
    # 0. validate the reduction method up front — before any file I/O, and not
    #    late inside the per-source loop (where a bad method wasted the whole
    #    reduction, or was never checked at all when zero sources were detected).
    reducers = {
        "double_ratio": mod.double_ratio,        # flat-field-independent (primary)
        "double_difference": mod.double_difference,  # first-order comparator
        "lsq": mod.lsq_modulation,               # least-squares modulation fit
    }
    m = method.lower()
    if m not in reducers:
        raise ValueError(
            f"Unknown method {method!r} (use {', '.join(sorted(reducers))})"
        )
    reduce_fn = reducers[m]

    # Group by EFW band FIRST: the α-BBO Savart split is dispersive, so each band
    # must be reduced with its own BeamGeometry (cfg.for_filter). Single-filter /
    # legacy sequences fall through as one band with cfg.beam unchanged.
    by_filter = pol_io.group_by_filter(list(frame_paths))

    def _reduce_band(band_paths: List[str], cfg_b: PolConfig,
                     band: str) -> List[StokesResult]:
        # 1. read + group by HWP angle. Repeat frames at the same angle are
        #    *median*-combined, not silently overwritten. This is the combination
        #    SOLVEPOL (Ramírez et al. 2017, Source B) applies to multiple images
        #    per waveplate position, and the robust combiner the project's CMOS
        #    reduction adopts against random-telegraph / salt-and-pepper outliers
        #    (reduction.md; Alarcón et al. 2023). The median rejects cosmic rays
        #    and transient hot pixels that a mean would smear into the aperture,
        #    and — like the mean — preserves the ADU + bias-pedestal scale the
        #    photometry noise model assumes. N=1 → the frame itself; N=2 → mean.
        groups = pol_io.group_by_hwp_angle(band_paths)
        frames_by_angle: Dict[float, np.ndarray] = {}
        n_by_angle: Dict[float, int] = {}
        instrot_vals: List[Optional[float]] = []
        for ang, paths_at_angle in groups.items():
            stack = []
            for p in paths_at_angle:
                data, _hwp, hdr = pol_io.read_pol_frame(p)
                stack.append(data)
                instrot_vals.append(_safe_float(hdr.get("INSTROT")))
            frames_by_angle[ang] = (stack[0] if len(stack) == 1
                                    else np.median(stack, axis=0))
            # frames combined here -> the photometric σ is suppressed by the
            # effective N downstream (C-3; Stockmans et al. eq. 16, Source B).
            n_by_angle[ang] = len(stack)

        # PWI4 field-rotator consistency across the sequence (WS2).
        _warn_if_instrot_varies(instrot_vals, band)

        # 2. positions: provided or detected (e-beam via this band's offset).
        o_pos = None if o_positions is None else list(o_positions)
        names_b = None if names is None else list(names)
        if o_pos is None:
            if not detect:
                raise ValueError("Provide o_positions or set detect=True")
            fwhm = fwhm_px if fwhm_px is not None else cfg_b.fwhm_px(seeing_arcsec)
            ref = frames_by_angle[sorted(frames_by_angle)[0]]
            det = phot.detect_sources(ref, fwhm, threshold_sigma=threshold_sigma)
            pairs = phot.pair_oe(det, cfg_b.beam, exclude_regions=exclude_regions)
            o_pos = [o for (o, e) in pairs]
            names_b = names_b or [f"src{i}" for i in range(len(o_pos))]
            if not o_pos:
                return []
        else:
            # provided positions: drop any whose o- or e-centre lands in a bad
            # FOV region (e.g. the Savart chipped corner) for this band's offset.
            if exclude_regions:
                dx, dy = cfg_b.beam.offset_xy()
                names_b = (names_b if names_b is not None
                           else [f"src{i}" for i in range(len(o_pos))])
                kept = [(o, nm) for o, nm in zip(o_pos, names_b)
                        if not (phot.point_in_regions(o[0], o[1], exclude_regions)
                                or phot.point_in_regions(o[0] + dx, o[1] + dy,
                                                         exclude_regions))]
                o_pos = [o for o, _ in kept]
                names_b = [nm for _, nm in kept]
                if not o_pos:
                    return []

        names_b = (list(names_b) if names_b is not None
                   else [f"src{i}" for i in range(len(o_pos))])

        # 3. aperture photometry across angles (C-1/C-2/C-3 threaded through).
        flux_by_src = phot.photometer_sequence(
            frames_by_angle, cfg_b, o_pos, names_b,
            r_ap=r_ap, r_in=r_in, r_out=r_out, bias_adu=bias_adu,
            n_by_angle=n_by_angle, combine="median",
            ron_map=ron_map, bad_pixel_mask=bad_pixel_mask,
        )

        # 4-6. modulation → calibration → Stokes assembly
        band_results: List[StokesResult] = []
        for nm, bfs in flux_by_src.items():
            qu = reduce_fn(bfs)

            if calibration is not None:
                qu = _calibrate_qu(qu, calibration)

            # saturation/linearity flag (C-6): a saturated o/e core biases the
            # beam ratio. Surface it (and warn) rather than reducing silently.
            sat_angles = [b.hwp_deg for b in bfs if b.saturated]
            if sat_angles:
                warnings.warn(
                    f"source {nm!r} [{band}]: aperture peak reached the "
                    f"saturation/linearity limit at HWP angle(s) {sat_angles}; "
                    f"q,u may be biased",
                    stacklevel=2,
                )

            # intensity reference: mean total flux across angles
            I0 = float(np.mean([b.f_o + b.f_e for b in bfs]))
            res = stk.assemble_stokes(
                qu, I0=I0, name=nm, method=m, estimator=estimator,
                extra_metadata={"n_angles": len(bfs),
                                "filter": band,
                                "calibrated": calibration is not None,
                                "saturated": bool(sat_angles),
                                "saturated_angles": sat_angles},
            )
            band_results.append(res)
        return band_results

    results: List[StokesResult] = []
    for band in sorted(by_filter):
        cfg_b = _config_for_filter(cfg, band)
        results.extend(_reduce_band(by_filter[band], cfg_b, band))
    return results
