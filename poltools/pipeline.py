"""
poltools.pipeline — End-to-end reduction: raw frames → calibrated StokesResult.

Orchestrates the full chain:

    frames → group by HWP → (detect + pair o/e) → aperture photometry
          → modulation (double_ratio / double_difference / lsq) → calibration
          → Stokes assembly + errors

One :class:`StokesResult` is produced per source. Source positions may be
provided (recommended when known, e.g. the simulator showcase) or detected with
``photutils`` (``detect=True``).
"""

from __future__ import annotations

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
    detect: bool = False,
    fwhm_px: Optional[float] = None,
    threshold_sigma: float = 5.0,
    seeing_arcsec: float = 2.0,
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

    # 1. read + group by HWP angle. Repeat frames at the same angle are
    #    mean-combined (preserving the ADU + bias-pedestal scale the photometry
    #    noise model assumes) rather than silently overwritten — multiple
    #    exposures per angle are standard for stacking / cosmic-ray rejection.
    groups = pol_io.group_by_hwp_angle(list(frame_paths))
    frames_by_angle: Dict[float, np.ndarray] = {}
    for ang, paths_at_angle in groups.items():
        stack = [pol_io.read_pol_frame(p)[0] for p in paths_at_angle]
        frames_by_angle[ang] = (stack[0] if len(stack) == 1
                                else np.mean(stack, axis=0))

    # 2. positions: provided or detected
    if o_positions is None:
        if not detect:
            raise ValueError("Provide o_positions or set detect=True")
        fwhm = fwhm_px if fwhm_px is not None else cfg.fwhm_px(seeing_arcsec)
        ref = frames_by_angle[sorted(frames_by_angle)[0]]
        det = phot.detect_sources(ref, fwhm, threshold_sigma=threshold_sigma)
        pairs = phot.pair_oe(det, cfg.beam)
        o_positions = [o for (o, e) in pairs]
        names = names or [f"src{i}" for i in range(len(o_positions))]
        if not o_positions:
            return []

    names = list(names) if names is not None else [f"src{i}" for i in range(len(o_positions))]

    # 3. aperture photometry across angles
    flux_by_src = phot.photometer_sequence(
        frames_by_angle, cfg, o_positions, names,
        r_ap=r_ap, r_in=r_in, r_out=r_out, bias_adu=bias_adu,
    )

    # 4-6. modulation → calibration → Stokes assembly
    results: List[StokesResult] = []
    for nm, bfs in flux_by_src.items():
        qu = reduce_fn(bfs)

        if calibration is not None:
            qu = _calibrate_qu(qu, calibration)

        # intensity reference: mean total flux across angles
        I0 = float(np.mean([b.f_o + b.f_e for b in bfs]))
        res = stk.assemble_stokes(
            qu, I0=I0, name=nm, method=m, estimator=estimator,
            extra_metadata={"n_angles": len(bfs),
                            "calibrated": calibration is not None},
        )
        results.append(res)
    return results
