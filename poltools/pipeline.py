"""
poltools.pipeline — End-to-end reduction from raw frames to StokesResult.

Processing steps::

    frames → group by filter → group by half-wave plate angle
          → detect and pair ordinary/extraordinary beams
          → aperture photometry → modulation fit → calibration → Stokes assembly

The α-BBO Savart split depends on wavelength; each filter band is reduced with
its own :class:`~poltools._types.BeamGeometry` via ``cfg.for_filter(name)``.
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
    """Apply :class:`PolCalibration` to a modulation-result dict."""
    q_c, u_c = calib.apply(qu["q"], qu["u"])
    out = dict(qu)
    out["q"], out["u"] = q_c, u_c
    if calib.efficiency not in (0.0, 1.0):
        out["sigma_q"] = qu["sigma_q"] / calib.efficiency
        out["sigma_u"] = qu["sigma_u"] / calib.efficiency
        if "cov_qu" in qu:
            out["cov_qu"] = qu["cov_qu"] / calib.efficiency ** 2
    return out


def _safe_float(v) -> Optional[float]:
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def _config_for_filter(cfg: PolConfig, band: str) -> PolConfig:
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
    """Warn if the instrument rotator angle drifts across a sequence.

    Position-angle calibration assumes a constant rotator angle per sequence;
    drift smears recovered polarization angles on the sky.
    """
    vals = [v for v in instrot_vals if v is not None and np.isfinite(v)]
    if len(vals) >= 2 and (max(vals) - min(vals)) > tol_deg:
        warnings.warn(
            f"filter {band!r}: INSTROT varies across the sequence "
            f"({min(vals):.3f}..{max(vals):.3f} deg). Recovered position "
            f"angles may be smeared.",
            stacklevel=3,
        )


def reduce_to_stokes(
    frame_paths: Sequence[str],
    cfg: PolConfig,
    *,
    o_positions: Optional[Sequence[Tuple[float, float]]] = None,
    names: Optional[Sequence[str]] = None,
    method: str = "lsq",
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
    allow_mixed_sequences: bool = False,
) -> List[StokesResult]:
    """Reduce a half-wave plate sequence to per-source Stokes results.

    Parameters
    ----------
    frame_paths : sequence of str
        FITS paths for one polarimetry sequence (one or more filters).
    cfg : PolConfig
        Instrument configuration; must match how the data were taken.
    o_positions : list of (x, y), optional
        Pixel positions of the **ordinary** beam for each source.
        Required unless ``detect=True``.
    method : {"lsq", "double_ratio", "double_difference"}
        How to extract normalized Stokes ``q, u`` from the modulation curve:

        * ``lsq`` — weighted least-squares fit (default; needs flat-fielded
          frames and at least four half-wave plate angles).
        * ``double_ratio`` — ratio-of-ratios; cancels flat-field errors
          (Tinbergen 1996). Needs angles 0°, 22.5°, 45°, 67.5°.
        * ``double_difference`` — simpler comparator; useful as a cross-check.
    estimator : {"mas", "wk", "naive"}
        How to debias the reported polarization fraction ``p_report``:

        * ``mas`` — Modified Asymptotic estimator (Plaszczynski et al. 2014).
        * ``wk`` — Wardle & Kronberg (1974) square-root debiasing.
        * ``naive`` — no debiasing (biased at low signal-to-noise).
    calibration : PolCalibration, optional
        Standard-star calibration applied before Stokes assembly.
    ron_map : ndarray, optional
        Per-pixel read-noise map [electrons] for the photometric uncertainty
        model.
    bad_pixel_mask : ndarray of bool, optional
        Mask of hot or bad pixels (True = bad); filled locally before photometry.
    exclude_regions : list of (x0, y0, x1, y1), optional
        Rectangular detector areas to skip (e.g. a vignetted Savart corner).
    allow_mixed_sequences : bool
        Permit intentional stacking across different ``OBJECT/POLSEQ`` values.
        False by default because mixing targets, repeats, or instrument rotations
        silently destroys their provenance.
    """
    reducers = {
        "lsq": mod.lsq_modulation,
        "double_ratio": mod.double_ratio,
        "double_difference": mod.double_difference,
    }
    m = method.lower()
    if m not in reducers:
        raise ValueError(
            f"Unknown method {method!r} (use {', '.join(sorted(reducers))})"
        )
    reduce_fn = reducers[m]

    sequence_groups = pol_io.group_by_pol_sequence(list(frame_paths))
    sequence_ids = {(obj, seq) for (obj, seq, _band) in sequence_groups}
    if len(sequence_ids) > 1 and not allow_mixed_sequences:
        detail = ", ".join(
            f"{obj!r}/{seq!r}" for obj, seq in sorted(sequence_ids)
        )
        raise ValueError(
            "frame_paths contain multiple OBJECT/POLSEQ sequences: "
            f"{detail}. Reduce each sequence separately or pass "
            "allow_mixed_sequences=True only for intentional repeat stacks."
        )

    by_filter = pol_io.group_by_filter(list(frame_paths))

    def _reduce_band(band_paths: List[str], cfg_b: PolConfig,
                     band: str) -> List[StokesResult]:
        # Median-combine repeat frames at the same half-wave plate angle.
        # Robust to cosmic rays and random telegraph noise spikes on CMOS sensors.
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
            n_by_angle[ang] = len(stack)

        _warn_if_instrot_varies(instrot_vals, band)

        o_pos = None if o_positions is None else list(o_positions)
        names_b = None if names is None else list(names)
        if o_pos is None:
            if not detect:
                raise ValueError("Provide o_positions or set detect=True")
            active_filter = cfg_b.active_filter()
            if active_filter is not None and not active_filter.characterized:
                warnings.warn(
                    f"filter {band!r}: beam geometry is an uncharacterized "
                    f"placeholder (separation={cfg_b.beam.separation_px:.2f} px, "
                    f"PA={cfg_b.beam.position_angle_deg:.2f} deg); automatic "
                    "o/e pairing may fail or mis-pair sources",
                    stacklevel=2,
                )
            fwhm = fwhm_px if fwhm_px is not None else cfg_b.fwhm_px(seeing_arcsec)
            ref = frames_by_angle[sorted(frames_by_angle)[0]]
            det = phot.detect_sources(ref, fwhm, threshold_sigma=threshold_sigma)
            pairs = phot.pair_oe(det, cfg_b.beam, exclude_regions=exclude_regions)
            o_pos = [o for (o, e) in pairs]
            names_b = names_b or [f"src{i}" for i in range(len(o_pos))]
            if not o_pos:
                return []
        elif exclude_regions:
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

        flux_by_src = phot.photometer_sequence(
            frames_by_angle, cfg_b, o_pos, names_b,
            r_ap=r_ap, r_in=r_in, r_out=r_out, bias_adu=bias_adu,
            n_by_angle=n_by_angle, combine="median",
            ron_map=ron_map, bad_pixel_mask=bad_pixel_mask,
        )

        band_results: List[StokesResult] = []
        for nm, bfs in flux_by_src.items():
            qu = reduce_fn(bfs)

            if calibration is not None:
                qu = _calibrate_qu(qu, calibration)

            sat_angles = [b.hwp_deg for b in bfs if b.saturated]
            if sat_angles:
                warnings.warn(
                    f"source {nm!r} [{band}]: aperture peak reached the "
                    f"saturation/linearity limit at HWP angle(s) {sat_angles}; "
                    f"q,u may be biased",
                    stacklevel=2,
                )

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


def reduce_pol_sequences(
    frame_paths: Sequence[str],
    cfg: PolConfig,
    **kwargs,
) -> Dict[Tuple[str, str, str], List[StokesResult]]:
    """Reduce each ``(OBJECT, POLSEQ, FILTER)`` group independently."""
    groups = pol_io.group_by_pol_sequence(list(frame_paths))
    return {
        key: reduce_to_stokes(paths, cfg, **kwargs)
        for key, paths in groups.items()
    }
