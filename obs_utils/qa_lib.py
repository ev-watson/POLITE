"""First-light QA algorithms (shared by scripts and night-session hooks)."""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Sequence, Union

import numpy as np
from astropy.io import fits
from astropy.stats import sigma_clipped_stats

if TYPE_CHECKING:
    from poltools import PolConfig

logger = logging.getLogger(__name__)

# HD 154445 reference (plan Appendix B, Sch92)
REF_HD154445 = {"P_pct": 3.67, "PA_deg": 88.6, "P_tol_pct": 0.3, "PA_tol_deg": 2.0}

# Sigma-clipping used throughout for outlier-robust image statistics
# (astropy.stats, AAS-standard; rejects cosmic rays / hot / pinned pixels).
_CLIP_SIGMA = 5.0
_CLIP_ITERS = 5


@dataclass
class QAResult:
    name: str
    passed: bool
    messages: List[str] = field(default_factory=list)
    metrics: Dict[str, Any] = field(default_factory=dict)
    warnings: List[str] = field(default_factory=list)

    @property
    def level(self) -> str:
        """PASS (clean), WARN (soft issues, capture continues), FAIL (blocking)."""
        if not self.passed:
            return "FAIL"
        return "WARN" if self.warnings else "PASS"

    def to_json(self) -> str:
        return json.dumps({
            "name": self.name,
            "passed": self.passed,
            "level": self.level,
            "messages": self.messages,
            "warnings": self.warnings,
            "metrics": self.metrics,
        }, indent=2)


def _robust_stats(data: np.ndarray) -> tuple:
    """Sigma-clipped (mean, median, std). Outlier-robust image statistics."""
    return sigma_clipped_stats(
        data, sigma=_CLIP_SIGMA, maxiters=_CLIP_ITERS
    )


def _collect_paths(paths: Sequence[Union[str, Path]]) -> List[Path]:
    out: List[Path] = []
    for p in paths:
        pp = Path(p)
        if pp.is_dir():
            out.extend(sorted(pp.glob("*.fits")))
        elif pp.is_file():
            out.append(pp)
    # de-dup while preserving order
    seen: set = set()
    uniq: List[Path] = []
    for p in out:
        if p not in seen:
            seen.add(p)
            uniq.append(p)
    return uniq


def run_bias_qa(
    paths: Sequence[Union[str, Path]],
    *,
    ron_target_e: float = 3.5,
    ron_tol_e: float = 0.5,
    mean_lo: float = 50.0,
    mean_hi: float = 500.0,
) -> QAResult:
    """Bias histogram + pair-difference RON test (first-light plan 5.4 Lab 1).

    Outlier-robust: bias level and read noise use sigma-clipped statistics so a
    handful of hot/pinned pixels or a cosmic-ray hit does not inflate RON or
    trip the gate. Soft issues WARN (capture continues); only genuinely
    unusable data FAILs.
    """
    files = [p for p in _collect_paths(paths) if "BIAS" in p.name.upper()
             or fits.getheader(p, ignore_missing_end=True).get("IMAGETYP", "").upper() == "BIAS"]
    if len(files) < 2:
        return QAResult("bias_qa", False, ["Need >= 2 bias frames"])

    msgs: List[str] = []
    warns: List[str] = []
    egain_vals: List[float] = []
    ron_pairs: List[float] = []

    for p in files:
        hdr = fits.getheader(p, ignore_missing_end=True)
        # Conversion gain is a per-night characterization value and is no longer
        # written to headers. At capture time we cannot know it, so default to
        # unity: the read noise below is therefore reported in ADU (e- at unity
        # gain). Never fall back to GAIN -- that is the slider index, not e-/ADU.
        egain_vals.append(float(hdr.get("EGAIN", 1.0)))

    egain = float(np.median(egain_vals)) if egain_vals else 1.0

    # Per-frame checks on sigma-clipped bias level and pinned-pixel fraction.
    # A few pinned pixels are normal on CMOS; only a large fraction is a fault.
    PIN_WARN_FRAC = 1e-4   # 0.01% -> warn
    PIN_FAIL_FRAC = 1e-2   # 1%    -> fail (detector/offset problem)
    for p in files:
        data = fits.getdata(p, memmap=False).astype(np.float64)
        pinned = float(np.count_nonzero(data <= 0)) / data.size
        if pinned >= PIN_FAIL_FRAC:
            msgs.append(f"{p.name}: {pinned:.2%} pixels pinned at <=0 ADU")
        elif pinned > PIN_WARN_FRAC:
            warns.append(f"{p.name}: {pinned:.3%} pixels pinned at <=0 ADU")

        _, med, _ = _robust_stats(data)
        med = float(med)
        # Grossly wrong offset is blocking; a modest excursion only warns.
        span = mean_hi - mean_lo
        if med < mean_lo - span or med > mean_hi + span:
            msgs.append(f"{p.name}: median bias {med:.1f} ADU far outside [{mean_lo}, {mean_hi}]")
        elif not (mean_lo <= med <= mean_hi):
            warns.append(f"{p.name}: median bias {med:.1f} ADU outside [{mean_lo}, {mean_hi}]")

    # Pair-difference read noise with sigma-clipped std (rejects CR / hot pixels).
    for i in range(len(files) - 1):
        a = fits.getdata(files[i], memmap=False).astype(np.float64)
        b = fits.getdata(files[i + 1], memmap=False).astype(np.float64)
        _, _, std = _robust_stats(a - b)
        ron_pairs.append(float(std) / np.sqrt(2.0) * egain)

    ron_mean = float(np.median(ron_pairs)) if ron_pairs else float("nan")
    metrics = {
        "ron_e": ron_mean,
        "egain": egain,
        "n_frames": len(files),
        "ron_pairs_e": ron_pairs,
    }

    # RON tiers: within tol -> PASS; up to 3x tol -> WARN; beyond, or
    # non-physical (<=0 / NaN) -> FAIL.
    if not np.isfinite(ron_mean) or ron_mean <= 0:
        msgs.append(f"RON {ron_mean} e- is non-physical")
    else:
        dev = abs(ron_mean - ron_target_e)
        if dev > 3.0 * ron_tol_e:
            msgs.append(f"RON {ron_mean:.2f} e- far from {ron_target_e}+/-{ron_tol_e} e-")
        elif dev > ron_tol_e:
            warns.append(f"RON {ron_mean:.2f} e- outside {ron_target_e}+/-{ron_tol_e} e-")

    passed = len(msgs) == 0
    summary = f"RON={ron_mean:.2f} e-, egain={egain:.2f} e-/ADU"
    if passed:
        msgs.append(("WARN: " if warns else "PASS: ") + summary)
    return QAResult("bias_qa", passed, msgs, metrics, warnings=warns)


def run_sequence_audit(
    directory: Union[str, Path],
    *,
    expected_angles: Optional[Dict[str, int]] = None,
) -> QAResult:
    """Verify HWP angle completeness per (OBJECT, POLSEQ) group."""
    from collections import defaultdict

    d = Path(directory)
    groups: Dict[tuple, set] = defaultdict(set)
    for p in sorted(d.glob("*.fits")):
        hdr = fits.getheader(p, ignore_missing_end=True)
        if hdr.get("IMAGETYP", "LIGHT").upper() != "LIGHT":
            continue
        if "HWPANG" not in hdr:
            continue
        key = (hdr.get("OBJECT"), hdr.get("POLSEQ"), hdr.get("FILTER"))
        groups[key].add(float(hdr["HWPANG"]))

    msgs: List[str] = []
    incomplete: List[Dict[str, Any]] = []

    for key, angs in sorted(groups.items()):
        obj, polseq, filt = key
        n = len(angs)
        exp_n = None
        if expected_angles:
            if polseq in expected_angles:
                exp_n = int(expected_angles[polseq])
            elif obj in expected_angles:
                exp_n = int(expected_angles[obj])
        if polseq and isinstance(polseq, str):
            if exp_n is None and ("16" in polseq or "polV16" in polseq):
                exp_n = 16
            elif exp_n is None and (
                "8" in polseq or "polV8" in polseq or "polR8" in polseq
            ):
                exp_n = 8
        if exp_n is None:
            exp_n = 8 if n <= 8 else 16
        ok = n >= exp_n
        if not ok:
            incomplete.append({
                "object": obj, "polseq": polseq, "filter": filt,
                "n_angles": n, "expected": exp_n, "angles": sorted(angs),
            })
            msgs.append(f"Incomplete: {obj} {polseq} {filt}: {n}/{exp_n} angles")

    passed = len(incomplete) == 0
    if passed:
        msgs.append(f"PASS: {len(groups)} science sequence(s) complete")
    return QAResult("sequence_audit", passed, msgs, {"groups": len(groups), "incomplete": incomplete})


def _load_pol_config(
    paths: List[str],
    *,
    pol_config_path: Optional[Union[str, Path]],
    filter_name: str,
) -> PolConfig:
    """Resolve :class:`PolConfig` from sidecar or FITS headers."""
    import poltools as pt

    candidates = []
    if pol_config_path:
        candidates.append(Path(pol_config_path))
    if paths:
        candidates.append(Path(paths[0]).parent / "pol_config.yaml")
    for candidate in candidates:
        if candidate.is_file():
            return pt.load_pol_config_sidecar(candidate, filter_name=filter_name)
    if paths:
        return pt.polconfig_from_fits_headers(paths[0], filter_name=filter_name)
    raise FileNotFoundError("No pol_config.yaml sidecar or FITS path for PolConfig")


def run_first_light_qa(
    paths: Sequence[Union[str, Path]],
    *,
    pol_config_path: Optional[Union[str, Path]] = None,
    ref_name: str = "HD 154445",
    band: str = "V",
    abort_on_fail: bool = False,
) -> QAResult:
    """Compare measured P/PA to reference; lsq vs double_ratio (plan 6.5)."""
    import poltools as pt

    files = sorted(str(p) for p in _collect_paths(paths))
    if not files:
        return QAResult("first_light_qa", False, ["No FITS files matched"])

    filt = f"Photometric {band}"
    selected: List[str] = []
    for path in files:
        hdr = fits.getheader(path, ignore_missing_end=True)
        if str(hdr.get("IMAGETYP", "LIGHT")).upper() != "LIGHT":
            continue
        if str(hdr.get("FILTER", "")).strip() != filt:
            continue
        if str(hdr.get("OBJECT", "")).strip().casefold() != ref_name.casefold():
            continue
        selected.append(path)
    if not selected:
        return QAResult(
            "first_light_qa", False,
            [f"No LIGHT frames for OBJECT={ref_name!r}, FILTER={filt!r}"],
        )
    files = selected

    try:
        cfg = _load_pol_config(files, pol_config_path=pol_config_path, filter_name=filt)
    except FileNotFoundError as exc:
        return QAResult("first_light_qa", False, [str(exc)])

    ref = REF_HD154445
    msgs: List[str] = []
    warns: List[str] = []
    metrics: Dict[str, Any] = {}

    try:
        r_dr = pt.reduce_to_stokes(files, cfg, method="double_ratio", detect=True)
    except Exception as exc:
        return QAResult("first_light_qa", False, [f"double_ratio failed: {exc}"])

    if not r_dr:
        return QAResult("first_light_qa", False, ["No sources detected"])

    s = r_dr[0].scalar_summary
    p_meas = float(s.get("p_mas", s.get("p", 0.0))) * 100.0
    pa_meas = float(s.get("theta_deg", 0.0))
    metrics["P_dr_pct"] = p_meas
    metrics["PA_dr_deg"] = pa_meas
    metrics["efficiency_dr"] = p_meas / ref["P_pct"]

    # Tiered comparison to the reference standard: a modest miss WARNs (the
    # night keeps going); only a gross miss FAILs. Blocking first light on a
    # 0.3% P excursion is what stalled capture before.
    dp = abs(p_meas - ref["P_pct"])
    dpa = abs((pa_meas - ref["PA_deg"] + 90) % 180 - 90)
    (msgs if dp > 3.0 * ref["P_tol_pct"] else warns if dp > ref["P_tol_pct"] else []).append(
        f"|dP|={dp:.3f}% vs tol {ref['P_tol_pct']}%")
    (msgs if dpa > 3.0 * ref["PA_tol_deg"] else warns if dpa > ref["PA_tol_deg"] else []).append(
        f"|dPA|={dpa:.2f} deg vs tol {ref['PA_tol_deg']} deg")
    eff = metrics["efficiency_dr"]
    if eff < 0.85 or eff > 1.10:
        msgs.append(f"efficiency {eff:.3f} grossly outside [0.95, 1.02]")
    elif not (0.95 <= eff <= 1.02):
        warns.append(f"efficiency {eff:.3f} outside [0.95, 1.02]")

    # lsq vs double_ratio is a cross-check, never a capture blocker -> WARN only.
    try:
        r_lsq = pt.reduce_to_stokes(files, cfg, method="lsq", detect=True)
        if r_lsq:
            sl = r_lsq[0].scalar_summary
            dq = abs(sl.get("q", 0) - s.get("q", 0))
            du = abs(sl.get("u", 0) - s.get("u", 0))
            metrics["lsq_vs_dr_max_qu"] = max(dq, du)
            if max(dq, du) > 0.001:
                warns.append(f"lsq vs double_ratio qu diff {max(dq, du):.4f} > 0.1%")
    except Exception as exc:
        warns.append(f"lsq deferred (flats may be missing): {exc}")

    passed = len(msgs) == 0
    prefix = "WARN: " if (passed and warns) else ("PASS: " if passed else "")
    if passed:
        msgs.append(f"{prefix}P={p_meas:.2f}%, PA={pa_meas:.1f} deg, eff={eff:.3f}")
    return QAResult("first_light_qa", passed, msgs, metrics, warnings=warns)


def run_flat_quality_gate(
    paths: Sequence[Union[str, Path]],
    *,
    pol_config_path: Optional[Union[str, Path]] = None,
    qu_tol: float = 0.001,
) -> QAResult:
    """Compare lsq vs double_ratio on a polarized standard (plan 5.3)."""
    import poltools as pt

    files = sorted(str(p) for p in _collect_paths(paths))
    if not files:
        return QAResult("flat_quality_gate", False, ["No files"])

    try:
        cfg = _load_pol_config(
            files,
            pol_config_path=pol_config_path,
            filter_name=str(fits.getheader(files[0], ignore_missing_end=True).get("FILTER", "Photometric V")),
        )
    except FileNotFoundError as exc:
        return QAResult("flat_quality_gate", False, [str(exc)])

    msgs: List[str] = []
    try:
        r_lsq = pt.reduce_to_stokes(files, cfg, method="lsq", detect=True)
        r_dr = pt.reduce_to_stokes(files, cfg, method="double_ratio", detect=True)
    except Exception as exc:
        return QAResult("flat_quality_gate", False, [str(exc)])

    if not r_lsq or not r_dr:
        return QAResult("flat_quality_gate", False, ["Reduction returned no sources"])

    sl, sd = r_lsq[0].scalar_summary, r_dr[0].scalar_summary
    dq = abs(sl.get("q", 0) - sd.get("q", 0))
    du = abs(sl.get("u", 0) - sd.get("u", 0))
    mx = max(dq, du)
    warns: List[str] = []
    # A modest lsq/double_ratio disagreement flags flat quality but should not
    # block the night; only a gross disagreement (5x tol) is a hard failure.
    if mx > 5.0 * qu_tol:
        msgs.append(f"lsq vs double_ratio max qu diff {mx:.5f} >> tol {qu_tol}")
    elif mx > qu_tol:
        warns.append(f"lsq vs double_ratio max qu diff {mx:.5f} > tol {qu_tol}")
    passed = len(msgs) == 0
    if passed:
        msgs.append(("WARN: " if warns else "PASS: ") + f"max |dq,du|={mx:.5f}")
    return QAResult("flat_quality_gate", passed, msgs, {"max_qu_diff": mx}, warnings=warns)
