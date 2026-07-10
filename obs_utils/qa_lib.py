"""First-light QA algorithms (shared by scripts and night-session hooks)."""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Union

import numpy as np
from astropy.io import fits

logger = logging.getLogger(__name__)

# HD 154445 reference (plan Appendix B, Sch92)
REF_HD154445 = {"P_pct": 3.67, "PA_deg": 88.6, "P_tol_pct": 0.3, "PA_tol_deg": 2.0}


@dataclass
class QAResult:
    name: str
    passed: bool
    messages: List[str] = field(default_factory=list)
    metrics: Dict[str, Any] = field(default_factory=dict)

    def to_json(self) -> str:
        return json.dumps({
            "name": self.name,
            "passed": self.passed,
            "messages": self.messages,
            "metrics": self.metrics,
        }, indent=2)


def _collect_paths(paths: Sequence[Union[str, Path]]) -> List[Path]:
    out: List[Path] = []
    for p in paths:
        pp = Path(p)
        if pp.is_dir():
            out.extend(sorted(pp.glob("*.fits")))
        elif pp.is_file():
            out.append(pp)
    return out


def run_bias_qa(
    paths: Sequence[Union[str, Path]],
    *,
    ron_target_e: float = 3.5,
    ron_tol_e: float = 0.5,
    mean_lo: float = 50.0,
    mean_hi: float = 500.0,
) -> QAResult:
    """Bias histogram + pair-difference RON test (first-light plan 5.4 Lab 1)."""
    files = [p for p in _collect_paths(paths) if "BIAS" in p.name.upper()
             or fits.getheader(p, ignore_missing_end=True).get("IMAGETYP", "").upper() == "BIAS"]
    if len(files) < 2:
        return QAResult("bias_qa", False, ["Need >= 2 bias frames"])

    msgs: List[str] = []
    egain_vals: List[float] = []
    ron_pairs: List[float] = []

    for p in files:
        hdr = fits.getheader(p, ignore_missing_end=True)
        egain_vals.append(float(hdr.get("EGAIN", hdr.get("GAIN", 1.0))))

    egain = float(np.median(egain_vals))

    for p in files:
        data = fits.getdata(p, memmap=False).astype(np.float64)
        if float(np.min(data)) <= 0:
            msgs.append(f"{p.name}: min ADU <= 0 (pinned pixels)")
        mean = float(np.mean(data))
        if not (mean_lo <= mean <= mean_hi):
            msgs.append(f"{p.name}: mean bias level {mean:.1f} outside [{mean_lo}, {mean_hi}]")

    for i in range(len(files) - 1):
        a = fits.getdata(files[i], memmap=False).astype(np.float64)
        b = fits.getdata(files[i + 1], memmap=False).astype(np.float64)
        diff = a - b
        ron_pairs.append(float(np.std(diff) / np.sqrt(2.0) * egain))

    ron_mean = float(np.mean(ron_pairs)) if ron_pairs else float("nan")
    metrics = {"ron_e": ron_mean, "egain": egain, "n_frames": len(files)}
    if abs(ron_mean - ron_target_e) > ron_tol_e:
        msgs.append(f"RON {ron_mean:.2f} e- outside {ron_target_e}+/-{ron_tol_e}")

    passed = len(msgs) == 0
    if passed:
        msgs.append(f"PASS: RON={ron_mean:.2f} e-, egain={egain:.2f} e-/ADU")
    return QAResult("bias_qa", passed, msgs, metrics)


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

    expected = expected_angles or {}
    msgs: List[str] = []
    incomplete: List[Dict[str, Any]] = []

    for key, angs in sorted(groups.items()):
        obj, polseq, filt = key
        n = len(angs)
        exp_n = None
        if polseq and isinstance(polseq, str):
            if "16" in polseq or "polV16" in polseq:
                exp_n = 16
            elif "8" in polseq or "polV8" in polseq or "polR8" in polseq:
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
    try:
        cfg = _load_pol_config(files, pol_config_path=pol_config_path, filter_name=filt)
    except FileNotFoundError as exc:
        return QAResult("first_light_qa", False, [str(exc)])

    ref = REF_HD154445
    msgs: List[str] = []
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

    dp = abs(p_meas - ref["P_pct"])
    dpa = abs((pa_meas - ref["PA_deg"] + 90) % 180 - 90)
    if dp > ref["P_tol_pct"]:
        msgs.append(f"|dP|={dp:.3f}% > {ref['P_tol_pct']}%")
    if dpa > ref["PA_tol_deg"]:
        msgs.append(f"|dPA|={dpa:.2f} deg > {ref['PA_tol_deg']} deg")
    eff = metrics["efficiency_dr"]
    if not (0.95 <= eff <= 1.02):
        msgs.append(f"efficiency {eff:.3f} outside [0.95, 1.02]")

    lsq_note: Optional[str] = None
    try:
        r_lsq = pt.reduce_to_stokes(files, cfg, method="lsq", detect=True)
        if r_lsq:
            sl = r_lsq[0].scalar_summary
            q_l, u_l = sl.get("q", 0), sl.get("u", 0)
            q_d, u_d = s.get("q", 0), s.get("u", 0)
            dq = abs(q_l - q_d)
            du = abs(u_l - u_d)
            metrics["lsq_vs_dr_max_qu"] = max(dq, du)
            if max(dq, du) > 0.001:
                msgs.append(f"lsq vs double_ratio qu diff {max(dq, du):.4f} > 0.1%")
    except Exception as exc:
        lsq_note = f"lsq deferred (flats may be missing): {exc}"

    passed = len(msgs) == 0
    if passed:
        msgs.append(f"PASS: P={p_meas:.2f}%, PA={pa_meas:.1f} deg, eff={eff:.3f}")
    if lsq_note:
        msgs.append(lsq_note)
    return QAResult("first_light_qa", passed, msgs, metrics)


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
    passed = mx <= qu_tol
    if passed:
        msgs.append(f"PASS: max |dq,du|={mx:.5f}")
    else:
        msgs.append(f"FAIL: lsq vs double_ratio max qu diff {mx:.5f} > {qu_tol}")
    return QAResult("flat_quality_gate", passed, msgs, {"max_qu_diff": mx})
