"""Building-block night plans: a small declarative layer over NightSessionConfig.

Instead of hand-typing FramePlan / TargetPlan dataclasses, describe a night as a
palette of reusable *bricks* plus a *plan* that lays them under targets. A loader
expands the bricks into the existing NightSessionConfig, so the run engine
(`run_night_session`) is unchanged.

Two files (see ``night_plans/``):

* a shared ``palette.yaml`` -- reusable ``bricks:`` + an optional target
  ``catalog:`` (RA/Dec), committed and reused night to night;
* a per-night file -- ``uses: palette.yaml`` to import it, then a ``plan:`` that
  lays bricks, with optional per-night ``bricks:`` and ``override:``.

Brick types (the ``type:`` field)::

    stack        filter, exp, n [, gain, offset, binx, biny]        -> 1 LIGHT stack
    pol_seq      filter, angles|hwp, exp, n [, retardance=180, ...]  -> 1 LIGHT stack per HWP angle
    filter_loop  filters:[...], exp, n [, gain, offset]             -> 1 LIGHT stack per filter
    cal          frame:DARK|BIAS|FLAT, exp, n [, filter]            -> 1 calibration stack
    cal_multi    frames:[{frame, exp, n}, ...]                       -> multiple cal stacks
    pol_flat     filter, angles, exp, n [, imagetyp=FLAT]            -> polarimetric twilight flats

Lay an item as a brick name (``polV8``) or a single-key map with inline
overrides (``{polV8: {exp: 45}}``). A plan entry is ``TargetName: [bricks...]``
(coords from the catalog), ``{target: X, ra: .., dec: .., lay: [...]}`` for
inline coords, or ``cal: [bricks...]`` for calibration (no pointing).
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import yaml

from .config import AlpacaConfig, Pwi4Config
from .logging import LoggingConfig
from .night_session import FramePlan, NightSessionConfig, TargetPlan
from .qa_gates import QAGate, qa_gate_from_yaml
from .session_context import SessionCaptureContext, session_context_from_yaml
from .startup import StartupConfig

logger = logging.getLogger(__name__)

# Fields on a stack/pol brick that map straight onto FramePlan capture params.
_CAPTURE_KEYS = ("gain", "offset", "binx", "biny", "numx", "numy", "readout_mode")


class NightPlanError(ValueError):
    """Raised for a malformed or unresolvable night plan."""


# --------------------------------------------------------------------------- #
# HWP angle sequences
# --------------------------------------------------------------------------- #
def hwp_angles(spec: Union[int, List[float]]) -> List[float]:
    """Resolve a pol_seq angle spec into a list of HWP angles [deg].

    ``spec`` is either an integer count -> that many equally spaced 22.5 deg
    steps (the standard sampling; 4 -> {0,22.5,45,67.5}, 8 -> 0..157.5,
    16 -> 0..337.5), or an explicit list of angles.
    """
    if isinstance(spec, bool):  # guard: bool is an int subclass
        raise NightPlanError("HWP angle spec must be a count or list, not a bool")
    if isinstance(spec, int):
        return [round(i * 22.5, 3) for i in range(spec)]
    if isinstance(spec, (list, tuple)):
        return [float(a) for a in spec]
    raise NightPlanError(f"Bad HWP angle spec: {spec!r} (use an int count or a list)")


def _check_pol_sampling(angles: List[float]) -> None:
    """Reject angle sets that can't constrain q and u.

    A dual-beam HWP polarimeter needs >= 4 angles to cancel first-order
    instrumental effects (Patat & Taubenberger 2011), and the {cos4t, sin4t}
    design matrix must be well conditioned. Mirrors the check in
    ``scripts/hwp_modulation_test.py``.
    """
    if len(angles) < 4:
        raise NightPlanError(
            f"pol_seq needs >= 4 HWP angles (got {len(angles)}); >= 4 are required "
            "to cancel first-order instrumental effects (Patat & Taubenberger 2011)."
        )
    import numpy as np

    th = np.deg2rad(np.asarray(angles, dtype=float))
    A = np.column_stack([np.cos(4 * th), np.sin(4 * th)])
    s = np.linalg.svd(A, compute_uv=False)
    if float(s.min()) < 1e-6 or float(s.max() / s.min()) > 20.0:
        raise NightPlanError(
            f"pol_seq HWP angles {angles} alias the 4-theta modulation "
            "(q/u under-constrained). Use 22.5 deg steps (angles: 4|8|16)."
        )


# --------------------------------------------------------------------------- #
# Brick -> FramePlan expansion
# --------------------------------------------------------------------------- #
def _capture_kwargs(spec: Dict[str, Any]) -> Dict[str, Any]:
    return {k: spec[k] for k in _CAPTURE_KEYS if k in spec}


def _expand_brick(name: str, spec: Dict[str, Any]) -> List[FramePlan]:
    """Expand one resolved brick spec into FramePlan(s)."""
    btype = spec.get("type")
    if btype is None:
        raise NightPlanError(f"Brick {name!r} has no 'type'")
    n = int(spec.get("n", 1))
    common = _capture_kwargs(spec)

    if btype == "stack":
        return [FramePlan(
            frame_type="LIGHT", exposure_s=float(spec["exp"]), count=n,
            filter=spec.get("filter"), object_name=spec.get("object_name"),
            source_brick=name, **common,
        )]

    if btype == "filter_loop":
        filters = spec.get("filters")
        if not filters:
            raise NightPlanError(f"filter_loop brick {name!r} needs a 'filters' list")
        return [FramePlan(
            frame_type="LIGHT", exposure_s=float(spec["exp"]), count=n,
            filter=f, source_brick=name, **common,
        ) for f in filters]

    if btype == "pol_seq":
        angles = hwp_angles(spec.get("hwp", spec.get("angles")))
        _check_pol_sampling(angles)
        retard = float(spec.get("retardance", 180.0))
        seq_id = spec.get("pol_seq_id", name)
        frames = []
        for i, a in enumerate(angles):
            frames.append(FramePlan(
                frame_type="LIGHT", exposure_s=float(spec["exp"]), count=n,
                filter=spec.get("filter"), hwp_angle_deg=float(a),
                retardance_deg=retard, pol_seq_id=seq_id, pol_seq_index=i,
                source_brick=name, **common,
            ))
        return frames

    if btype == "pol_flat":
        angles = hwp_angles(spec.get("angles", 4))
        _check_pol_sampling(angles)
        retard = float(spec.get("retardance", 180.0))
        seq_id = spec.get("pol_seq_id", name)
        obj = spec.get("object_name", "TwilightFlat")
        frame = str(spec.get("imagetyp", "FLAT")).upper()
        frames = []
        for i, a in enumerate(angles):
            frames.append(FramePlan(
                frame_type=frame, exposure_s=float(spec["exp"]), count=n,
                filter=spec.get("filter"), hwp_angle_deg=float(a),
                retardance_deg=retard, pol_seq_id=seq_id, pol_seq_index=i,
                object_name=obj, source_brick=name, **common,
            ))
        return frames

    if btype == "cal_multi":
        subframes = spec.get("frames")
        if not subframes:
            raise NightPlanError(f"cal_multi brick {name!r} needs a 'frames' list")
        out: List[FramePlan] = []
        for sub in subframes:
            frame = str(sub.get("frame", "DARK")).upper()
            out.append(FramePlan(
                frame_type=frame,
                exposure_s=float(sub.get("exp", 0.0)),
                count=int(sub.get("n", 1)),
                filter=sub.get("filter"),
                source_brick=name,
                **_capture_kwargs({**spec, **sub}),
            ))
        return out

    if btype == "cal":
        frame = str(spec.get("frame", "DARK")).upper()
        return [FramePlan(
            frame_type=frame, exposure_s=float(spec.get("exp", 0.0)), count=n,
            filter=spec.get("filter"), source_brick=name, **common,
        )]

    raise NightPlanError(f"Unknown brick type {btype!r} in {name!r}")


# --------------------------------------------------------------------------- #
# Plan resolution
# --------------------------------------------------------------------------- #
def _resolve_spec(
    name: str,
    inline: Optional[Dict[str, Any]],
    palette: Dict[str, Dict[str, Any]],
    overrides: Dict[str, Dict[str, Any]],
) -> Dict[str, Any]:
    if name not in palette:
        raise NightPlanError(f"Brick {name!r} not in palette (have: {sorted(palette)})")
    spec = dict(palette[name])
    spec.update(overrides.get(name, {}))   # per-night top-level override:
    if inline:
        spec.update(inline)                # lay-site {name: {overrides}}
    return spec


def _lay_item(item: Union[str, Dict[str, Any]]) -> Tuple[str, Optional[Dict[str, Any]]]:
    """Normalise a lay item to (brick_name, inline_overrides)."""
    if isinstance(item, str):
        return item, None
    if isinstance(item, dict) and len(item) == 1:
        (name, override), = item.items()
        if override is not None and not isinstance(override, dict):
            raise NightPlanError(f"Inline override for {name!r} must be a map")
        return name, override
    raise NightPlanError(f"Bad lay item: {item!r} (use a name or {{name: {{overrides}}}})")


def _expand_lay(
    lay: List[Any],
    palette: Dict[str, Dict[str, Any]],
    overrides: Dict[str, Dict[str, Any]],
) -> List[FramePlan]:
    frames: List[FramePlan] = []
    for item in lay:
        name, inline = _lay_item(item)
        frames.extend(_expand_brick(name, _resolve_spec(name, inline, palette, overrides)))
    return frames


def _load_yaml(path: Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as fh:
        doc = yaml.safe_load(fh)
    if not isinstance(doc, dict):
        raise NightPlanError(f"{path} must be a YAML mapping")
    return doc


def build_config(
    doc: Dict[str, Any],
    palette: Dict[str, Dict[str, Any]],
    catalog: Dict[str, Dict[str, Any]],
    startup: StartupConfig,
) -> NightSessionConfig:
    """Assemble a NightSessionConfig from a parsed plan document + palette."""
    overrides = doc.get("override", {}) or {}
    targets: List[TargetPlan] = []
    cal_before: List[FramePlan] = []
    cal_after: List[FramePlan] = []
    seen_targets = False

    for entry in doc.get("plan", []):
        if not isinstance(entry, dict) or len(entry) != 1 and "target" not in entry:
            raise NightPlanError(f"Bad plan entry: {entry!r}")

        if "cal" in entry:
            expanded = _expand_lay(entry["cal"], palette, overrides)
            if seen_targets:
                cal_after.extend(expanded)
            else:
                cal_before.extend(expanded)
            continue

        seen_targets = True

        if "target" in entry:
            name = entry["target"]
            lay = entry.get("lay", [])
            ra, dec = entry.get("ra"), entry.get("dec")
            alt, az = entry.get("alt"), entry.get("az")
            track = bool(entry.get("track", True))
        else:
            (name, lay), = entry.items()
            coords = catalog.get(name, {})
            ra, dec = coords.get("ra"), coords.get("dec")
            alt, az = coords.get("alt"), coords.get("az")
            track = bool(coords.get("track", True))
            if ra is None and alt is None:
                raise NightPlanError(
                    f"Target {name!r} not in catalog and has no inline coords; "
                    "add it to the catalog or use the {target:, ra:, dec:, lay:} form."
                )

        targets.append(TargetPlan(
            name=str(name),
            ra_hours=ra, dec_deg=dec, alt_deg=alt, az_deg=az,
            track=track,
            frames=_expand_lay(lay, palette, overrides),
        ))

    qa_gates: List[QAGate] = []
    for g in doc.get("qa_gates") or []:
        qa_gates.append(qa_gate_from_yaml(g))

    return NightSessionConfig(
        startup=startup,
        targets=targets,
        calibration_before=cal_before,
        calibration_after=cal_after,
        calibration_stage=doc.get("calibrate_stage", "after"),
        session_name=str(doc["session"]) if doc.get("session") is not None else None,
        base_data_dir=doc.get("base_data_dir", "data"),
        observer=doc.get("observer"),
        telescope=doc.get("telescope"),
        observatory=doc.get("observatory"),
        capture_context=session_context_from_yaml(doc.get("camera")),
        naming=doc.get("naming", "polite"),
        qa_gates=qa_gates,
        catalog=dict(catalog),
    )


def load_night_plan(
    path: Union[str, Path],
    startup: Optional[StartupConfig] = None,
) -> NightSessionConfig:
    """Load a per-night plan file (resolving its ``uses:`` palette) -> config.

    ``startup`` defaults to a StartupConfig built from the project's
    ``user_config`` (QHY/EFW/Pyxis Alpaca + PWI4), with the session name set from
    the plan file.
    """
    path = Path(path).expanduser().resolve()
    doc = _load_yaml(path)

    palette: Dict[str, Dict[str, Any]] = {}
    catalog: Dict[str, Dict[str, Any]] = {}
    uses = doc.get("uses")
    if uses:
        pal_path = (path.parent / uses).resolve()
        pal_doc = _load_yaml(pal_path)
        palette.update(pal_doc.get("bricks", {}) or {})
        catalog.update(pal_doc.get("catalog", {}) or {})
    # Per-night bricks/catalog add to or override the shared palette.
    palette.update(doc.get("bricks", {}) or {})
    catalog.update(doc.get("catalog", {}) or {})

    if startup is None:
        startup = _default_startup(doc.get("session"))
    config = build_config(doc, palette, catalog, startup)
    # Merge plan-level observer/telescope into capture context when set.
    ctx = config.capture_context
    if ctx is not None:
        updates = {}
        if config.observer and not ctx.observer:
            updates["observer"] = config.observer
        if config.telescope and ctx.telescope == "CDK20" and doc.get("telescope"):
            updates["telescope"] = config.telescope
        if config.observatory and ctx.observatory == "Julian, CA" and doc.get("observatory"):
            updates["observatory"] = config.observatory
        if updates:
            from dataclasses import replace as dc_replace
            config = dc_replace(config, capture_context=dc_replace(ctx, **updates))
    return config


def _default_startup(session: Optional[Any]) -> StartupConfig:
    try:
        from .user_config import ALPACA_CONFIG, PWI4_CONFIG
    except Exception:  # pragma: no cover - user_config may be absent
        ALPACA_CONFIG, PWI4_CONFIG = AlpacaConfig(), Pwi4Config()
    session_name = str(session) if session is not None else None
    return StartupConfig(
        alpaca=ALPACA_CONFIG,
        pwi4=PWI4_CONFIG,
        logging=LoggingConfig(session_name=session_name),
    )


# --------------------------------------------------------------------------- #
# Dry-run preview
# --------------------------------------------------------------------------- #
def describe(config: NightSessionConfig) -> str:
    """Human-readable expansion of a plan: tree of frames + exposure totals."""
    lines: List[str] = []
    total_frames = 0
    total_exp = 0.0

    def _fmt_frames(frames: List[FramePlan], indent: str) -> Tuple[int, float]:
        nf, te = 0, 0.0
        for f in frames:
            tag = f.frame_type.upper()
            filt = f" {f.filter}" if f.filter is not None else ""
            hwp = f" hwp={f.hwp_angle_deg:g}deg" if f.hwp_angle_deg is not None else ""
            lines.append(f"{indent}{tag}{filt}{hwp}  {f.exposure_s:g}s x{f.count}")
            nf += f.count
            te += f.exposure_s * f.count
        return nf, te

    for t in config.targets:
        coord = ""
        if t.ra_hours is not None:
            coord = f"  (ra={t.ra_hours:g}h dec={t.dec_deg:g}deg)"
        elif t.alt_deg is not None:
            coord = f"  (alt={t.alt_deg:g} az={t.az_deg:g})"
        lines.append(f"* {t.name}{coord}")
        nf, te = _fmt_frames(t.frames, "    ")
        total_frames += nf
        total_exp += te

    if config.calibration_before or config.calibration_after:
        lines.append(f"* calibration (before={len(config.calibration_before)} "
                     f"after={len(config.calibration_after)}, "
                     f"stage={config.calibration_stage})")
        if config.calibration_before:
            lines.append("  [before targets]")
            nf, te = _fmt_frames(config.calibration_before, "    ")
            total_frames += nf
            total_exp += te
        if config.calibration_after:
            lines.append("  [after targets]")
            nf, te = _fmt_frames(config.calibration_after, "    ")
            total_frames += nf
            total_exp += te

    if config.qa_gates:
        lines.append(f"* qa_gates: {len(config.qa_gates)}")
        for g in config.qa_gates:
            trig = g.after_target or g.after_cal or "?"
            lines.append(f"    {g.handler} after {trig}")

    lines.append("")
    if config.capture_context is not None:
        ctx = config.capture_context
        lines.append("camera:")
        lines.append(f"  readout_mode={ctx.readout_mode} ({ctx.readout_mode_name})")
        lines.append(f"  gain={ctx.gain_setting}  offset={ctx.offset_setting}")
        lines.append(f"  egain={ctx.egain_e_per_adu} e-/ADU  ron={ctx.ron_e} e-")
        lines.append(f"  cooler_setpoint={ctx.cooler_setpoint_c} C")
        lines.append(f"  naming={config.naming}  base_data_dir={config.base_data_dir}")
        lines.append("")
    lines.append(f"total: {total_frames} frames, {total_exp:g}s "
                 f"({total_exp / 3600.0:.2f}h) of open-shutter exposure "
                 "(excludes readout/slew/HWP-move overhead)")
    return "\n".join(lines)
