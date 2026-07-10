"""QA gate configuration and dispatch for the night session runner."""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from .qa_lib import (
    QAResult,
    run_bias_qa,
    run_first_light_qa,
    run_flat_quality_gate,
    run_sequence_audit,
)

logger = logging.getLogger(__name__)


@dataclass
class QAGate:
    handler: str
    after_target: Optional[str] = None
    after_cal: Optional[str] = None
    args: Dict[str, Any] = field(default_factory=dict)


_HANDLERS = {
    "bias_qa": run_bias_qa,
    "first_light_qa": run_first_light_qa,
    "flat_quality_gate": run_flat_quality_gate,
    "sequence_audit": run_sequence_audit,
}


def qa_gate_from_yaml(entry: Dict[str, Any]) -> QAGate:
    return QAGate(
        handler=str(entry["handler"]),
        after_target=entry.get("after_target"),
        after_cal=entry.get("after_cal"),
        args=dict(entry.get("args") or {}),
    )


def dispatch_qa_gate(
    gate: QAGate,
    *,
    session_dir: Path,
    target_name: Optional[str] = None,
    cal_brick: Optional[str] = None,
) -> QAResult:
    """Run one QA gate handler against the session data directory."""
    fn = _HANDLERS.get(gate.handler)
    if fn is None:
        return QAResult(gate.handler, False, [f"Unknown handler {gate.handler!r}"])

    args = dict(gate.args)
    abort = bool(args.pop("abort_on_fail", False))

    if gate.handler == "bias_qa":
        result = fn(session_dir, **args)
    elif gate.handler == "sequence_audit":
        result = fn(session_dir, **args)
    elif gate.handler in ("first_light_qa", "flat_quality_gate"):
        slug = (target_name or "").replace(" ", "_").replace("+", "")
        paths = sorted(session_dir.glob(f"*{slug}*")) if slug else []
        if not paths:
            paths = sorted(session_dir.glob("*.fits"))
        sidecar = session_dir / "pol_config.yaml"
        result = fn(
            paths,
            pol_config_path=sidecar if sidecar.exists() else None,
            **args,
        )
    else:
        result = fn(session_dir, **args)

    who = target_name or cal_brick
    log = logger.error if result.level == "FAIL" else (
        logger.warning if result.level == "WARN" else logger.info)
    log("QA gate %s [%s] (%s): %s", gate.handler, result.level, who, result.messages)
    for w in result.warnings:
        logger.warning("QA gate %s WARN (%s): %s", gate.handler, who, w)
    if not result.passed and abort:
        raise RuntimeError(f"QA gate {gate.handler} failed: {result.messages}")
    return result


def gates_after_target(gates: List[QAGate], target_name: str) -> List[QAGate]:
    return [g for g in gates if g.after_target == target_name]


def gates_after_cal(gates: List[QAGate], brick_name: str) -> List[QAGate]:
    return [g for g in gates if g.after_cal == brick_name]
