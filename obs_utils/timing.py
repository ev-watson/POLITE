"""
Exposure timing provenance for POLITE FITS headers.

Observatory (Windows): query ``w32tm /query /status`` once at session start and
stamp the resulting snapshot on every frame that night.

Lab (macOS): skip NTP discipline checks; stamp ``TIMESRC='LAB-LOCAL'`` with a
higher adopted ``TIMEUNC``.
"""

from __future__ import annotations

import logging
import re
import subprocess
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import List, Optional

from astropy.io import fits


logger = logging.getLogger(__name__)


@dataclass
class TimingConfig:
    """Session-level timing policy (see reference-sheets/hardware/timing-policy.html)."""

    enabled: bool = True
    # ``auto``: Windows -> observatory (w32tm); macOS/Linux -> lab snapshot.
    mode: str = "auto"
    warn_offset_s: float = 0.5
    abort_offset_s: float = 2.0
    abort_on_failure: bool = False
    max_sync_age_s: float = 86400.0
    observatory_timeunc_s: float = 0.5
    lab_timeunc_s: float = 1.5
    w32tm_command: tuple[str, ...] = ("w32tm", "/query", "/status")


@dataclass
class ParsedW32tmStatus:
    source: Optional[str] = None
    phase_offset_s: Optional[float] = None
    last_sync_local: Optional[datetime] = None
    sync_age_s: Optional[float] = None
    raw: str = ""


@dataclass
class NtpStatus:
    """One-shot session snapshot of host clock discipline at startup.

    Queried once when the night session starts, then copied unchanged onto every
    FITS frame. ``DATE-OBS`` still comes from Alpaca per exposure; this block
    records the clock quality in effect for the whole night.
    """

    timesrc: str
    timeunc_s: float
    timeref: Optional[str] = None
    ntpoffs_s: Optional[float] = None
    ntpage_s: Optional[float] = None
    last_sync_utc: Optional[datetime] = None
    warnings: List[str] = field(default_factory=list)
    raw_status: Optional[str] = None

    def to_provenance(self) -> TimingProvenance:
        return TimingProvenance(
            timesrc=self.timesrc,
            timeunc=self.timeunc_s,
            timeref=self.timeref,
            ntpoffs=self.ntpoffs_s,
            ntpage=self.ntpage_s,
        )


@dataclass
class TimingProvenance:
    timesrc: str
    timeunc: float
    timeref: Optional[str] = None
    ntpoffs: Optional[float] = None
    ntpage: Optional[float] = None


@dataclass
class FilterWheelState:
    fwpos: int
    filtrdy: bool


class TimingError(RuntimeError):
    """Raised when timing checks fail and ``abort_on_failure`` is enabled."""


_RE_SOURCE = re.compile(r"^Source:\s*(.+)$", re.MULTILINE | re.IGNORECASE)
_RE_PHASE_OFFSET = re.compile(
    r"^Phase Offset:\s*([+-]?\d+(?:\.\d+)?)\s*s",
    re.MULTILINE | re.IGNORECASE,
)
_RE_LAST_SYNC = re.compile(
    r"^Last Successful Sync Time:\s*(.+)$",
    re.MULTILINE | re.IGNORECASE,
)


def _resolve_mode(config: TimingConfig) -> str:
    mode = (config.mode or "auto").lower()
    if mode == "auto":
        return "observatory" if sys.platform == "win32" else "lab"
    if mode not in ("observatory", "lab"):
        raise ValueError(f"Unknown timing mode: {config.mode!r}")
    return mode


def query_w32tm_status(command: tuple[str, ...] = TimingConfig.w32tm_command) -> str:
    result = subprocess.run(
        list(command),
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        stderr = (result.stderr or "").strip()
        stdout = (result.stdout or "").strip()
        detail = stderr or stdout or f"exit code {result.returncode}"
        raise RuntimeError(f"w32tm failed: {detail}")
    return result.stdout


def _parse_windows_sync_time(text: str) -> Optional[datetime]:
    """Best-effort parse of US-style ``w32tm`` sync timestamp (local wall clock)."""
    text = text.strip()
    formats = (
        "%m/%d/%Y %I:%M:%S %p",
        "%m/%d/%Y %H:%M:%S",
        "%d/%m/%Y %H:%M:%S",
        "%Y-%m-%d %H:%M:%S",
    )
    for fmt in formats:
        try:
            return datetime.strptime(text, fmt)
        except ValueError:
            continue
    return None


def parse_w32tm_status(raw: str, *, now: Optional[datetime] = None) -> ParsedW32tmStatus:
    """Parse English ``w32tm /query /status`` output."""
    now_utc = now or datetime.now(timezone.utc)
    parsed = ParsedW32tmStatus(raw=raw)

    m = _RE_SOURCE.search(raw)
    if m:
        parsed.source = m.group(1).strip()

    m = _RE_PHASE_OFFSET.search(raw)
    if m:
        parsed.phase_offset_s = float(m.group(1))

    m = _RE_LAST_SYNC.search(raw)
    if m:
        sync_local = _parse_windows_sync_time(m.group(1))
        if sync_local is not None:
            parsed.last_sync_local = sync_local
            sync_utc = sync_local.replace(tzinfo=timezone.utc)
            parsed.sync_age_s = max(0.0, (now_utc - sync_utc).total_seconds())

    return parsed


def evaluate_w32tm_status(
    parsed: ParsedW32tmStatus,
    config: TimingConfig,
) -> List[str]:
    warnings: List[str] = []

    if parsed.phase_offset_s is None:
        warnings.append("w32tm status missing Phase Offset; NTPOFFS not recorded")
    elif abs(parsed.phase_offset_s) > config.warn_offset_s:
        warnings.append(
            f"NTP phase offset {parsed.phase_offset_s:.3f} s exceeds warn threshold "
            f"{config.warn_offset_s} s"
        )

    if parsed.sync_age_s is None:
        warnings.append("w32tm status missing Last Successful Sync Time; NTPAGE not recorded")
    elif parsed.sync_age_s > config.max_sync_age_s:
        warnings.append(
            f"Last NTP sync age {parsed.sync_age_s:.0f} s exceeds max "
            f"{config.max_sync_age_s:.0f} s"
        )

    abort = False
    if parsed.phase_offset_s is not None and abs(parsed.phase_offset_s) > config.abort_offset_s:
        warnings.append(
            f"NTP phase offset {parsed.phase_offset_s:.3f} s exceeds abort threshold "
            f"{config.abort_offset_s} s"
        )
        abort = True
    if parsed.sync_age_s is not None and parsed.sync_age_s > config.max_sync_age_s:
        abort = True

    if abort and config.abort_on_failure:
        raise TimingError("; ".join(warnings))

    return warnings


def lab_timing_snapshot(config: TimingConfig) -> NtpStatus:
    return NtpStatus(
        timesrc="LAB-LOCAL",
        timeunc_s=config.lab_timeunc_s,
        warnings=["Lab mode: host NTP discipline not checked (no w32tm on this platform)"],
    )


def observatory_timing_snapshot(config: TimingConfig) -> NtpStatus:
    raw = query_w32tm_status(config.w32tm_command)
    parsed = parse_w32tm_status(raw)
    warnings = evaluate_w32tm_status(parsed, config)
    for msg in warnings:
        logger.warning("Timing: %s", msg)

    last_sync_utc = None
    if parsed.last_sync_local is not None:
        last_sync_utc = parsed.last_sync_local.replace(tzinfo=timezone.utc)

    return NtpStatus(
        timesrc="WIN-NTP",
        timeunc_s=config.observatory_timeunc_s,
        timeref=parsed.source,
        ntpoffs_s=parsed.phase_offset_s,
        ntpage_s=parsed.sync_age_s,
        last_sync_utc=last_sync_utc,
        warnings=list(warnings),
        raw_status=raw,
    )


def acquire_session_timing_snapshot(config: TimingConfig) -> NtpStatus:
    """Capture the once-per-session NTP snapshot used for all FITS headers."""
    if not config.enabled:
        return NtpStatus(
            timesrc="DISABLED",
            timeunc_s=config.observatory_timeunc_s,
            warnings=["Timing checks disabled"],
        )

    mode = _resolve_mode(config)
    if mode == "lab":
        status = lab_timing_snapshot(config)
        for msg in status.warnings:
            logger.info("Timing: %s", msg)
        return status

    try:
        return observatory_timing_snapshot(config)
    except Exception as exc:
        msg = f"w32tm query failed: {exc}"
        logger.warning("Timing: %s", msg)
        if config.abort_on_failure:
            raise TimingError(msg) from exc
        return NtpStatus(
            timesrc="WIN-NTP",
            timeunc_s=config.observatory_timeunc_s,
            warnings=[msg],
        )


def stamp_timing_cards(
    hdr: fits.Header,
    provenance: TimingProvenance,
    filter_state: Optional[FilterWheelState] = None,
) -> None:
    hdr["TIMESRC"] = (provenance.timesrc, "Timing source class")
    hdr["TIMEUNC"] = (provenance.timeunc, "1-sigma timing uncertainty [s]")
    if provenance.timeref:
        hdr["TIMEREF"] = (provenance.timeref, "NTP reference in use")
    if provenance.ntpoffs is not None:
        hdr["NTPOFFS"] = (provenance.ntpoffs, "NTP phase offset at session start [s]")
    if provenance.ntpage is not None:
        hdr["NTPAGE"] = (provenance.ntpage, "Seconds since last NTP sync at session start")
    if filter_state is not None:
        hdr["FWPOS"] = (filter_state.fwpos, "Filter-wheel slot index")
        hdr["FILTRDY"] = (
            "T" if filter_state.filtrdy else "F",
            "Filter wheel ready before exposure",
        )
