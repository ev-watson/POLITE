"""Rich terminal display for the night-session runner.

Provides :class:`NightReporter`, a context manager that renders a live
progress bar for the whole night plus styled banners, per-block headers and
QA summaries. It degrades gracefully: if ``rich`` is unavailable or output is
not a TTY, it becomes a quiet no-op and ordinary logging carries the night.

The palette is a set of muted dark pastels (POLITE house style).
"""
from __future__ import annotations

import logging
import sys
from contextlib import contextmanager
from typing import List, Optional

logger = logging.getLogger(__name__)

# --- dark pastel palette ----------------------------------------------------
# taupe (dim/borders), slate (primary/bar), stone (secondary text),
# mauve (highlight), plum (titles), moss (success/PASS).
PALETTE = {
    "taupe": "#4D413B",
    "slate": "#586480",
    "stone": "#8A826D",
    "mauve": "#776C82",
    "plum": "#483A54",
    "moss": "#414F40",
}

_LEVEL_STYLE = {"PASS": PALETTE["moss"], "WARN": PALETTE["mauve"], "FAIL": "red"}

try:  # rich is optional; the reporter no-ops without it
    from rich.console import Console
    from rich.logging import RichHandler
    from rich.panel import Panel
    from rich.progress import (
        BarColumn,
        MofNCompleteColumn,
        Progress,
        TextColumn,
        TimeElapsedColumn,
        TimeRemainingColumn,
    )
    from rich.table import Table
    from rich.text import Text

    _HAVE_RICH = True
except Exception:  # pragma: no cover - exercised only where rich is absent
    _HAVE_RICH = False


def total_frame_count(frame_plans) -> int:
    """Sum of individual exposures across a list of FramePlan objects."""
    return sum(int(getattr(p, "count", 0)) for p in (frame_plans or []))


class NightReporter:
    """Live night-progress display. Use as a context manager.

    All methods are safe to call whether or not rich/TTY is available; when it
    is not, they either no-op or fall back to plain logging.
    """

    def __init__(self, total_frames: int, *, title: str = "POLITE night", enabled: Optional[bool] = None):
        self.total_frames = max(0, int(total_frames))
        self.title = title
        self._done = 0
        self._block: Optional[str] = None
        self._saved_handlers: List[logging.Handler] = []
        if enabled is None:
            enabled = _HAVE_RICH and sys.stderr.isatty()
        self.enabled = bool(enabled)
        self._console = Console(stderr=True) if _HAVE_RICH else None
        self._progress = None
        self._task = None

    # -- lifecycle ----------------------------------------------------------
    def __enter__(self) -> "NightReporter":
        if not self.enabled:
            return self
        self._progress = Progress(
            TextColumn("[bold]{task.description}", style=PALETTE["slate"]),
            BarColumn(
                bar_width=None,
                complete_style=PALETTE["slate"],
                finished_style=PALETTE["moss"],
                pulse_style=PALETTE["mauve"],
            ),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%", style=PALETTE["stone"]),
            MofNCompleteColumn(),
            TextColumn("frames", style=PALETTE["taupe"]),
            TimeElapsedColumn(),
            TextColumn("elapsed", style=PALETTE["taupe"]),
            TimeRemainingColumn(),
            TextColumn("ETA", style=PALETTE["taupe"]),
            console=self._console,
            transient=False,
        )
        self._progress.start()
        self._task = self._progress.add_task(self.title, total=max(1, self.total_frames))
        # Route logging through the same console so log lines don't tear the bar.
        root = logging.getLogger()
        self._saved_handlers = list(root.handlers)
        for h in self._saved_handlers:
            root.removeHandler(h)
        rich_handler = RichHandler(
            console=self._console, show_path=False, rich_tracebacks=True,
            markup=False, log_time_format="%H:%M:%S",
        )
        rich_handler.setLevel(logging.INFO)
        root.addHandler(rich_handler)
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        if not self.enabled:
            return
        if self._progress is not None:
            self._progress.stop()
        root = logging.getLogger()
        for h in list(root.handlers):
            root.removeHandler(h)
        for h in self._saved_handlers:
            root.addHandler(h)

    # -- output -------------------------------------------------------------
    def banner(self, lines: List[tuple]) -> None:
        """Render the plan overview. ``lines`` is a list of (label, value)."""
        if not self.enabled:
            for label, value in lines:
                logger.info("%s: %s", label, value)
            return
        table = Table.grid(padding=(0, 2))
        table.add_column(justify="right", style=PALETTE["stone"])
        table.add_column(style=PALETTE["slate"])
        for label, value in lines:
            table.add_row(str(label), str(value))
        self._console.print(
            Panel(table, title=Text(self.title, style=f"bold {PALETTE['mauve']}"),
                  border_style=PALETTE["plum"])
        )

    def start_block(self, label: str, *, subtitle: str = "") -> None:
        self._block = label
        if not self.enabled:
            logger.info("Block: %s %s", label, subtitle)
            return
        text = Text(f"▸ {label}", style=f"bold {PALETTE['mauve']}")
        if subtitle:
            text.append(f"  {subtitle}", style=PALETTE["stone"])
        self._console.print(text)
        if self._task is not None:
            self._progress.update(self._task, description=label)

    def frame_captured(self, frame_type: str, *, filt: Optional[str] = None,
                       hwp: Optional[float] = None, exp: float = 0.0,
                       idx: int = 0, count: int = 0) -> None:
        self._done += 1
        if not self.enabled:
            return
        detail = f"{frame_type}"
        if filt:
            detail += f"  {filt}"
        if hwp is not None:
            detail += f"  HWP {hwp:.0f}°"
        detail += f"  {exp:.1f}s  [{idx}/{count}]"
        desc = f"{self._block or self.title}  {detail}"
        if self._task is not None:
            self._progress.update(self._task, completed=self._done, description=desc)

    def qa(self, result) -> None:
        level = getattr(result, "level", "PASS" if getattr(result, "passed", True) else "FAIL")
        name = getattr(result, "name", "qa")
        msg = "; ".join(getattr(result, "messages", []) or [])
        if not self.enabled:
            return  # dispatch_qa_gate already logs
        style = _LEVEL_STYLE.get(level, PALETTE["stone"])
        line = Text(f"  QA {name} ", style=PALETTE["stone"])
        line.append(f"[{level}]", style=f"bold {style}")
        if msg:
            line.append(f" {msg}", style=PALETTE["taupe"])
        self._console.print(line)

    def note(self, text: str, *, style: str = "stone") -> None:
        if not self.enabled:
            logger.info("%s", text)
            return
        self._console.print(Text(text, style=PALETTE.get(style, PALETTE["stone"])))


@contextmanager
def null_reporter():
    """A disabled reporter (no display) for callers that opt out."""
    r = NightReporter(0, enabled=False)
    yield r
