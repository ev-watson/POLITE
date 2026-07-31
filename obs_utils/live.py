"""Live night-operation surface for the control notebooks.

The scripted runner (``scripts/execute_night.py``) captures a plan and exits.
This module is the other half: it lets a notebook kernel *watch* the frames the
runner is writing, one figure and one statistics line per frame, and run the QA
gates on the session directory without retyping the analysis.

Design rules that make the notebook templates safe to run cell-by-cell:

* **Read-only.** Nothing here commands hardware.  The connect helpers are thin
  re-exports of :mod:`obs_utils.interactive`; every other entry point only
  *reads* FITS off disk.  A stale cell re-run costs a redrawn figure, never a
  frame.
* **No module-scope hardware or plotting import.**  ``matplotlib`` and
  :mod:`obs_utils.interactive` load on first use, so ``import obs_utils.live``
  works on a headless reduction box that has no camera drivers.
* **Same statistics as the gates.**  Frame stats use the sigma-clipping
  parameters of :mod:`obs_utils.qa_lib` (sigma=5, 5 iterations), so a number
  printed live is the number the gate will judge -- a live "RON" that disagreed
  with the gate's would be worse than no number at all.
* **Same plot conventions as the reductions.**  ``origin='upper'``, steelblue
  histograms with a dashed red median: figures made at the telescope look like
  the ones in ``notebooks/reductions/``.

Typical notebook use::

    from obs_utils import live
    live.session_table("FITSDATA/20260717")     # what landed so far
    live.frame_report(live.latest_frame(d))     # image + histogram + stats
    live.watch(d, timeout_s=1800)               # block; report each new frame
    live.qa_print(live.sequence_audit(d))       # end-of-night completeness

Anything beyond a single frame goes through :func:`select`, which narrows on
headers alone and hands paths to the rest::

    flats = live.select(d, imagetyp="FLAT", filter_name="Photometric V")
    live.group_table(flats, by="exptime")       # n, level, spread, sigma
    live.trend(flats, x="exptime", y="median")  # linearity at a glance
    live.histogram(flats)                       # overlaid, shared bins
    live.contact_sheet(flats)                   # thumbnails
"""

from __future__ import annotations

import time
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, Iterator, List, Optional, Sequence, Union

import numpy as np
from astropy.io import fits
from astropy.stats import sigma_clipped_stats

# Match obs_utils.qa_lib exactly: a live number that disagreed with the gate's
# would be actively misleading at 3 a.m.
_CLIP_SIGMA = 5.0
_CLIP_ITERS = 5

# QHY268M reads out through a 16-bit ADC; the digital ceiling is the saturation
# reference for a live look.  True full-well in electrons is a calibration
# product (see caltools), not something to guess from a single frame.
FULL_SCALE_ADU = 65535

PathLike = Union[str, Path]


# --------------------------------------------------------------------------- #
# Device connect helpers (thin re-exports)
# --------------------------------------------------------------------------- #
def _interactive():
    """Import the hardware layer only when a connect helper is actually used."""
    from . import interactive

    return interactive


def connect_camera(**kwargs):
    """Connect the QHY268M only. See :func:`obs_utils.interactive.connect_camera`."""
    return _interactive().connect_camera(**kwargs)


def connect_filter_wheel(**kwargs):
    """Connect the ZWO EFW only."""
    return _interactive().connect_filter_wheel(**kwargs)


def connect_hwp(**kwargs):
    """Connect the Alpaca HWP rotator only (needs ``rotator_index`` set).

    The **half-wave plate**, not the field rotator: see
    :func:`obs_utils.interactive.connect_hwp`.
    """
    return _interactive().connect_hwp(**kwargs)


def connect_hwp_serial(**kwargs):
    """Connect the Pyxis over its native serial port (lab bench path)."""
    return _interactive().connect_hwp_serial(**kwargs)


def connect_field_rotator(**kwargs):
    """Connect and enable the PWI4 field rotator (instrument de-rotator)."""
    return _interactive().connect_field_rotator(**kwargs)


def connect_focuser(**kwargs):
    """Connect and enable the PWI4 focuser."""
    return _interactive().connect_focuser(**kwargs)


def connect_all(**kwargs):
    """Connect every fitted device, fault-isolated. Moves nothing."""
    return _interactive().connect_all(**kwargs)


def session():
    """The live :class:`~obs_utils.interactive.ObservatorySession` singleton."""
    return _interactive().session()


def shutdown() -> None:
    """Release camera / wheel / HWP. Does not touch the mount."""
    _interactive().shutdown()


# --------------------------------------------------------------------------- #
# Frame discovery
# --------------------------------------------------------------------------- #
def find_frames(directory: PathLike, pattern: str = "*.fits") -> List[Path]:
    """Every FITS under ``directory``, recursively, sorted by name.

    Per-invocation subdirectories (``execute_night`` writes one per run) mean a
    session directory is a tree, not a flat folder.
    """
    root = Path(directory)
    if root.is_file():
        return [root]
    return sorted(p for p in root.rglob(pattern) if p.is_file())


def latest_frame(directory: PathLike, pattern: str = "*.fits") -> Optional[Path]:
    """Most recently modified FITS under ``directory``, or ``None`` if empty."""
    frames = find_frames(directory, pattern)
    if not frames:
        return None
    return max(frames, key=lambda p: p.stat().st_mtime)


def _header(path: PathLike) -> fits.Header:
    return fits.getheader(str(path), ignore_missing_end=True)


def _detector_temp(hdr: fits.Header) -> Optional[float]:
    """Per-frame detector temperature, tolerating the pre-2026-07 spelling.

    Mirrors :func:`caltools.io` so live numbers and reduced numbers come from
    the same card on the same file.
    """
    for key in ("DET-TEMP", "CCD-TEMP"):
        if key in hdr:
            return float(hdr[key])
    return None


def _date_obs(hdr) -> Optional[datetime]:
    """DATE-OBS as a datetime; None when the card is absent or unparseable.

    Used only to order frames and to place them on a time axis.  It is never a
    calibration input, so a bad card costs a plot, not a number -- which is the
    right failure here, because this camera *has* written wrong timestamps
    before (the GPS-header misparse fixed in ``qhy_alpaca``).
    """
    raw = hdr.get("DATE-OBS")
    if not raw:
        return None
    try:
        return datetime.fromisoformat(str(raw).strip())
    except ValueError:
        return None


def _data(path: PathLike) -> np.ndarray:
    # BZERO=32768 in QHY headers: memmap must stay off or the unsigned offset is
    # applied lazily and every downstream statistic is wrong.
    with fits.open(str(path), memmap=False) as hdul:
        return np.asarray(hdul[0].data, dtype=float)


# --------------------------------------------------------------------------- #
# Per-frame statistics
# --------------------------------------------------------------------------- #
@dataclass
class FrameStats:
    """Sigma-clipped statistics plus the header fields a night is steered by."""

    path: Path
    imagetyp: str = "?"
    filter_name: Optional[str] = None
    exptime: Optional[float] = None
    hwp_angle_deg: Optional[float] = None
    pol_seq: Optional[str] = None
    object_name: Optional[str] = None
    det_temp_c: Optional[float] = None
    date_obs: Optional[datetime] = None
    mean: float = float("nan")
    median: float = float("nan")
    std: float = float("nan")
    minimum: float = float("nan")
    maximum: float = float("nan")
    saturated_px: int = 0
    saturated_frac: float = 0.0
    shape: tuple = ()

    def line(self) -> str:
        """One terminal-width summary line."""
        bits = [self.path.name, f"{self.imagetyp:<5s}"]
        if self.exptime is not None:
            bits.append(f"{self.exptime:7.3f}s")
        if self.filter_name:
            bits.append(f"{self.filter_name:<14s}")
        if self.hwp_angle_deg is not None:
            bits.append(f"HWP {self.hwp_angle_deg:6.2f}d")
        bits.append(f"med {self.median:9.2f}")
        bits.append(f"sig {self.std:7.2f} ADU")
        if self.det_temp_c is not None:
            bits.append(f"{self.det_temp_c:+.1f}C")
        if self.saturated_px:
            bits.append(f"SAT {self.saturated_px}px")
        return "  ".join(bits)

    def __str__(self) -> str:  # pragma: no cover - display sugar
        return self.line()


def frame_stats(path: PathLike, *, saturation_adu: int = FULL_SCALE_ADU) -> FrameStats:
    """Sigma-clipped statistics for one frame.

    Robust by construction: a cosmic ray or a cluster of hot pixels must not
    move the reported noise, or the operator learns to ignore the number.
    """
    path = Path(path)
    hdr = _header(path)
    data = _data(path)
    mean, median, std = sigma_clipped_stats(
        data, sigma=_CLIP_SIGMA, maxiters=_CLIP_ITERS
    )
    n_sat = int(np.count_nonzero(data >= saturation_adu))
    return FrameStats(
        path=path,
        imagetyp=str(hdr.get("IMAGETYP", "?")).upper(),
        filter_name=(str(hdr["FILTER"]).strip() if "FILTER" in hdr else None),
        exptime=(float(hdr["EXPTIME"]) if "EXPTIME" in hdr else None),
        hwp_angle_deg=(float(hdr["HWPANG"]) if "HWPANG" in hdr else None),
        pol_seq=(str(hdr["POLSEQ"]) if "POLSEQ" in hdr else None),
        object_name=(str(hdr["OBJECT"]).strip() if "OBJECT" in hdr else None),
        det_temp_c=_detector_temp(hdr),
        date_obs=_date_obs(hdr),
        mean=float(mean),
        median=float(median),
        std=float(std),
        minimum=float(np.min(data)),
        maximum=float(np.max(data)),
        saturated_px=n_sat,
        saturated_frac=n_sat / data.size,
        shape=tuple(data.shape),
    )


def session_table(directory: PathLike, *, pattern: str = "*.fits") -> Dict[tuple, int]:
    """Tally frames by (IMAGETYP, EXPTIME, FILTER) and print the roll-up.

    The fast "did the plan actually do what I asked?" check -- header-only, so
    it stays instant on a directory with hundreds of full-frame images.
    """
    frames = find_frames(directory, pattern)
    counts: Dict[tuple, int] = defaultdict(int)
    for path in frames:
        hdr = _header(path)
        key = (
            str(hdr.get("IMAGETYP", "?")).upper(),
            float(hdr.get("EXPTIME", float("nan"))),
            str(hdr.get("FILTER", "-")).strip(),
        )
        counts[key] += 1

    print(f"{directory}: {len(frames)} frame(s)")
    for (itype, exp, filt) in sorted(counts, key=lambda k: (k[0], k[1], k[2])):
        print(f"  {itype:<6s} {exp:8.3f}s  {filt:<16s} x{counts[(itype, exp, filt)]}")
    return dict(counts)


# --------------------------------------------------------------------------- #
# Live viewing
# --------------------------------------------------------------------------- #
def _plt():
    import matplotlib.pyplot as plt

    return plt


def show_frame(
    path: PathLike,
    *,
    percentile_clip: tuple = (1, 99),
    bins: int = 200,
    figsize: tuple = (16, 6),
    stats: Optional[FrameStats] = None,
    cmap: str = "viridis",
):
    """Image + histogram for one frame, in the project reduction style.

    ``origin='upper'`` (astronomical convention), steelblue histogram, dashed
    red median -- deliberately identical to ``notebooks/reductions/`` so a frame
    inspected at the telescope and the same frame inspected next morning are
    read the same way.
    """
    from caltools.plotting import image_with_colorbar

    plt = _plt()
    path = Path(path)
    data = _data(path)
    st = stats if stats is not None else frame_stats(path)

    fig, axes = plt.subplots(1, 2, figsize=figsize)
    image_with_colorbar(
        axes[0], data, label="ADU", percentile_clip=percentile_clip, cmap=cmap
    )
    axes[0].set_title(f"{path.name}\n{_frame_subtitle(st)}")

    axes[1].hist(data.ravel(), bins=bins, color="steelblue", alpha=0.7)
    axes[1].axvline(st.median, ls="--", color="red", lw=1.5, label=f"median = {st.median:.1f}")
    axes[1].set_yscale("log")
    axes[1].set_xlabel("ADU")
    axes[1].set_ylabel("# of pixels")
    axes[1].set_title(
        f"clipped median {st.median:.2f}, sigma {st.std:.2f} ADU  "
        f"(max {st.maximum:.0f})"
    )
    axes[1].legend()
    fig.tight_layout()
    return fig, axes


def _frame_subtitle(st: FrameStats) -> str:
    bits = [st.imagetyp]
    if st.exptime is not None:
        bits.append(f"{st.exptime:g}s")
    if st.filter_name:
        bits.append(st.filter_name)
    if st.hwp_angle_deg is not None:
        bits.append(f"HWP {st.hwp_angle_deg:g}deg")
    if st.det_temp_c is not None:
        bits.append(f"{st.det_temp_c:+.1f}C")
    return "  ".join(bits)


def frame_report(
    path: PathLike, *, show: bool = True, **show_kwargs
) -> FrameStats:
    """Print the statistics line and, unless ``show=False``, draw the figure."""
    st = frame_stats(path)
    print(st.line())
    if show:
        plt = _plt()
        show_frame(path, stats=st, **show_kwargs)
        plt.show()
    return st


# --------------------------------------------------------------------------- #
# Watching a running session
# --------------------------------------------------------------------------- #
def iter_new_frames(
    directory: PathLike,
    *,
    poll_s: float = 5.0,
    timeout_s: float = 3600.0,
    settle_s: float = 1.0,
    pattern: str = "*.fits",
    include_existing: bool = False,
) -> Iterator[Path]:
    """Yield each FITS as it lands in ``directory``, until ``timeout_s``.

    ``settle_s`` guards against reading a file the writer still has open: a
    frame is only yielded once its mtime has stopped moving.  Stops early on
    ``KeyboardInterrupt`` (interrupting the kernel is the intended way to end a
    watch), so the caller keeps whatever it has already collected.
    """
    directory = Path(directory)
    seen = set() if include_existing else set(find_frames(directory, pattern))
    deadline = time.monotonic() + float(timeout_s)

    try:
        while time.monotonic() < deadline:
            for path in find_frames(directory, pattern):
                if path in seen:
                    continue
                if time.time() - path.stat().st_mtime < settle_s:
                    continue  # still being written; pick it up next poll
                seen.add(path)
                yield path
            time.sleep(poll_s)
    except KeyboardInterrupt:  # pragma: no cover - interactive stop
        print("watch interrupted")
        return


def watch(
    directory: PathLike,
    *,
    poll_s: float = 5.0,
    timeout_s: float = 3600.0,
    settle_s: float = 1.0,
    show: bool = True,
    every: int = 1,
    on_frame: Optional[Callable[[FrameStats], Any]] = None,
    include_existing: bool = False,
    pattern: str = "*.fits",
) -> List[FrameStats]:
    """Block, reporting each new frame as it lands; return every frame's stats.

    ``show=True`` draws the image+histogram figure for one frame in ``every``
    (the statistics line still prints for all of them) -- a 16-angle sequence
    does not need 16 figures to tell you the run is healthy.  ``on_frame`` is
    called with each :class:`FrameStats` for custom cells (beam separation,
    guiding trends, alerting).

    Interrupt the kernel to stop early; the frames collected so far are
    returned.
    """
    collected: List[FrameStats] = []
    for path in iter_new_frames(
        directory,
        poll_s=poll_s,
        timeout_s=timeout_s,
        settle_s=settle_s,
        pattern=pattern,
        include_existing=include_existing,
    ):
        try:
            st = frame_stats(path)
        except Exception as exc:  # pragma: no cover - partial/corrupt file
            print(f"{path.name}: unreadable ({exc!r})")
            continue
        collected.append(st)
        print(st.line())
        if show and (len(collected) - 1) % max(1, int(every)) == 0:
            plt = _plt()
            show_frame(path, stats=st)
            plt.show()
        if on_frame is not None:
            on_frame(st)
    return collected


# --------------------------------------------------------------------------- #
# Selecting a group of frames
# --------------------------------------------------------------------------- #
# The filter keywords, the ``by=`` keys and the ``x=``/``y=`` axes below are all
# spelled with the *FrameStats field names*.  One vocabulary, learned once: if a
# quantity shows up in a statistics line it can also be selected on, grouped by,
# and plotted against.
_CARD_FOR = {
    "imagetyp": "IMAGETYP",
    "filter_name": "FILTER",
    "exptime": "EXPTIME",
    "hwp_angle_deg": "HWPANG",
    "object_name": "OBJECT",
    "pol_seq": "POLSEQ",
}


def _card_value(hdr, field: str):
    """One FrameStats field read straight from a header (no pixel access)."""
    card = _CARD_FOR[field]
    if card not in hdr:
        return None
    raw = hdr[card]
    if field in ("exptime", "hwp_angle_deg"):
        return float(raw)
    value = str(raw).strip()
    return value.upper() if field == "imagetyp" else value


def _match(actual, wanted) -> bool:
    """Compare one header value against one filter argument.

    Deliberately forgiving in the ways a 3 a.m. typo is forgiving: strings match
    case- and whitespace-insensitively, numbers within a float epsilon, a
    sequence means "any of these", and a callable is used as a predicate so
    ``exptime=lambda t: t > 10`` works without a second query language.
    """
    if callable(wanted):
        return bool(wanted(actual))
    if isinstance(wanted, (list, tuple, set, frozenset)):
        return any(_match(actual, one) for one in wanted)
    if actual is None or wanted is None:
        return actual is None and wanted is None
    if isinstance(wanted, str):
        return str(actual).strip().casefold() == wanted.strip().casefold()
    return abs(float(actual) - float(wanted)) <= 1e-6


def _as_datetime(value) -> Optional[datetime]:
    if value is None or isinstance(value, datetime):
        return value
    return datetime.fromisoformat(str(value).strip())


def _frame_paths(source, pattern: str = "*.fits") -> List[Path]:
    """Normalize a directory, one frame, or a sequence of either into paths.

    :class:`FrameStats` are accepted too, so the output of one helper feeds the
    next without the caller unpacking ``.path`` by hand.
    """
    if isinstance(source, FrameStats):
        return [source.path]
    if isinstance(source, (str, Path)):
        candidate = Path(source)
        return find_frames(candidate, pattern) if candidate.is_dir() else [candidate]
    out: List[Path] = []
    for item in source:
        out.extend(_frame_paths(item, pattern))
    return out


def _as_stats(source, *, pattern: str = "*.fits") -> List[FrameStats]:
    """Accept frames as paths *or* as already-measured :class:`FrameStats`.

    Statistics cost a full pixel read, so a caller that already has them must
    never be made to pay twice -- which is the whole point of ``select`` ->
    ``stats_table`` -> everything else.
    """
    if isinstance(source, FrameStats):
        return [source]
    if not isinstance(source, (str, Path)):
        items = list(source)
        if items and all(isinstance(item, FrameStats) for item in items):
            return items
        return [frame_stats(p) for p in _frame_paths(items, pattern)]
    return [frame_stats(p) for p in _frame_paths(source, pattern)]


def select(
    source,
    *,
    pattern: str = "*.fits",
    after=None,
    before=None,
    **filters,
) -> List[Path]:
    """Frames matching every filter, in discovery order. Header-only, so fast.

    Filter keywords are :class:`FrameStats` field names -- ``imagetyp``,
    ``filter_name``, ``exptime``, ``hwp_angle_deg``, ``object_name``,
    ``pol_seq`` -- each matched by :func:`_match`.  ``after`` / ``before`` window
    on DATE-OBS and accept a datetime or an ISO string.

    This is the primitive the rest of the group API consumes: it never opens
    pixel data, so narrowing a night of full-frame images down to the eight that
    matter costs milliseconds, and only those eight are then read::

        v = live.select(SESSION_DIR, imagetyp="FLAT", filter_name="Photometric V")
        live.group_table(live.stats_table(v, show=False), by="exptime")
    """
    unknown = sorted(set(filters) - set(_CARD_FOR))
    if unknown:
        raise ValueError(
            f"unknown filter(s) {unknown}; use {sorted(_CARD_FOR)}, after=, before="
        )
    after, before = _as_datetime(after), _as_datetime(before)
    out: List[Path] = []
    for path in _frame_paths(source, pattern):
        hdr = _header(path)
        if not all(_match(_card_value(hdr, k), v) for k, v in filters.items()):
            continue
        if after is not None or before is not None:
            when = _date_obs(hdr)
            if when is None:
                continue
            if (after is not None and when < after) or (
                before is not None and when > before
            ):
                continue
        out.append(path)
    return out


# --------------------------------------------------------------------------- #
# Group statistics
# --------------------------------------------------------------------------- #
def stats_table(
    source,
    *,
    pattern: str = "*.fits",
    imagetyp: Optional[str] = None,
    show: bool = True,
) -> List[FrameStats]:
    """:func:`frame_stats` for every frame in ``source``, printed one per line.

    ``source`` is a directory, a frame, or anything :func:`select` returned.
    Pass ``show=False`` when feeding :func:`group_table` -- two hundred lines
    scrolling past is not a summary.
    """
    out: List[FrameStats] = []
    for path in _frame_paths(source, pattern):
        st = frame_stats(path)
        if imagetyp and st.imagetyp != imagetyp.upper():
            continue
        out.append(st)
        if show:
            print(st.line())
    return out


@dataclass
class GroupStats:
    """One row of :func:`group_table` -- a set of like frames, summarized."""

    key: tuple
    n: int
    median: float
    spread: float
    sigma: float
    minimum: float
    maximum: float
    saturated_px: int

    def line(self) -> str:
        label = " ".join("-" if k is None else f"{k}" for k in self.key)
        return (
            f"{label:<34s} n={self.n:<4d} med {self.median:10.2f}  "
            f"spread {self.spread:8.2f}  sig {self.sigma:8.2f}  "
            f"[{self.minimum:.0f}, {self.maximum:.0f}]"
            + (f"  SAT {self.saturated_px}px" if self.saturated_px else "")
        )

    def __str__(self) -> str:  # pragma: no cover - display sugar
        return self.line()


def group_table(
    source,
    *,
    by: Union[str, Sequence[str]] = ("imagetyp", "exptime", "filter_name"),
    show: bool = True,
) -> List[GroupStats]:
    """Aggregate frames into groups and summarize each one.

    ``spread`` is max-minus-min of the per-frame clipped medians and is the
    number to read first: for a bias or dark set it should be small, and a large
    one says the set is not internally consistent no matter how good the mean
    looks.  ``sigma`` is the median of the per-frame clipped sigmas, not the
    sigma of the group, so it stays comparable to a single frame's line.
    """
    keys = (by,) if isinstance(by, str) else tuple(by)
    stats = _as_stats(source)
    groups: Dict[tuple, List[FrameStats]] = defaultdict(list)
    for st in stats:
        groups[tuple(getattr(st, k) for k in keys)].append(st)

    rows: List[GroupStats] = []
    for key in sorted(groups, key=lambda k: tuple(str(part) for part in k)):
        members = groups[key]
        med = np.array([s.median for s in members], dtype=float)
        rows.append(
            GroupStats(
                key=key,
                n=len(members),
                median=float(np.median(med)),
                spread=float(med.max() - med.min()),
                sigma=float(np.median([s.std for s in members])),
                minimum=float(min(s.minimum for s in members)),
                maximum=float(max(s.maximum for s in members)),
                saturated_px=int(sum(s.saturated_px for s in members)),
            )
        )

    if show:
        print(f"{len(stats)} frame(s) in {len(rows)} group(s) by {'/'.join(keys)}")
        for row in rows:
            print("  " + row.line())
    return rows


# --------------------------------------------------------------------------- #
# Group plots
# --------------------------------------------------------------------------- #
_AXIS_LABELS = {
    "index": "Frame",
    "time": "UTC",
    "exptime": "Exposure (s)",
    "hwp_angle_deg": "HWP angle (deg)",
    "mean": "Clipped mean (ADU)",
    "median": "Clipped median (ADU)",
    "std": "Clipped sigma (ADU)",
    "minimum": "Minimum (ADU)",
    "maximum": "Maximum (ADU)",
    "det_temp_c": "Detector temperature (C)",
    "saturated_px": "Saturated pixels",
    "saturated_frac": "Saturated fraction",
}


def _axis_values(stats: Sequence[FrameStats], key: str) -> list:
    if key == "index":
        return list(range(len(stats)))
    if key == "time":
        return [s.date_obs for s in stats]
    if key not in _AXIS_LABELS:
        raise ValueError(f"unknown axis {key!r}; use one of {sorted(_AXIS_LABELS)}")
    return [getattr(s, key) for s in stats]


def trend(
    source,
    *,
    x: str = "index",
    y: str = "median",
    by: Optional[str] = None,
    ax=None,
    figsize: tuple = (10, 5),
    show_median: bool = True,
):
    """One generic curve through a group: any ``y`` against any ``x``.

    Both axes are :class:`FrameStats` fields, plus the two pseudo-fields
    ``index`` (capture order) and ``time`` (DATE-OBS).  So level-versus-exposure,
    level-versus-HWP-angle, noise-versus-level and temperature-versus-time are
    all the same call with different arguments rather than four functions.

    ``by=`` splits into one series per value -- ``by="filter_name"`` over a flat
    ladder is the per-filter view.  Frames missing either value are dropped and
    the count is printed, because a silently short curve reads as data.
    """
    plt = _plt()
    stats = _as_stats(source)
    fig = None
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.get_figure()

    if by is None:
        series: Dict[Any, List[FrameStats]] = {None: list(stats)}
    else:
        series = defaultdict(list)
        for st in stats:
            series[getattr(st, by)].append(st)

    dropped = 0
    plotted = 0
    for name in sorted(series, key=str):
        members = series[name]
        xs = _axis_values(members, x)
        ys = _axis_values(members, y)
        pairs = [(a, b) for a, b in zip(xs, ys) if a is not None and b is not None]
        dropped += len(members) - len(pairs)
        if not pairs:
            continue
        pairs.sort(key=lambda p: p[0])
        ax.plot(
            [p[0] for p in pairs],
            [p[1] for p in pairs],
            "o-",
            ms=3,
            label=None if name is None else f"{name} (n={len(pairs)})",
        )
        plotted += len(pairs)

    if show_median and plotted:
        values = [v for v in _axis_values(stats, y) if v is not None]
        ax.axhline(float(np.median(values)), ls="--", color="gray", lw=1)
    ax.set_xlabel(_AXIS_LABELS.get(x, x))
    ax.set_ylabel(_AXIS_LABELS.get(y, y))
    ax.set_title(f"{_AXIS_LABELS.get(y, y)} vs {_AXIS_LABELS.get(x, x)} — {plotted} frame(s)")
    if by is not None:
        ax.legend(fontsize="small")
    if x == "time":
        fig.autofmt_xdate()
    if dropped:
        print(f"{dropped} frame(s) dropped: no {x!r} or no {y!r}")
    if fig is not None:
        fig.tight_layout()
    return fig, ax


def level_trend(source, *, figsize: tuple = (16, 5)):
    """Per-frame clipped level and noise versus frame index.

    Drift here is the earliest visible sign of a cooler that has not settled, a
    light leak, or sky brightening -- all of which are cheaper to catch during
    the run than in reduction.  Two :func:`trend` panels.
    """
    plt = _plt()
    stats = _as_stats(source)
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    trend(stats, x="index", y="median", ax=axes[0])
    trend(stats, x="index", y="std", ax=axes[1])
    fig.tight_layout()
    return fig, axes


def temperature_trend(source, *, figsize: tuple = (8, 5)):
    """Detector temperature versus frame index.

    Reduction uses the per-frame detector temperature, not the setpoint, so this
    is the plot that says whether the darks in hand match the lights in hand.
    """
    stats = _as_stats(source)
    if not any(s.det_temp_c is not None for s in stats):
        print("No detector temperature card in these frames.")
        return None
    return trend(stats, x="index", y="det_temp_c", figsize=figsize)


def histogram(
    source,
    *,
    bins: int = 200,
    adu_range: Optional[tuple] = None,
    max_frames: int = 12,
    figsize: tuple = (10, 6),
    log: bool = True,
):
    """Overlaid pixel histograms for a group, on one shared set of bins.

    Shared bins are the point: frames binned separately cannot be compared by
    eye.  The default range is taken from the group's own clipped statistics
    (median +/- 6 sigma, clamped to the observed extremes), so a few hot pixels
    do not stretch the axis until the distribution is a spike.  Works for one
    frame or for a set.
    """
    plt = _plt()
    stats = _as_stats(source)
    if not stats:
        print("No frames to histogram.")
        return None
    if len(stats) > max_frames:
        print(f"{len(stats)} frames; showing the first {max_frames}")
        stats = stats[:max_frames]

    if adu_range is None:
        lo = max(min(s.minimum for s in stats), min(s.median - 6 * s.std for s in stats))
        hi = min(max(s.maximum for s in stats), max(s.median + 6 * s.std for s in stats))
        adu_range = (float(lo), float(hi)) if hi > lo else None

    fig, ax = plt.subplots(figsize=figsize)
    edges = None
    for st in stats:
        counts, edges = np.histogram(_data(st.path).ravel(), bins=bins, range=adu_range)
        centres = 0.5 * (edges[:-1] + edges[1:])
        ax.step(centres, counts, where="mid", lw=1, label=st.path.name)
    if log:
        ax.set_yscale("log")
    ax.set_xlabel("ADU")
    ax.set_ylabel("# of pixels")
    ax.set_title(f"{len(stats)} frame(s), {bins} bins")
    ax.legend(fontsize="small", ncol=2)
    fig.tight_layout()
    return fig, ax


def focus_curve(sweep, *, metric: str = "hfd_px", figsize: tuple = (9, 5)):
    """Focus metric versus focuser position, from :meth:`focus_sweep`.

    ``sweep`` is the ``[(position, StarProfile), ...]`` that
    :meth:`obs_utils.interactive.ObservatorySession.focus_sweep` returns, or the
    same shape with a bare number in place of the profile.

    **This function does not choose a focus position.**  It draws the curve and
    prints the table; you read the minimum and set it.  A fitted minimum would
    be a number with no error bar sitting on top of a curve whose scatter is the
    only honest statement about how well the focus is known.
    """
    plt = _plt()
    points = [
        (float(pos), float(getattr(value, metric, value)))
        for pos, value in sweep
    ]
    points.sort(key=lambda p: p[0])
    for pos, val in points:
        print(f"  {pos:10.1f}   {metric} {val:8.3f}")

    fig, ax = plt.subplots(figsize=figsize)
    ax.plot([p[0] for p in points], [p[1] for p in points], "o-", color="steelblue")
    ax.set_xlabel("Focuser position")
    ax.set_ylabel(metric)
    ax.set_title(f"Focus sweep — {len(points)} point(s); read the minimum yourself")
    fig.tight_layout()
    return fig, ax


# --------------------------------------------------------------------------- #
# Group images
# --------------------------------------------------------------------------- #
def _show_array(data: np.ndarray, title: str, *, percentile_clip=(1, 99),
                figsize: tuple = (9, 7), cmap: str = "viridis"):
    from caltools.plotting import image_with_colorbar

    plt = _plt()
    fig, ax = plt.subplots(figsize=figsize)
    image_with_colorbar(ax, data, label="ADU", percentile_clip=percentile_clip, cmap=cmap)
    ax.set_title(title)
    fig.tight_layout()
    return fig, ax


def _thumbnail(data: np.ndarray, max_px: int) -> np.ndarray:
    """Decimate by striding, for display only.

    Striding aliases: a strided thumbnail of a bias frame is not a picture of
    its read noise.  It is here to answer "is the field in the frame, are the
    stars round" at a glance, and nothing else.
    """
    step = max(1, int(np.ceil(max(data.shape) / float(max_px))))
    return data[::step, ::step]


def contact_sheet(
    source,
    *,
    ncols: int = 4,
    max_frames: int = 16,
    thumb_px: int = 256,
    percentile_clip: tuple = (1, 99),
    cmap: str = "viridis",
    figsize: Optional[tuple] = None,
):
    """Thumbnail grid over a group -- the "did anything odd happen" glance.

    Thumbnails are strided (see :func:`_thumbnail`), so read this sheet for
    framing, trailing and cloud, never for a pixel-level judgement.
    """
    plt = _plt()
    paths = _frame_paths(source)
    if not paths:
        print("No frames to show.")
        return None
    if len(paths) > max_frames:
        print(f"{len(paths)} frames; showing the first {max_frames}")
        paths = paths[:max_frames]

    nrows = int(np.ceil(len(paths) / ncols))
    fig, axes = plt.subplots(
        nrows, ncols, figsize=figsize or (3.2 * ncols, 3.0 * nrows), squeeze=False
    )
    for ax, path in zip(axes.ravel(), paths):
        thumb = _thumbnail(_data(path), thumb_px)
        lo, hi = np.percentile(thumb, percentile_clip)
        ax.imshow(thumb, origin="upper", cmap=cmap, vmin=lo, vmax=hi)
        ax.set_title(path.name, fontsize="small")
        ax.set_xticks([])
        ax.set_yticks([])
    for ax in axes.ravel()[len(paths):]:
        ax.axis("off")
    fig.tight_layout()
    return fig, axes


def stack_preview(
    source,
    *,
    method: str = "median",
    max_frames: int = 16,
    show: bool = True,
    **show_kwargs,
) -> np.ndarray:
    """Quick median (or mean) of a group, full resolution.

    **Not a calibration product.**  Masters come from
    :mod:`caltools.stacking` (``master_bias`` / ``master_dark`` / ``master_flat``),
    which handles scaling, rejection and provenance.  This is the look that tells
    you at the telescope whether the set is worth keeping.

    All frames are held in memory at once, hence ``max_frames``: sixteen
    full-frame QHY268M images is already ~1.7 GB as float32.
    """
    paths = _frame_paths(source)
    if not paths:
        raise ValueError("no frames to stack")
    if len(paths) > max_frames:
        print(f"{len(paths)} frames; stacking the first {max_frames} (max_frames=)")
        paths = paths[:max_frames]

    cube = np.stack([_data(p).astype(np.float32) for p in paths])
    stacked = np.median(cube, axis=0) if method == "median" else np.mean(cube, axis=0)
    if show:
        plt = _plt()
        _show_array(stacked, f"{method} of {len(paths)} frame(s)", **show_kwargs)
        plt.show()
    return stacked


def diff(a, b, *, show: bool = True, **show_kwargs) -> np.ndarray:
    """``a - b`` for two frames, printed and shown.

    The fastest way to see what changed between two exposures -- a light leak
    appearing, a filter that did not move, drift between two darks.  Its sigma
    is *not* the read noise: the difference of two frames carries sqrt(2) times
    the single-frame noise (``caltools.noise`` does that properly).
    """
    data = _data(_frame_paths(a)[0]) - _data(_frame_paths(b)[0])
    mean, median, std = sigma_clipped_stats(data, sigma=_CLIP_SIGMA, maxiters=_CLIP_ITERS)
    print(f"difference: median {median:.3f}  sigma {std:.3f} ADU  "
          f"[{data.min():.1f}, {data.max():.1f}]")
    if show:
        plt = _plt()
        _show_array(data, f"difference — median {median:.2f}, sigma {std:.2f} ADU", **show_kwargs)
        plt.show()
    return data


def hwp_coverage(directory: PathLike, *, pattern: str = "*.fits") -> Dict[tuple, list]:
    """Angles captured so far per (OBJECT, POLSEQ, FILTER), printed.

    The live counterpart to :func:`obs_utils.qa_lib.run_sequence_audit`: that
    one judges a finished session, this one shows a sequence filling in while
    there is still time to re-take a missing angle.
    """
    groups: Dict[tuple, set] = defaultdict(set)
    for path in find_frames(directory, pattern):
        hdr = _header(path)
        if str(hdr.get("IMAGETYP", "LIGHT")).upper() != "LIGHT":
            continue
        if "HWPANG" not in hdr:
            continue
        key = (hdr.get("OBJECT"), hdr.get("POLSEQ"), hdr.get("FILTER"))
        groups[key].add(float(hdr["HWPANG"]))

    out = {k: sorted(v) for k, v in groups.items()}
    for (obj, polseq, filt), angles in sorted(out.items(), key=lambda kv: str(kv[0])):
        pretty = ", ".join(f"{a:g}" for a in angles)
        print(f"{obj} / {polseq} / {filt}: {len(angles)} angle(s) [{pretty}]")
    if not out:
        print("No LIGHT frames with HWPANG yet.")
    return out


# --------------------------------------------------------------------------- #
# QA gate one-liners
# --------------------------------------------------------------------------- #
def qa_print(result) -> Any:
    """Print a :class:`~obs_utils.qa_lib.QAResult` as ``LEVEL name`` + messages."""
    print(f"[{result.level}] {result.name}")
    for msg in result.messages:
        print(f"  {msg}")
    for msg in getattr(result, "warnings", []):
        print(f"  WARN: {msg}")
    return result


def bias_qa(paths: Union[PathLike, Sequence[PathLike]], **kwargs):
    """Bias level / read-noise gate over a directory or explicit file list."""
    from .qa_lib import run_bias_qa

    return run_bias_qa(_as_paths(paths), **kwargs)


def sequence_audit(directory: PathLike, **kwargs):
    """HWP angle completeness per (OBJECT, POLSEQ, FILTER) for a finished run."""
    from .qa_lib import run_sequence_audit

    return run_sequence_audit(directory, **kwargs)


def first_light_qa(paths: Union[PathLike, Sequence[PathLike]], **kwargs):
    """Measured P/PA versus a polarized standard (needs poltools + pol_config)."""
    from .qa_lib import run_first_light_qa

    return run_first_light_qa(_as_paths(paths), **kwargs)


def flat_quality_gate(paths: Union[PathLike, Sequence[PathLike]], **kwargs):
    """lsq vs double_ratio agreement -- the flat-field sanity check."""
    from .qa_lib import run_flat_quality_gate

    return run_flat_quality_gate(_as_paths(paths), **kwargs)


def _as_paths(paths: Union[PathLike, Sequence[PathLike]]) -> List[str]:
    """Accept a directory, a glob-free path, or a sequence of either.

    The gates below want strings; :func:`_frame_paths` does the walking, so a
    :func:`select` result can be handed to a gate directly.
    """
    return [str(p) for p in _frame_paths(paths)]


__all__ = [
    "FULL_SCALE_ADU",
    "FrameStats",
    "GroupStats",
    "bias_qa",
    "connect_all",
    "connect_camera",
    "connect_field_rotator",
    "connect_filter_wheel",
    "connect_focuser",
    "connect_hwp",
    "connect_hwp_serial",
    "contact_sheet",
    "diff",
    "find_frames",
    "first_light_qa",
    "flat_quality_gate",
    "focus_curve",
    "frame_report",
    "frame_stats",
    "group_table",
    "histogram",
    "hwp_coverage",
    "iter_new_frames",
    "latest_frame",
    "level_trend",
    "qa_print",
    "select",
    "sequence_audit",
    "session",
    "session_table",
    "show_frame",
    "shutdown",
    "stack_preview",
    "stats_table",
    "temperature_trend",
    "trend",
    "watch",
]
