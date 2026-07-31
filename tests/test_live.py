"""The notebook live surface: frame discovery, robust stats, watch loop.

``obs_utils.live`` is what an operator stares at while a plan is running, so the
properties worth pinning are (a) it never misses or double-reports a frame, and
(b) its numbers are the same sigma-clipped numbers the QA gates will use.
"""

from __future__ import annotations

import subprocess
import sys
import threading
import time
from pathlib import Path

import numpy as np
import pytest
from astropy.io import fits

from obs_utils import live

ROOT = Path(__file__).resolve().parents[1]


def _write_frame(path: Path, *, level=800.0, noise=7.0, hot=0, cards=None, seed=0):
    """A small synthetic QHY-like frame with optional saturated hot pixels."""
    rng = np.random.default_rng(seed)
    data = rng.normal(level, noise, size=(64, 64))
    if hot:
        flat = data.ravel()
        flat[:hot] = live.FULL_SCALE_ADU
        data = flat.reshape(64, 64)
    hdr = fits.Header()
    hdr["IMAGETYP"] = "DARK"
    hdr["EXPTIME"] = 5.0
    hdr["FILTER"] = "Dark"
    for key, value in (cards or {}).items():
        hdr[key] = value
    path.parent.mkdir(parents=True, exist_ok=True)
    fits.writeto(path, data.astype(np.float32), hdr, overwrite=True)
    return path


# --------------------------------------------------------------------------- #
# Discovery
# --------------------------------------------------------------------------- #
def test_find_frames_walks_per_invocation_subdirectories(tmp_path):
    """execute_night writes one subdir per run; a flat glob would miss them."""
    _write_frame(tmp_path / "run_a" / "f001.fits")
    _write_frame(tmp_path / "run_b" / "f002.fits")
    _write_frame(tmp_path / "top.fits")
    found = live.find_frames(tmp_path)
    assert [p.name for p in found] == ["f001.fits", "f002.fits", "top.fits"]


def test_latest_frame_is_newest_not_last_alphabetically(tmp_path):
    old = _write_frame(tmp_path / "z_old.fits")
    new = _write_frame(tmp_path / "a_new.fits")
    import os

    os.utime(old, (time.time() - 600, time.time() - 600))
    assert live.latest_frame(tmp_path) == new


def test_latest_frame_of_empty_directory_is_none(tmp_path):
    assert live.latest_frame(tmp_path) is None


# --------------------------------------------------------------------------- #
# Frame statistics
# --------------------------------------------------------------------------- #
def test_frame_stats_are_robust_to_hot_pixels(tmp_path):
    """A live sigma that a few hot pixels can move is a number operators learn
    to ignore -- it must match the clipped statistic the gate uses."""
    path = _write_frame(tmp_path / "hot.fits", level=800.0, noise=7.0, hot=40)
    st = live.frame_stats(path)
    assert st.median == pytest.approx(800.0, abs=1.0)
    assert st.std == pytest.approx(7.0, abs=1.0)
    assert st.maximum == live.FULL_SCALE_ADU
    assert st.saturated_px == 40


def test_frame_stats_reads_pol_and_temperature_cards(tmp_path):
    path = _write_frame(
        tmp_path / "pol.fits",
        cards={
            "IMAGETYP": "LIGHT",
            "HWPANG": 22.5,
            "POLSEQ": "polV16",
            "OBJECT": "HD 154445",
            "DET-TEMP": -13.9,
        },
    )
    st = live.frame_stats(path)
    assert st.imagetyp == "LIGHT"
    assert st.hwp_angle_deg == 22.5
    assert st.pol_seq == "polV16"
    assert st.object_name == "HD 154445"
    assert st.det_temp_c == -13.9
    assert "HWP  22.50d" in st.line()


def test_legacy_ccd_temp_spelling_still_reports(tmp_path):
    """Pre-2026-07 data carries CCD-TEMP; caltools.io accepts both, so must this."""
    path = _write_frame(tmp_path / "legacy.fits", cards={"CCD-TEMP": -12.2})
    assert live.frame_stats(path).det_temp_c == -12.2


def test_det_temp_wins_over_legacy_spelling(tmp_path):
    path = _write_frame(
        tmp_path / "both.fits", cards={"CCD-TEMP": -12.2, "DET-TEMP": -14.0},
    )
    assert live.frame_stats(path).det_temp_c == -14.0


# --------------------------------------------------------------------------- #
# Watch loop
# --------------------------------------------------------------------------- #
def test_watch_reports_each_new_frame_exactly_once(tmp_path):
    """The real case: frames land in a subdirectory while the loop is polling."""
    _write_frame(tmp_path / "before.fits")
    run_dir = tmp_path / "run_2110"
    seen = []

    def feeder():
        for i in range(3):
            time.sleep(0.05)
            _write_frame(run_dir / f"new_{i}.fits", seed=i + 1)

    thread = threading.Thread(target=feeder)
    thread.start()
    try:
        collected = live.watch(
            tmp_path, poll_s=0.01, timeout_s=1.0, settle_s=0.0, show=False,
            include_existing=False, on_frame=seen.append,
        )
    finally:
        thread.join()

    # ``before.fits`` predates the watch and must not be replayed.
    names = sorted(s.path.name for s in collected)
    assert names == ["new_0.fits", "new_1.fits", "new_2.fits"], names
    assert len(seen) == 3


def test_include_existing_replays_the_backlog(tmp_path):
    _write_frame(tmp_path / "a.fits")
    _write_frame(tmp_path / "b.fits", seed=1)
    collected = live.watch(
        tmp_path, poll_s=0.01, timeout_s=0.2, settle_s=0.0, show=False,
        include_existing=True,
    )
    assert sorted(s.path.name for s in collected) == ["a.fits", "b.fits"]


def test_frames_still_being_written_are_not_read(tmp_path):
    """A half-written FITS must be skipped, not reported as a corrupt frame."""
    _write_frame(tmp_path / "fresh.fits")
    got = list(
        live.iter_new_frames(
            tmp_path, poll_s=0.01, timeout_s=0.1,
            settle_s=3600.0, include_existing=True,
        )
    )
    assert got == []


def test_unreadable_frame_does_not_stop_the_watch(tmp_path, capsys):
    _write_frame(tmp_path / "good.fits")
    (tmp_path / "bad.fits").write_bytes(b"not a FITS file")
    collected = live.watch(
        tmp_path, poll_s=0.01, timeout_s=0.2, settle_s=0.0, show=False,
        include_existing=True,
    )
    assert [s.path.name for s in collected] == ["good.fits"]
    assert "unreadable" in capsys.readouterr().out


# --------------------------------------------------------------------------- #
# Roll-ups and gate wiring
# --------------------------------------------------------------------------- #
def test_session_table_counts_by_type_exposure_filter(tmp_path):
    for i in range(3):
        _write_frame(tmp_path / f"d{i}.fits", seed=i)
    _write_frame(tmp_path / "flat.fits", cards={"IMAGETYP": "FLAT", "EXPTIME": 0.5,
                                                "FILTER": "Photometric V"})
    counts = live.session_table(tmp_path)
    assert counts[("DARK", 5.0, "Dark")] == 3
    assert counts[("FLAT", 0.5, "Photometric V")] == 1


def test_hwp_coverage_groups_light_frames_only(tmp_path):
    for angle in (0.0, 22.5, 45.0):
        _write_frame(
            tmp_path / f"pol_{angle}.fits",
            cards={"IMAGETYP": "LIGHT", "HWPANG": angle,
                   "POLSEQ": "polV16", "OBJECT": "HD 154445"},
        )
    # A dark taken mid-sequence must not count toward angle coverage.
    _write_frame(tmp_path / "dark.fits", cards={"HWPANG": 90.0})
    coverage = live.hwp_coverage(tmp_path)
    assert list(coverage.values()) == [[0.0, 22.5, 45.0]]


def test_as_paths_accepts_directory_or_sequence(tmp_path):
    a = _write_frame(tmp_path / "a.fits")
    b = _write_frame(tmp_path / "sub" / "b.fits", seed=1)
    assert sorted(live._as_paths(tmp_path)) == sorted([str(a), str(b)])
    assert live._as_paths([a]) == [str(a)]
    assert live._as_paths(a) == [str(a)]


def test_group_api_selects_groups_and_plots_header_series(tmp_path):
    for filt in ("Photometric B", "Photometric V"):
        for exp in (1.0, 2.0):
            for repeat in range(2):
                _write_frame(
                    tmp_path / f"{filt[-1]}_{exp}_{repeat}.fits",
                    level=1000.0 * exp,
                    cards={"IMAGETYP": "FLAT", "FILTER": filt, "EXPTIME": exp,
                           "DATE-OBS": f"2026-07-17T04:0{repeat}:00"},
                    seed=repeat,
                )

    chosen = live.select(tmp_path, imagetyp="flat", filter_name=" photometric v ")
    assert len(chosen) == 4
    assert len(live.select(tmp_path, exptime=lambda value: value > 1.0)) == 4
    with pytest.raises(ValueError, match="unknown filter"):
        live.select(tmp_path, filtre="V")

    rows = live.group_table(chosen, by=("filter_name", "exptime"), show=False)
    assert [(row.key, row.n) for row in rows] == [(("Photometric V", 1.0), 2), (("Photometric V", 2.0), 2)]
    fig, ax = live.trend(chosen, x="exptime", y="median", by="filter_name")
    assert len(ax.get_lines()) >= 1
    fig.clf()


# --------------------------------------------------------------------------- #
# Import hygiene
# --------------------------------------------------------------------------- #
def test_importing_live_pulls_in_no_hardware_or_plotting():
    """Notebook cells import this on machines with no camera driver installed."""
    code = '''
import sys
import obs_utils.live  # noqa: F401

for module in ("obs_utils.imaging", "obs_utils.interactive", "matplotlib"):
    assert module not in sys.modules, module
'''
    result = subprocess.run(
        [sys.executable, "-c", code], cwd=ROOT, text=True, capture_output=True,
    )
    assert result.returncode == 0, result.stderr
