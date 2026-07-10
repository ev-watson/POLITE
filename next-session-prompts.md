# Next-session prompts

## TOP PROMPT — Assess salvage-night results (or run it if not yet executed)

**Context:** DEC drive is dead (engaging DEC auto-disconnects the mount → no
slew/home/point). We salvaged 2026-07-09/10 as qualitative *instrument*
commissioning. All code + the salvage plan are on `origin/main`. The no-mount
runner is `scripts/run_salvage_night.py`; it connects only camera+EFW+HWP and
reuses the tested capture path. `plan_night.py --run` must NOT be used (it homes
the mount). EFW order is now Clear,B,V,R,Dark in both the Python config and the
ASCOM Remote driver.

**Pre-flight (server moved + DATE-OBS fix, 2026-07-10):** The QHY Alpaca server
is now `qhy_alpaca/` (was `third_party/alpaca-qhyccd-camera/`). On the observatory
PC: `git pull`, relaunch via `scripts/start_qhy_alpaca_server.ps1`, then capture
one frame and confirm `DATE-OBS` is the current UTC (not 1995-10-09) and
`TIME-SRC=SYSCLOCK`. `qhy_alpaca/config.windows.yaml` sets `has_gps: false`
(QHY268M has no GPS module) — do not enable unless a real receiver is installed.
Also `pip install rich` in the observatory POLITE env for the new night display
(optional; degrades to plain logging without it).

**What to do:**
1. If the salvage night has NOT run yet, on the observatory PC:
   `git pull` →
   `scripts/run_salvage_night.py night_plans/20260709_salvage.yaml` (dry-run) →
   `... --run`. Console-tune `exp`/`n` for `polV8_drift` + `superflatV_drift` in
   `night_plans/20260709_salvage.yaml` (keep the two exposures EQUAL; the field
   must visibly drift between frames; trail < ~2 px/frame).
2. If it HAS run, pull the FITS from `FITSDATA/20260709/` and assess the three
   qualitative questions: (a) is HWP modulation present in the beam pair? (b) is
   the beam pair trackable frame-by-frame as it drifts? (c) does the
   detector-frame Stokes vector transform correctly under the +45° whole-
   polarimeter rotation (compare the `driftA_polV8_rot45` set vs `driftA_polV8`)?
3. Build the night-sky super-flat: median/sigma-clip-combine the
   `superflatV_drift` frames PER HWP ANGLE; confirm it captures the two-beam
   Savart geometry + relative throughput.

**Success criterion:** A yes/no answer to each of the three qualitative questions
with supporting frames, and a usable per-HWP-angle super-flat. DEC failure and
all metadata documented per `salvage_no_pointing_checklist.md`.

**Files involved:** `scripts/run_salvage_night.py`,
`night_plans/20260709_salvage.yaml`, `salvage_no_pointing_checklist.md`,
`FITSDATA/20260709/`, `obs_utils/interactive.py`, `poltools/` (reduction),
`notebooks/reduction.ipynb`.

---

## DONE

- **2026-07-10** — Tooling session (committed+pushed, `024db68`). (1) Un-vendored
  the QHY Alpaca server `third_party/alpaca-qhyccd-camera/` → `qhy_alpaca/` and
  fixed the DATE-OBS 1990s bug (GPS header misparse on GPS-less QHY268M) via
  `defaults.has_gps` gate + `gps_max_clock_skew_s` sanity window. (2) Made the QA
  pipeline outlier-robust (sigma-clipped stats) and non-blocking (PASS/WARN/FAIL
  tiers in `obs_utils/qa_lib.py`; WARN keeps capturing); added
  `tests/test_qa_lib.py`. (3) Added `obs_utils/night_display.py` rich NightReporter
  (live progress bar, banner, colored QA) threaded through night + salvage runners.
  125 tests pass. Server not yet import/hardware-tested (pydantic/libqhyccd on obs PC).

- **2026-07-10** — Salvage-night prep for dead-DEC commissioning. Built no-mount
  runner `scripts/run_salvage_night.py`; fixed EFW `filter_names` order
  (Clear,B,V,R,Dark) in `config.py`+`user_config.py`; replaced impossible
  illuminated flats with a night-sky super-flat drift block in
  `night_plans/20260709_salvage.yaml`; fixed `FilterWheelState %d` logging crash
  in `scripts/observatory_smoke_test.py`; tracked notebooks in-repo. All pushed
  to origin/main (through commit a06d08d).
