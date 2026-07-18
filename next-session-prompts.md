# Next-session prompts

## TOP PROMPT — Execute the 2026-07-17 calibration night (obs PC), then reduce

**Context:** Cloudy night, DEC dead (no slew/home; motors can't even engage;
manual tube moves don't hold). Prepared on the lab MacBook 2026-07-17: a
NO-MOUNT calibration runner + two plans + checklist targeting the
first-light-publish gaps (dark-vs-temperature; opportunistic PTC gain/full-well
from cloudy twilight). **Must be committed+pushed before the obs PC can pull.**

**What to do (observatory PC):**
1. `git pull`, then follow `20260717_calibration_night_checklist.md` top to
   bottom (pre-flight → dry-runs → darkcal @ 0/−10/−20 °C → PTC twilight
   sweeps ×3 in the 19:50–20:40 PDT window).
2. Runner: `scripts/run_calibration_night.py <plan> --run [--setpoint T]`.
   It prints+logs every setting read back from hardware and gates on cooler
   stabilization (continue-at-achieved-T is fine; CCD-TEMP per frame is truth).
   NEVER `plan_night.py --run`.
3. After the run: reduce. Dark-vs-T fit (Widenhorn 2002 style: mean dark vs
   exposure per setpoint → slope vs T), RON/bias vs T, hot-pixel census; PTC
   pairs (Janesick) → e⁻/ADU + full-well if twilight frames usable. Extend
   `notebooks/reductions/reduction_20260709.ipynb` conventions (ADU→e⁻ once
   gain lands).

**Operating point:** gain 30 is QHY's vendor-stated QHY268M unity-gain setting
(**CONJECTURED** until the PTC); fixed offset 50. Do not spend the night on
histogram offset-tuning: remove the captured pedestal using matching bias/master
bias frames in reduction.

**Success criterion:** a significant dark-current slope (or an explicit upper
limit) at ≥3 temperatures with a doubling-temperature fit where measurable,
RON/bias per setpoint, and — if twilight cooperated — a PTC gain + full-well.
All values header-verified (CCD-TEMP, gain 30, offset 50, Mode 0).

**Files:** `scripts/run_calibration_night.py`,
`night_plans/20260717_darkcal.yaml`, `night_plans/20260717_ptc_twilight.yaml`,
`20260717_calibration_night_checklist.md`, `scripts/quick_unity_gain.py`,
`obs_utils/qa_gates.py` (bias_qa dispatch fix), data →
`FITSDATA/20260717/<block>_<HHMM>/`.

---

## DONE

- **2026-07-17** — Calibration-night prep (lab MacBook, at site). Built
  `scripts/run_calibration_night.py` (no-mount, camera+EFW only: settings
  banner with hardware read-back to screen+log, fail-closed gain/offset/readout
  check, read-only cooler stabilization gate after one setpoint command +
  achieved-T fallback,
  per-invocation output subdirs). Plans: `20260717_darkcal.yaml` (bias×25 +
  dark ladder 5/60/150 s ×5, all through opaque Dark slot — no shutter!),
  `20260717_ptc_twilight.yaml` (overcast-twilight PTC exposure ladder + trailing
  bias). Fixed `dispatch_qa_gate` bias_qa call (passed bare Path to a
  Sequence-taking handler → TypeError; also missed `calibrations/` subdir) and
  made QA dispatch exception-proof. 154 tests pass. Twilight times computed for
  site (sunset 19:49 PDT, civil 20:21, naut 20:55).

- **2026-07-14** — Salvage assessment completed (see memory
  `salvage-reduction-20260709`): reduction notebook + verdicts (beam geometry
  ESTABLISHED, modulation INCONCLUSIVE, flat FAILED); rebuilt
  `first_light_polarimetry_characterization_report.md` with 14 verified refs.

- **2026-07-10** — Tooling session (committed+pushed, `024db68`). (1) Un-vendored
  the QHY Alpaca server `third_party/alpaca-qhyccd-camera/` → `qhy_alpaca/` and
  fixed the DATE-OBS 1990s bug (GPS header misparse on GPS-less QHY268M) via
  `defaults.has_gps` gate + `gps_max_clock_skew_s` sanity window. (2) Made the QA
  pipeline outlier-robust (sigma-clipped stats) and non-blocking (PASS/WARN/FAIL
  tiers in `obs_utils/qa_lib.py`; WARN keeps capturing); added
  `tests/test_qa_lib.py`. (3) Added `obs_utils/night_display.py` rich NightReporter
  (live progress bar, banner, colored QA) threaded through night + salvage runners.

- **2026-07-10** — Salvage-night prep for dead-DEC commissioning. Built no-mount
  runner `scripts/run_salvage_night.py` (removed in `c4012b9` cleanup; recover
  from `024db68` if needed); fixed EFW `filter_names` order (Clear,B,V,R,Dark)
  in `config.py`+`user_config.py`; night-sky super-flat drift block in
  `night_plans/20260709_salvage.yaml`. All pushed through a06d08d.
