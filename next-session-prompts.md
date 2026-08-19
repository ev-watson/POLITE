# Next-session prompts

## TOP PROMPT — Commission PlateSolve3 read-only on the observatory PC

**Context:** The 2026-07-30 drift-aware tracking overlays have been reviewed
by the user for all seven sequences: each selects the intended physical A/B
pair. The resulting calibration and tracking helpers now live in `caltools`
and `poltools`, and the reduction notebook uses them. `obs_utils.platesolve`
is a new read-only adapter for PlaneWave's local PS3CLI executable; the
capture template contains the first commissioning cells. Its synthetic-FITS
test proves unchanged pixels reach a simulated solver and its raw numeric
result is returned, but **does not prove that PS3CLI can catalogue-solve a
POLITE Savart-doubled field**. Tests: 279 passed, 2 existing DAO warnings.

**What to do:**

1. On observatory startup, use the PWI4 GUI and then the query-only check in
   `notebooks/templates/capture.ipynb` to confirm the intended pointing model
   is active: print its filename, enabled/total point counts, and RMS. Creating
   the PWI4 client does not change the model; only `startup_observatory()`
   explicitly loads its configured model filename.
2. On the host that actually has PS3CLI and the Kepler catalogue, set
   `PS3CLI_EXE` and `PS3_CATALOG` in the template. Select one existing,
   rich-star-field raw FITS and its measured plate scale, then run only the
   **PlateSolve3 proof** cells.
3. Record the PWI4 J2000 values before/after and preserve a sanitized copy of
   the actual PS3CLI raw `key=value` fields. Do not slew, offset, capture,
   create/clear/load/save a PWI4 model, or modify the raw FITS.
4. If PS3CLI fails on the doubled field, record its exact output. Do not
   automatically alter the image; decide from evidence whether to try an
   unsplit field or a reviewed synthetic-single-beam/source-list path.
5. Do not reconnect `map_point` or `build_pointing_model` until the real
   PS3CLI field names and units are reviewed against PWI4's model-point API.

**Success criterion:** a read-only real PS3CLI result (or an exact,
reproducible failure) is recorded with PWI4 state unchanged, and its coordinate
keys/units are known before any pointing-model design resumes.

**Files:** `notebooks/templates/capture.ipynb`, `obs_utils/platesolve.py`,
`obs_utils/pointing.py`, `tests/test_platesolve.py`.

---

## REVIEWED — 2026-07-30 drift-aware pair tracking

**Context:** `poltools` now automatically detects Savart pairs in every
2026-07-30 frame, estimates their adjacent-frame translation, and tracks a
single canonical A/B pair. The HWP cycle exposed an edge case: the first
frame's brightest proposed pair is at x≈4823 and moves to x≈5239 in the next
frame, too close to the 5252-px edge for a usable dual-beam detection. The
initializer now selects a supported predecessor for the next frame's bright
pair instead. Raw-frame pair coordinates validate the first field-1 HWP step:
`(1640.9, 2913.8) → (2056.4, 2350.8)` with shift `(415.4, −564.0)` px and
1.21-px prediction RMS. **CONJECTURED** until the user inspects overlays on
the corrected frames. Tests: 271 passed, 2 pre-existing DAO no-detection
warnings. No FITS frames were modified.

**What to do:**

1. Reload `notebooks/reductions/reduction_20260730.ipynb` from disk. Restart
   its kernel or re-run its autoreload/import cell, then re-run the automatic
   tracking cells (`iter_tracked_pairs`, burst photometry, and
   `exploratory_cycle`).
2. For each burst, run `show_tracked_sequence(block)`; for each HWP cycle,
   run `exploratory_cycle(target)` and then
   `show_tracked_sequence('optional_same_pair_cycle', target=target)`.
3. Visually verify that green rings stay on the same physical pair and that
   the printed shifts/RMS are finite. Do not relax the 12-px translation or
   15-px match tolerances to force a result; preserve any fail-closed error
   and inspect its overlay/candidate diagnostics first.
4. Only after those checks, assess flux continuity and the empirical burst
   scatter. Do not interpret q/u beyond the notebook's stated exploratory
   status.

**Success criterion:** every selected sequence has a user-reviewed overlay
showing one physical A/B pair throughout, or a recorded fail-closed reason
why it cannot be reduced.

**Files:** `poltools/photometry.py`, `poltools/__init__.py`,
`tests/test_photometry.py`, `notebooks/reductions/reduction_20260730.ipynb`.

---

## STILL PENDING — Commit the streamline work

**Context:** Three sessions of "Streamline & Ship" work is complete but
**nothing is committed**. 2026-07-26 trimmed `scripts/` to a generic set (one
runner, `execute_night.py`; `plan_night.py` preview-only). 2026-07-28 added HWP
*and* mount support to the runner, promoted the shared gates to
`obs_utils/night_safety.py`, shipped the `obs_utils/live.py` +
`notebooks/templates/` control-notebook family, and cleared **all** of
`CLEANUP.md` §5 P2–P4 plus Amendment A.1, A.2 and A.3. Tests: **230 passed, 0
failed** — the `arrhenius_A` bug is fixed (it was real: the plot read Arrhenius
coefficients the analysis only emits on an opt-in, converged fit).

2026-07-29 added the **FITS header provenance rule**: a card records as-acquired
state only, so `BEAMSEP`, `BEAMPA`, `SAVMAT`, `SAVTHK`, `RETARD`, `POLEFF` and
`WAVELEN` are no longer written and beam geometry lives only in the per-session
`pol_config.yaml` sidecar (which can flag it uncharacterized). `WPUNCERT` is kept
by owner decision — the open-loop Optec rotator steps discretely, so it is the
uncertainty on `HWPANG` in the same frame. Rule + the `BEAMSEP` case are in
`docs/polarimetry/06_report.md` §"FITS header provenance" and
`03_design_decisions.md` Q7.

Same day, the **config** half was closed (`03_design_decisions.md` **Q8**): the 60 px
beam-separation placeholder is scrubbed project-wide, and because it was also the
*loud* failure (~75 % low ⇒ pairing failed outright) while the only defensible
default — the 0.9 mm spec, ≈239.4 px vs 238.4 px measured — makes pairing **succeed
and look right**, the reduction is now fail-closed instead.
`nominal_beam_separation_px()` is the only default shipped;
`polconfig_from_fits_headers` has no geometry argument; measurements enter only via
`PolConfig.with_beam_geometry()`; pre-2026-07-29 sidecars are scrubbed on load
(`FITSDATA/` is read-only); `reduce_to_stokes(detect=True)` raises unless
`allow_uncharacterized_geometry=True`; `qa_lib` passes that flag and labels its WARN.
Tests: **240 passed, 0 failed.**

**`caltools`/`poltools` are now vendored POLITE source** (owner decision
2026-07-28 — they are tools most groups already have in better-supported form,
so they ship inside POLITE rather than standing alone). Their nested `.git/`
dirs live at `~/archive/{caltools,poltools}.git`; **do not delete those** until
the AGPL rewrite is pushed — they are the only copy of the MIT history.

**What to do:**
1. **Review and commit.** `git status` shows three sessions of work, with
   deletions and the new `docs/` tree already staged. `git add -A` is **now
   safe** — the gitlink trap is disarmed — but still verify with
   `git ls-files -s caltools poltools` (mode `100644` only) before any push, and
   read the diff first. Suggested split: (a) `night_safety` + execute_night
   HWP/mount + tests, (b) `live.py` + templates +
   `caltools.conversion_gain_from_flat_pair` + the `arrhenius_A` fix + tests,
   (c) docs, `.gitignore`, `docs/`, deletions, (d) the FITS header provenance
   change (`fits_writer.py`, `poltools/{io,pol_config,simulate}.py`,
   `obs_utils/{night_plan,night_session,pol_config}.py`, three test files, the
   polarimetry docs + notebook sources), (e) the beam-geometry scrub
   (`poltools/{_types,pol_config,pipeline}.py`, `poltools/README.md`,
   `obs_utils/qa_lib.py`, `tests/{test_per_filter,test_simulate_io}.py`, both
   notebooks, `docs/polarimetry/{03,06,08}*`, `.cursor/poltools-memory/`).
   One follow-up, low priority: `notebooks/polarimetry_simulation.ipynb` sources
   are correct but its **stored outputs are stale** — they still show the retired
   cards *and* the old 60 px split (the crop is now 512×700, since a 239 px pair
   does not fit in 512). The field was verified out-of-tree at the nominal split
   (16 beams → 8 clean pairs, nothing off-frame), but re-running rewrites
   `FITSDATA/SIM_20260608/` — owner's call, since `FITSDATA/` is otherwise never
   touched. `obs_utils/session_context.py::filter_wavelength_nm` is now unused (it
   fed `WAVELEN`); left in place as a config accessor for the per-band work.
2. **Two owner actions carried over — neither is a session’s to take alone:**
   - Archive or delete `github.com/ev-watson/{caltools,poltools}`. See
     `RELICENSE-AGPLv3.md` **Step 5** (rewritten as a takedown, not a rewrite):
     archive is reversible and keeps provenance, delete is permanent. Both are
     on `main` and fully pushed, so nothing local is at risk either way.
   - `rm -rf generated/salvage_first_light_20260709` (~331 MB, regenerable from
     `FITSDATA/20260709` via `reduction_20260709.ipynb`). **Only `generated/`**
     — `tmpimg/0709_*` and `paperfigs/` are deliberately kept. **Never** touch
     raw `FITSDATA/`.

**Success criterion:** the streamline work is committed on `main` with
`caltools`/`poltools` tracked as regular files (mode `100644`).

**Files:** `RELICENSE-AGPLv3.md` (Amendment A + Steps 3–6), `CLEANUP.md` §5,
`.gitignore`.

---

## STILL PENDING — Reduce the 2026-07-17 calibration night

**The acquisition happened.** `FITSDATA/20260717/` holds 204 frames across three
darkcal blocks (`darkcal_T+0C_1932`, `darkcal_T-10C_2039`, `darkcal_T-14C_2110`)
and two twilight-PTC blocks (`ptc_twilight_T-15C_2015`, plus one more). Do **not**
re-acquire.

**Blocker first:** `notebooks/reductions/reduction_20260717.ipynb` discovers the
right files in code, but 25 of its **saved outputs are copied from the July-9
reduction** — they report `gain 0` and carry `0709_` figure names.
**ESTABLISHED** by inspecting the stored outputs. Clear and re-run the notebook,
or keep it source-only and write reviewed products to a dated `generated/`
subtree. Until that is done, nothing in its output is July-17 science.

**What to do:**
1. Dark-vs-T: mean dark vs exposure per setpoint → slope vs T (Widenhorn 2002
   style), RON/bias per setpoint, hot-pixel census. Header-verify every value
   against DET-TEMP / gain / offset / Mode, per frame.
2. PTC pairs (Janesick) → e⁻/ADU + full-well, if the twilight frames are usable.
   `caltools.conversion_gain_from_flat_pair` is the quick single-point check;
   `caltools.photon_transfer_curve` over the ladder is the result that counts.
3. Follow `notebooks/reductions/reduction_20260709.ipynb` conventions
   (ADU→e⁻ once gain lands).

**Operating point as acquired:** gain 30, offset 50, Mode 0. Gain 30 is QHY's
vendor-stated QHY268M unity-gain setting (**CONJECTURED** until this PTC settles
it). Do not correct the pedestal by tuning; remove it with the matching bias /
master bias frames in reduction.

> **The project default has since changed (2026-07-29) to Mode 5, gain 56, offset
> 20** — QHY's lowest-read-noise QHY268M mode at the lowest gain reaching that
> floor, chosen to preserve dynamic range (a first-order polarimetric constraint;
> see `CLAUDE.md` → "Detector operating point"). Offset 20 targets a ~100 ADU bias
> level, which `run_bias_qa`'s `mean_hi=500` default is sized for. **None of this
> applies to the frames being reduced here** — they are Mode 0 / gain 30 / offset 50
> (~811 ADU) and must be reduced as such. A PTC gain measured here is valid only at
> that setting; the new operating point needs its own PTC.

**Success criterion:** a significant dark-current slope (or an explicit upper
limit) at ≥3 temperatures with a doubling-temperature fit where measurable,
RON/bias per setpoint, and — if twilight cooperated — a PTC gain + full-well.
All values header-verified.

**Files:** `notebooks/reductions/reduction_20260717.ipynb`,
`night_plans/20260717_darkcal.yaml`, `night_plans/20260717_ptc_twilight.yaml`,
`notebooks/templates/cal_night.ipynb` §8, data in
`FITSDATA/20260717/<block>_<HHMM>/`.

---

## DONE

- **2026-08-18 — 0730 calibration/tracking tool promotion + PlateSolve3
  commissioning preparation (not committed).** User reviewed the tracked
  target in all seven drift sequences; the selected A/B pair is correct in
  each. Promoted `caltools.subtract_bias_and_dark` (explicitly requires a
  bias-subtracted master dark), `poltools` compact-source anchoring,
  Savart-pair proposal, generic pair tracking, and upper-left diagnostic
  overlays. `reduction_20260730.ipynb` now uses these package tools and calls
  calibrated pixels `data`; its special first-two-frame edge recovery remains
  local to 0730. Added a read-only PS3CLI adapter and capture-template proof;
  pointing-model mutation functions are hard-disabled pending a real result.
  A synthetic multi-source FITS adapter test verifies that the simulated solver
  receives unchanged pixels and the raw result is parsed; it is **not** a
  catalogue-match test. `XORGSUBF`/`YORGSUBF` now preserve camera ROI origins
  in raw FITS headers. Tests: **279 passed, 2 existing DAO warnings**. Files:
  `caltools/{stacking.py,__init__.py}`, `poltools/{photometry.py,__init__.py}`,
  `obs_utils/{platesolve.py,pointing.py}`, `notebooks/{reductions/reduction_20260730.ipynb,templates/capture.ipynb}`,
  `alpyca_tools/fits_writer.py`, and associated tests.

- **2026-08-10 — 0730 automatic pair tracking (not committed; user overlay
  review pending).** Added `track_matched_pair`, which requires a supported
  constellation translation to continue the selected canonical A/B pair, and
  `select_trackable_pair`, which prevents an edge-bound initial bright pair
  from starting a sequence. The 0730 notebook re-detects each frame, tracks
  bursts/cycles in capture order, logs translation/RMS, and supplies
  `show_tracked_sequence(...)` overlays. Real raw frame-19→20 field-1 HWP
  candidate lists give `(415.4, −564.0)` px and 1.21-px RMS for the chosen
  persistent pair. **CONJECTURED** until corrected-frame overlays are
  inspected. Tests: 271 passed, 2 existing DAO warnings. Files:
  `poltools/{photometry.py,__init__.py}`, `tests/test_photometry.py`,
  `notebooks/reductions/reduction_20260730.ipynb`.

- **2026-07-29** — FITS header provenance + beam-geometry scrub (not committed;
  awaiting review). **Rule:** a card records *as-acquired state* only, so `BEAMSEP`,
  `BEAMPA`, `SAVMAT`, `SAVTHK`, `RETARD`, `POLEFF`, `WAVELEN` are retired;
  `WPUNCERT` kept by owner call (it qualifies `HWPANG` in the same frame — the
  open-loop Pyxis Gen3 never lands on a round angle). **Config half (`Q8`):** the
  60 px placeholder is scrubbed project-wide; `nominal_beam_separation_px()` is the
  only default; `polconfig_from_fits_headers` has no geometry argument;
  `PolConfig.with_beam_geometry()` is the sole validated way in; pre-2026-07-29
  sidecars are scrubbed on load; `reduce_to_stokes(detect=True)` is fail-closed
  (`allow_uncharacterized_geometry=True` = diagnostic), because the nominal is ~1 px
  from the measured 238.4 px and would otherwise pair *successfully* where 60 px
  failed loudly. Fixed a latent `AttributeError` in
  `notebooks/templates/polarimetry_night.ipynb` §5 (`cfg.beam_separation_px`).
  Sim notebook crop grew to 512×700 (a 239 px pair does not fit in 512), verified
  out-of-tree: 16 beams → 8 clean pairs, nothing off-frame. Files:
  `alpyca_tools/fits_writer.py`, `poltools/{_types,pol_config,pipeline,io,simulate}.py`,
  `poltools/README.md`, `obs_utils/{night_plan,night_session,pol_config,qa_lib}.py`,
  `tests/{test_simulate_io,test_per_filter,...}.py`, both notebooks,
  `docs/polarimetry/{03,06,08}*`, `.cursor/poltools-memory/`, `AGENTS.md`.
  Tests: **240 passed, 0 failed.**

- **2026-07-26** — Repo streamline toward shippable (not committed; awaiting
  review). Deleted `manual.md` + salvage/cal one-off scripts (`analyze_salvage_*`,
  `reduce_salvage_drift_sequence`, `sequence_audit.py`, their tests); the
  sequence-audit logic stays in `qa_lib`/`qa_gates` for the control notebook.
  Renamed `run_calibration_night.py` → generic `scripts/execute_night.py` (all
  safety machinery intact); `plan_night.py` is now preview-only (`--run` removed).
  Fixed every dangling ref (README, scripts/README rewrite, all `night_plans/*.yaml`
  headers). Rewrote `CLEANUP.md` into a "Streamline & Ship Plan". Tests: 171
  passed, 1 pre-existing caltools failure (`arrhenius_A`, reproduces at HEAD).

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
