# POLITE First Light — Software Implementation Plan

**Document purpose:** Close every gap between the *First Light Observation Plan*
(`first_light_obs_plan.pdf` / `first_light_obs_plan.html`, night of **2026-07-09**)
and the current POLITE capture/reduction stack. This is an engineering plan for
developers and operators: what is broken or missing today, what must be built, in
what order, with acceptance criteria and file-level touch points.

**Status:** Draft — 2026-07-08  
**Audience:** Observatory operators running first light; developers extending
`obs_utils`, `alpyca_tools`, `poltools`, and `caltools`.

---

## Table of contents

1. [Executive summary](#1-executive-summary)
2. [Source requirements (from the observation plan)](#2-source-requirements-from-the-observation-plan)
3. [Current architecture snapshot](#3-current-architecture-snapshot)
4. [Complete gap inventory](#4-complete-gap-inventory)
5. [FITS header specification (target state)](#5-fits-header-specification-target-state)
6. [Phase 1 — FITS provenance and PolConfig sidecar](#6-phase-1--fits-provenance-and-polconfig-sidecar)
7. [Phase 2 — First-light night plan and data layout](#7-phase-2--first-light-night-plan-and-data-layout)
8. [Phase 3 — Block manifest and environmental logging](#8-phase-3--block-manifest-and-environmental-logging)
9. [Phase 4 — Inline QA gates and end-of-night audits](#9-phase-4--inline-qa-gates-and-end-of-night-audits)
10. [Phase 5 — Characterization pipelines](#10-phase-5--characterization-pipelines)
11. [Phase 6 — Exposure calculator and long-term polish](#11-phase-6--exposure-calculator-and-long-term-polish)
12. [First-light manual fallbacks](#12-first-light-manual-fallbacks)
13. [Testing and verification matrix](#13-testing-and-verification-matrix)
14. [File touch map and dependency graph](#14-file-touch-map-and-dependency-graph)
15. [Priority schedule relative to 2026-07-09](#15-priority-schedule-relative-to-2026-07-09)
16. [Open questions and decisions](#16-open-questions-and-decisions)

---

## 1. Executive summary

The first-light observation plan is scientifically complete: targets, timing,
calibration protocol, QA gates, and contingency rules are all specified. The
**software stack is not yet aligned** with what the plan assumes an operator can
rely on at the telescope.

The single most consequential gap is **detector and polarimetry provenance in FITS
headers**. The plan (§9 Data management) explicitly states:

> *The capture software writes no GAIN keyword, so record `readout_mode`,
> `gain_setting`, offset, and sensor temperature in the night log and in
> `PolConfig` for every block — measured gain/RON are valid for exactly one
> mode/gain/offset combination.*

In practice:

- `alpyca_tools/fits_writer.py` **does** attempt to write `GAIN`, but that card
  holds the **Alpaca slider index** (e.g. `0`), not the **measured conversion
  gain** (~1.0 e?/ADU) that `poltools`, `caltools`, and the CMOS error model
  require.
- `poltools` simulated frames write **`EGAIN`** (e?/ADU); the live capture path
  does **not**.
- **`INSTROT`** (PWI4 field-rotator angle) is defined in the header schema but
  **never populated** during `run_night_session` — yet `poltools.pipeline` warns
  when it drifts across a sequence, and the §6.5 QA gate depends on rotator
  bookkeeping.
- There is **no `PolConfig` serialization** from capture; reduction must be
  hand-configured per night.
- The declarative night plan **`night_plans/20260709.yaml` does not exist**;
  data lands under `data/` with a different filename convention than the plan's
  `FITSDATA/20260709/2026-07-09_target_filter_exposure` scheme.
- **QA gates** (bias histogram, first polarized standard, flat lsq vs
  double_ratio, frame-count audit) are described in the plan but **not automated**.

**Recommended minimum before first light:** implement **Phase 1 entirely** and
the critical subset of **Phase 2** (the `20260709.yaml` plan, `FITSDATA` paths,
correct per-target exposures). Phases 3–4 can be partially manual on the first
night using the fallbacks in §12, but FITS provenance is not optional for
defensible polarimetry reduction.

---

## 2. Source requirements (from the observation plan)

This section maps observation-plan sections to software obligations. Cross-reference
`first_light_obs_plan.html` while implementing.

### 2.1 Goals (plan header box)

| Goal | Software obligation|
|------|---------------------|
| Instrumental polarization (q?,u?) from 4 unpol standards | Complete HWP sequences per target; `HWPANG`, `POLSEQ`/`POLSEQN` on every frame; reduction with known `EGAIN`|
| EVPA zero-point from 4 pol standards | Same + `INSTROT` stable within sequence; reference catalog available to QA script|
| Polarimetric flats + lsq vs double_ratio validation | Flat capture brick; automated or scripted flat-quality gate (§5.3)|
| Asteroids Melpomene + Juno | `polV16` sequences; Horizons ephemeris at acquisition; phase angle + scattering-plane PA logged|
| Polarimetric efficiency P_meas/P_ref | QA script computes and logs per pol standard; benchmark 0.97–1.00|
| Detector verification (§5.4) | Bias QA, PTC ladder, headers sufficient for `caltools.gain`|
### 2.2 Timeline blocks (§4 Master timeline)

| PDT block | Plan activity | Current automation|
|-----------|---------------|-------------------|
| ~15:30–17:00 | Bench: `hwp_modulation_test.py hardware`, polarizer-film test, offset check | **Script exists** (`scripts/hwp_modulation_test.py`); not integrated into night runner|
| 18:40–19:30 | Startup checklist, cooler ?10 °C, HWP home | **Partial** — `startup_observatory()` + manual checklist; cooler setpoint not logged to FITS|
| 19:35–19:55 | Bias ×25 + offset/RON QA | Palette `bias` is n=20; **no bias QA hook**|
| 20:15–20:40 | Twilight polarimetric sky flats (CasPol) | **No flat brick**; archive script `night_session_20260302.py` has ad-hoc flat logic only|
| 21:00–03:40 | Eight standards + two asteroids, mixed exposures | **Bricks exist** (`polV8`, `polV16`, `polR8`) but fixed 30 s; plan needs 0.1–60 s range|
| 22:50 | QA gate after HD 154445 | **Manual** poltools reduction|
| 03:40–04:05 | Darks at 30 s + short exposure sets | `darks30` only; **no multi-exposure dark brick**|
| 04:50–05:45 | Dawn Option B: morning flats + PTC ladder | **No bricks**|
| End | Frame-count verification | **Not implemented**|
### 2.3 Data management (§9)

Explicit operator logging requirements that must become software artifacts:

1. All frames ? `FITSDATA/20260709/` with naming
   `2026-07-09_target_filter_exposure`.
2. Drive night from `night_plans/20260709.yaml` (`uses: palette.yaml`).
3. Structured logging on.
4. Per block log: UT start/end, HWP zero check, rotator angle, cooler temp,
   sky conditions, moon separation; for asteroids also phase angle +
   scattering-plane PA.
5. Gain-mode provenance: `readout_mode`, `gain_setting`, offset, sensor
   temperature ? night log **and** `PolConfig` per block.
6. End of night: verify frame counts per target/angle.

### 2.4 Calibration and QA rules

| Rule | Plan reference | Software|
|------|----------------|----------|
| LSQ requires flat-fielded frames | §5.2, project Q6 | Flats must be captured and tagged `IMAGETYP=FLAT` with per-angle masters|
| lsq vs double_ratio discrepancy > 0.1% ? bad flats | §5.3 | Gate script or inline hook|
| Bias histogram above zero, RON ? 3.5 e? | §5.4 Lab 1 | `bias_qa.py` using `EGAIN`|
| PTC gain within ~5% of 1.0 e?/ADU | §5.4 | Morning ladder + `caltools.gain`|
| POL 1: P within ~0.3%, PA within ~2° | §6.5 | `first_light_qa.py`|
| Efficiency outside 0.95–1.02 flags problem | §6.5 | Same QA script|
---

## 3. Current architecture snapshot

Understanding where data flows today prevents duplicate or divergent implementations.

### 3.1 Device and control paths

```
???????????????????????????????????????????????????????????????????????????
?  Night driver                                                           ?
?  scripts/plan_night.py  ?  obs_utils.night_plan.load_night_plan()       ?
?                         ?  obs_utils.night_session.run_night_session()  ?
???????????????????????????????????????????????????????????????????????????
         ?                              ?
         ?                              ?
???????????????????????    ????????????????????????????????????????????????
? PWI4 (:8220)        ?    ? INDIGO ? Alpaca (:11111)                     ?
? Mount slew/track    ?    ? QHY268M camera, ZWO EFW filter wheel         ?
? Field rotator       ?    ? (Optional: Alpaca rotator for Pyxis on Win)  ?
? ? should feed INSTROT?    ? Lab: Pyxis Gen3 via obs_utils.pyxis_gen3   ?
???????????????????????    ????????????????????????????????????????????????
         ?                              ?
         ????????????????????????????????
                        ?
         alpyca_tools.fits_writer.capture_fits()
         obs_utils.imaging.capture_fits_file()
                        ?
              FITS on disk (data/… or FITSDATA/…)
                        ?
         poltools / caltools reduction (manual or scripted)
```

**Key modules:**

| Module | Role|
|--------|------|
| `obs_utils/night_plan.py` | YAML bricks ? `FramePlan` / `TargetPlan` expansion|
| `obs_utils/night_session.py` | Slew, filter/HWP select, capture loop, header build|
| `alpyca_tools/fits_writer.py` | Primary FITS header writer for live capture|
| `obs_utils/fits_routine.py` | Parallel/legacy capture path; keep keyword sets in sync|
| `poltools/io.py` | `POL_KEYWORDS` registry; sim frame writer; sequence grouping|
| `poltools/_types.py` | `PolConfig` — reduction instrument model|
| `caltools/caltools/io.py` | `sensor_config_from_header()` — **requires external gain**|
| `obs_utils/logging.py` | Plain-text session log to `logs/YYYY-MM-DD/`|
| `obs_utils/timing.py` | NTP snapshot ? `TIMESRC`, `TIMEUNC`, `FWPOS`, `FILTRDY`|
### 3.2 Brick system (night_plan.py)

Supported brick `type` values today:

| Type | Expands to | FITS / behavior|
|------|------------|-----------------|
| `stack` | One `FramePlan` LIGHT stack | Single filter, fixed exposure|
| `pol_seq` | N `FramePlan`s, one per HWP angle | Sets `hwp_angle_deg`, `pol_seq_id`, `pol_seq_index` ? `HWPANG`, `POLSEQ`, `POLSEQN`|
| `filter_loop` | One stack per filter in list | Multiple filters, same exposure|
| `cal` | BIAS / DARK / (FLAT if frame type set) | `IMAGETYP` from `frame` field|
**Not supported today:** twilight polarimetric flats with antisolar pointing,
tracking off, dithering, exposure rescaling; multi-exposure dark sets; PTC ladder
pairs; inline QA gates; Horizons fetch at slew.

### 3.3 Filename and directory convention (current)

`run_night_session()` builds paths as:

```
{base_data_dir}/{YYYY-MM-DD}/{session_id}/
  targets/{target_slug}/LIGHT_{object}_f{filter}_exp{exposure}s_{index:03d}.fits
  calibrations/{frame_type}/...
```

Defaults: `base_data_dir = "data"`, session id from log file stem or timestamp.

**Plan convention:**

```
FITSDATA/20260709/
  2026-07-09_HD154445_Photometric_V_2.0s_001.fits
  ...
```

These are incompatible without explicit configuration and a naming mode.

### 3.4 FITS keywords written today (capture path)

From `alpyca_tools/fits_writer.build_header()`:

| Keyword | Written? | Source | Notes|
|---------|----------|--------|-------|
| `EXPTIME`, `EXPOSURE` | Yes | `camera.LastExposureDuration` ||
| `DATE-OBS` | Yes | `camera.LastExposureStartTime` | UTC via Alpaca|
| `TIMESRC`, `TIMEUNC`, `TIMEREF`, `NTPOFFS`, `NTPAGE` | If timing snapshot | `obs_utils.timing` | Once per session|
| `FWPOS`, `FILTRDY` | If filter selected | Filter wheel state ||
| `XBINNING`, `YBINNING` | Yes | Camera ||
| `IMAGETYP`, `OBSTYPE` | Yes | `FitsHeaderConfig.imagetyp` ||
| `OBJECT`, `OBSERVER`, `TELESCOP`, `OBSERVAT`, `INSTRUME` | If set | Config / camera ||
| `GAIN` | Try | `camera.Gain` | **Slider index, not e?/ADU**|
| `OFFSET`, `PEDESTAL` | Try | `camera.Offset` ||
| `CCD-TEMP` | Try | `camera.CCDTemperature` | Actual temp only|
| `FILTER` | If set | EFW slot name ||
| `HWPANG`, `RETARD`, `POLSEQ`, `POLSEQN` | If pol frame | `PolarimetryCards` | From `FramePlan`|
| `INSTROT` | **Only if set** | — | **Never set in night_session**|
| `POLBEAM`, `POLEFF`, `BEAMSEP`, `BEAMPA`, `SAVMAT`, `SAVTHK`, `WAVELEN` | If set | — | Defaults not filled from registry|
| `RA`, `DEC`, `HA`, `EQUINOX`, `AIRMASS` | Partial | PWI4 auto or target plan ||
| `EGAIN` | **No** | — | Required by poltools sim + error model|
| `READMODE` | **No** | — | Plan §9 provenance|
| `SET-TEMP` | **No** | — | Plan §5.1 cooler setpoint|
| `XPIXSZ`, `YPIXSZ`, `PIXSCALE` | **No** | — | Sim writes `XPIXSZ`; caltools may require pitch|
| `CHECKSUM`, `DATASUM` | Yes (default) | astropy ||
### 3.5 PolConfig vs capture (conceptual gap)

`poltools.PolConfig` is the reduction-side instrument model. Fields that **must**
match how data were taken:

```python
# poltools/_types.py (abridged)
readout_mode: str = "Mode0"      # FITS: not written
gain_setting: int = 0            # FITS: GAIN (slider) conflated with EGAIN
sensor.gain_e_per_adu: float     # FITS: EGAIN missing
read_noise_e: float              # From bias pairs × EGAIN; needs both
instrument_rotator_deg: float    # FITS: INSTROT missing
beam: BeamGeometry               # FITS: BEAMSEP/BEAMPA optional, usually empty
filter_name: str                 # FITS: FILTER — written
hwp_angles_deg: tuple            # Derived from sequence grouping on disk
```

There is **no code path** that builds `PolConfig` from a captured FITS set or
writes a session-level YAML that reduction loads automatically.

---

## 4. Complete gap inventory

Every item below is traceable to the first-light plan or to a hard dependency in
`poltools` / `caltools`. Severity: **P0** = blocks trustworthy first-light
reduction; **P1** = plan assumes it but manual workaround exists; **P2** =
post-first-light or nice-to-have.

### 4.1 FITS header and provenance

| ID | Gap | Severity | Plan / code ref|
|----|-----|----------|-----------------|
| F1 | `EGAIN` (e?/ADU) not written; `GAIN` is slider index only | **P0** | §9; `caltools.io` docstring; `poltools.io`|
| F2 | `READMODE` / readout mode string not in FITS | **P0** | §9; CMOS error model §1.5|
| F3 | `INSTROT` not populated from PWI4 at exposure time | **P0** | §6.5; `poltools.pipeline._warn_if_instrot_varies`|
| F4 | Cooler **setpoint** not recorded (`SET-TEMP`); only actual `CCD-TEMP` | **P1** | §5.1|
| F5 | `XPIXSZ`/`YPIXSZ` not written; `sensor_config_from_header` may raise | **P1** | §5.4 plate scale; `caltools/caltools/io.py`|
| F6 | `WAVELEN` not filled from filter registry | **P2** | `PolarimetryCards.eff_wavelength_nm` exists|
| F7 | `BEAMSEP`/`BEAMPA` not filled until measured | **P2** | §5.4; expected empty at first light|
| F8 | No per-block `PolConfig` sidecar | **P0** | §9 explicit|
| F9 | `fits_routine.py` and `fits_writer.py` can drift | **P1** | Two parallel header builders|
### 4.2 Night automation

| ID | Gap | Severity | Plan / code ref|
|----|-----|----------|-----------------|
| N1 | `night_plans/20260709.yaml` missing | **P0** | §9|
| N2 | `base_data_dir` defaults to `data/`, not `FITSDATA/` | **P1** | §9|
| N3 | Filename convention mismatch | **P1** | §9|
| N4 | No `pol_flat` / twilight flat brick | **P1** | §5.2, timeline 20:15–20:40|
| N5 | Palette exposures fixed at 30 s; plan needs 0.1–60 s | **P0** | §6.2 table|
| N6 | `bias` brick n=20; plan wants n=25 | **P1** | Timeline 19:35|
| N7 | No multi-exposure dark brick (30 s + 0.2/2/5 s sets) | **P1** | Timeline 03:40|
| N8 | No dawn PTC ladder / morning flat bricks | **P2** | §7 Option B|
| N9 | Catalog lacks first-light standard stars and asteroids | **P0** | §2 target tables|
| N10 | `calibrate_stage` for bias-before / darks-after needs `both` | **P1** | Timeline|
| N11 | Interactive notebook vs `plan_night.py --run` not unified | **P2** | Timeline startup|
| N12 | Antisolar flat pointing (alt/az) not in brick model | **P1** | §5.2|
### 4.3 Logging and environmental metadata

| ID | Gap | Severity | Plan / code ref|
|----|-----|----------|-----------------|
| L1 | No structured block manifest (JSONL/YAML) | **P1** | §9|
| L2 | HWP homing / zero-check not recorded structurally | **P1** | §9, timeline|
| L3 | Sky conditions (seeing, transparency) not captured | **P1** | §9|
| L4 | Moon separation not computed or logged | **P1** | §1, §7 Vesta note|
| L5 | Asteroid phase angle + scattering-plane PA not logged | **P1** | §6.4|
| L6 | Focus position and per-filter offsets not logged | **P2** | Timeline 20:40|
| L7 | Polarimetric efficiency not logged per pol standard | **P1** | §6.5|
| L8 | Plain-text log only (`obs_utils/logging.py`) | **P1** | §9 structured logging|
### 4.4 QA and verification

| ID | Gap | Severity | Plan / code ref|
|----|-----|----------|-----------------|
| Q1 | No automated first-light QA after HD 154445 | **P1** | §6.5|
| Q2 | No bias histogram / RON QA script | **P1** | §5.4, timeline 19:35|
| Q3 | No flat-quality gate (lsq vs double_ratio) | **P1** | §5.3|
| Q4 | No end-of-night HWP sequence completeness audit | **P1** | §9|
| Q5 | No Horizons fetch at asteroid acquisition | **P1** | §6.4|
| Q6 | Exposure guide marked CONJECTURED — no calculator | **P2** | §6.2 tag|
### 4.5 Reduction pipeline dependencies

| ID | Gap | Severity | Notes|
|----|-----|----------|-------|
| R1 | `reduce_to_stokes` needs hand-supplied `PolConfig` | **P0** | Tied to F1, F3, F8|
| R2 | `group_pol_sequence` requires `HWPANG` on every frame | **P0** | Already written for `pol_seq` bricks|
| R3 | LSQ path needs angle-matched polarimetric flats | **P1** | Tied to N4|
| R4 | Plate-scale / beam geometry analysis not wired post-capture | **P2** | §5.4 Lab 3|
---

## 5. FITS header specification (target state)

This section is the **authoritative target** for Phase 1. All capture paths must
converge on these definitions. Simulated frames (`poltools.io.write_pol_fits`) and
live frames must remain keyword-compatible.

### 5.1 Keyword registry (POLITE convention)

Extend the existing `poltools.io.POL_KEYWORDS` registry. New cards use 8-character
FITS keys where possible; longer names use `HIERARCH` via astropy.

#### 5.1.1 Detector and readout (new `DetectorCards` dataclass)

| Keyword | Type | Unit | Required | Comment|
|---------|------|------|----------|---------|
| `GAIN` | int | — | Yes | **Camera gain slider setting** (Alpaca `Gain`). Comment: `"QHY gain index, not e-/ADU"`.|
| `EGAIN` | float | e?/ADU | Yes | **Measured conversion gain** at this mode/gain/offset. Default 1.0 until PTC updates.|
| `READMODE` | int | — | Yes | Alpaca `ReadoutMode` index (0 = Mode 0 for QHY268M).|
| `HIERARCH READMODE NAME` or `RMODE` | str | — | Recommended | Human label, e.g. `"Mode 0"`.|
| `OFFSET` | int | ADU | Yes | Camera offset pedestal setting.|
| `CCD-TEMP` | float | °C | Yes | Actual sensor temperature at readout.|
| `SET-TEMP` | float | °C | Yes | Cooler setpoint (`SetCCDTemperature` target).|
| `XPIXSZ` | float | µm | Yes | 3.76 for QHY268M (from datasheet / bench).|
| `YPIXSZ` | float | µm | Yes | Same as `XPIXSZ` unless anamorphic.|
| `RON` | float | e? | Optional | Session-level read noise if measured at startup bias QA; else omit.|
**Critical distinction (document in header comments and operator docs):**

- `GAIN` = hardware setting (integer slider).
- `EGAIN` = physical conversion gain used in all electron-domain math.

`caltools.sensor_config_from_header()` must be updated to:

1. Read `EGAIN` if present.
2. Else read `GAIN` only if a `HIERARCH EGAIN SOURCE` card says it is e?/ADU
   (never true for QHY).
3. Else require explicit `gain=` argument and emit `warnings.warn`.

#### 5.1.2 Polarimetry (existing `PolarimetryCards` — populate consistently)

| Keyword | Required for pol frames | Source at capture|
|---------|-------------------------|-------------------|
| `HWPANG` | Yes | Achieved Pyxis PA after settle (`select_hwp_angle` return value)|
| `RETARD` | Yes | 180.0 (nominal HWP) unless measured otherwise|
| `INSTROT` | Yes | PWI4 field rotator angle at start of exposure|
| `POLBEAM` | Yes | `"dual"` (both Savart beams on single detector)|
| `POLSEQ` | Yes | Brick name or target+filter id, e.g. `"HD154445_polV8"`|
| `POLSEQN` | Yes | 0-based index in HWP sequence|
| `SAVMAT` | Yes | `"alpha-BBO"`|
| `SAVTHK` | Yes | `18.0`|
| `WAVELEN` | Recommended | From `FilterConfig.eff_wavelength_nm` registry|
| `BEAMSEP` | When known | From calibration YAML; omit at first light|
| `BEAMPA` | When known | From calibration YAML|
| `POLEFF` | Optional | Modulation efficiency if pre-characterized|
#### 5.1.3 Timing and filter wheel (already implemented)

| Keyword | Notes|
|---------|-------|
| `TIMESRC`, `TIMEUNC`, `TIMEREF`, `NTPOFFS`, `NTPAGE` | Session NTP snapshot via `stamp_timing_cards`|
| `FWPOS`, `FILTRDY` | Filter wheel index and readiness|
#### 5.1.4 Optional environmental (Phase 3)

| Keyword | Purpose|
|---------|---------|
| `MOONSEP` | Target–Moon separation [deg] at `DATE-OBS`|
| `PHASEANG` | Asteroid phase angle [deg] from Horizons|
| `SCATPA` | Scattering plane position angle [deg]|
| `SEEING` | Arcsec FWHM if measured|
| `TRANSPAR` | Transparency estimate (1–5 or percent)|
Use `extra_cards` or a dedicated `EnvironmentCards` dataclass; do not break
`poltools` readers that ignore unknown keys.

### 5.2 Header build order (implementation contract)

In `build_header()`, after Alpaca camera cards and before WCS:

1. Detector cards (`EGAIN`, `READMODE`, `SET-TEMP`, …) from `SessionCaptureContext`.
2. Polarimetry cards from `PolarimetryCards` (with `HWPANG` = achieved angle).
3. Query PWI4 **once per frame** (or cache if exposure batch is fast) for `INSTROT`.
4. Stamp timing + filter wheel state.
5. Merge `extra_cards` (environment, Horizons).

### 5.3 SessionCaptureContext (new shared object)

Introduce a dataclass carried through `run_night_session`:

```python
@dataclass
class SessionCaptureContext:
    """Immutable per-session detector settings + mutable measured values."""
    readout_mode: int = 0
    readout_mode_name: str = "Mode 0"
    gain_setting: int = 0
    offset_setting: int = 0          # from bench — operator config
    egain_e_per_adu: float = 1.0     # bench default; updated after PTC
    ron_e: Optional[float] = 3.5     # bench default; updated after bias QA
    cooler_setpoint_c: float = -10.0
    pixel_size_um: float = 3.76
    plate_scale_arcsec: float = 0.224  # until measured
    observer: Optional[str] = None
    observatory: str = "Julian, CA"
    telescope: str = "CDK20"
    # Filter registry: name -> FilterConfig (beam, eff_wavelength_nm)
    filters: Tuple[FilterConfig, ...] = field(default_factory=default_efw_filters)
```

Loaded from:

- `night_plans/*.yaml` top-level `camera:` block.
- Optional `instrument_calibration.yaml` (beam separations after first light).

Passed into `_build_header_config()` and serialized to `pol_config.yaml`.

---

## 6. Phase 1 — FITS provenance and PolConfig sidecar

**Objective:** Every FITS frame is self-describing for reduction without a
handwritten log. **Estimated effort:** 2–4 developer days. **Blocks:** F1–F5, F8,
F9, R1.

### 6.1 Task 1.1 — `DetectorCards` in fits_writer

**Files:** `alpyca_tools/fits_writer.py`, `obs_utils/fits_routine.py`

1. Add `@dataclass class DetectorCards` mirroring §5.1.1.
2. Add `detector: Optional[DetectorCards] = None` to `FitsHeaderConfig` /
   `CaptureConfig`.
3. In `build_header()`, write detector cards **after** attempting Alpaca
   `GAIN`/`OFFSET`/`CCD-TEMP`, with explicit precedence:
   - `EGAIN` always from `DetectorCards.egain_e_per_adu`.
   - `GAIN` from `DetectorCards.gain_setting` or camera.
   - `SET-TEMP` from context; `CCD-TEMP` from camera post-exposure.
4. Update `GAIN` comment string to warn it is not e?/ADU.

**Acceptance:**

- Single bias frame contains `GAIN=0`, `EGAIN=1.0`, `READMODE=0`, `SET-TEMP=-10`,
  `CCD-TEMP` within a few °C of setpoint, `XPIXSZ=3.76`.

### 6.2 Task 1.2 — Populate `INSTROT` and achieved `HWPANG`

**Files:** `obs_utils/night_session.py`

1. Add helper `_pwi4_instrot_deg(pwi4) -> Optional[float]`:
   - Read `pwi4.status().mount` for field rotator angle (confirm exact field name
     during implementation — may be `rotator_angle_degs` or similar in PWI4 status
     JSON).
2. In `_build_header_config()`, when building `PolarimetryCards`:
   - Set `instrument_rotator_deg` from PWI4.
   - If `plan.hwp_angle_deg` is set, also store **commanded** angle; after
     `select_hwp_angle()` returns `achieved`, pass achieved value into header
     build (may require restructuring loop to build header after HWP move).
3. Record HWP residual (commanded ? achieved) in block manifest (Phase 3) or log.

**Acceptance:**

- `pol_seq` sequence of 8 frames: all have `HWPANG`; all have `INSTROT` within
  0.05° of each other for a stable rotator.

### 6.3 Task 1.3 — `SessionCaptureContext` + YAML load

**Files:** new `obs_utils/session_context.py`, `obs_utils/night_plan.py`,
`obs_utils/night_session.py`

1. Define `SessionCaptureContext` (§5.3).
2. Extend night plan YAML schema:

```yaml
camera:
  readout_mode: 0
  readout_mode_name: "Mode 0"
  gain: 0
  offset: 30          # example — use bench value
  egain_e_per_adu: 1.0
  ron_e: 3.5
  cooler_setpoint_c: -10.0
```

3. `load_night_plan()` returns `(NightSessionConfig, SessionCaptureContext)`.
4. `run_night_session()`:
   - Apply cooler setpoint at startup if Alpaca supports `SetCCDTemperature`.
   - Apply gain/offset/readout_mode once at session start and before each block
     if brick overrides.
   - Pass context into every `_build_header_config()` call.

**Acceptance:**

- Dry-run load of plan prints camera settings.
- Live capture uses configured gain/offset without per-frame manual INDIGO clicks.

### 6.4 Task 1.4 — PolConfig sidecar

**Files:** new `obs_utils/pol_config.py`, extend `poltools/io.py`

1. `def session_pol_config_to_polconfig(ctx, filter_name) -> PolConfig`
   — maps session context + active filter to `poltools.PolConfig`.
2. `def write_pol_config_sidecar(path, ctx, blocks: list)` — writes
   `FITSDATA/YYYYMMDD/pol_config.yaml`:

```yaml
session: "20260709"
created_utc: "2026-07-10T02:50:00Z"
detector:
  readout_mode: 0
  gain_setting: 0
  offset: 30
  egain_e_per_adu: 1.0
  ron_e: 3.5
  cooler_setpoint_c: -10.0
blocks:
  - id: "HD154445_polV8"
    ut_start: "..."
    filter: "Photometric V"
    pol_config: { ... }   # snapshot of relevant PolConfig fields
```

3. `def load_pol_config_sidecar(path) -> PolConfig` for reduction scripts.
4. Call `write_pol_config_sidecar` at session end (and incrementally per block).

**Acceptance:**

- `poltools.reduce_to_stokes(paths, pt.load_pol_config_sidecar(...))` runs without
  hand-built `PolConfig` on bench data.

### 6.5 Task 1.5 — caltools header reader upgrade

**Files:** `caltools/caltools/io.py`, `caltools/io.py` (keep duplicates in sync)

1. `sensor_config_from_header`: prefer `EGAIN`; warn on fallback.
2. Default `pixel_size_um=3.76` for `INSTRUME` containing `QHY268` if `XPIXSZ`
   missing (with warning), rather than raising — **only** for this sensor.

**Acceptance:**

- `ct.sensor_config_from_header("bias.fits")` returns gain 1.0 without extra args.

### 6.6 Task 1.6 — Keep fits_writer and fits_routine in sync

Add a unit test `tests/test_fits_header_contract.py` that asserts both builders
produce the same keys for equivalent inputs (or consolidate to one builder imported
by both modules — preferred long-term).

---

## 7. Phase 2 — First-light night plan and data layout

**Objective:** `scripts/plan_night.py night_plans/20260709.yaml --run` executes
the §4 timeline. **Estimated effort:** 3–5 days. **Blocks:** N1–N10, N12, N5, N9.

### 7.1 Task 2.1 — `night_plans/20260709.yaml`

Create the per-night plan with structure:

```yaml
uses: palette.yaml
session: "20260709"
observer: "<name>"
base_data_dir: FITSDATA
naming: polite                    # new: see Task 2.4
calibrate_stage: both

camera:
  readout_mode: 0
  gain: 0
  offset: <bench>
  egain_e_per_adu: 1.0
  cooler_setpoint_c: -10.0

catalog:
  # Unpolarized standards (RA hours, Dec deg — from plan §2)
  "gamma Boo":        {ra: 14.53464, dec: 38.30833}
  "HD 154892":        {ra: 17.12817, dec: 15.21056}
  "BD+32 3739":       {ra: 20.20058, dec: 32.79556}
  "HD 212311":        {ra: 22.36628, dec: 56.53194}
  # Polarized standards
  "HD 154445":        {ra: 17.09228, dec: -0.89222}
  "HD 161056":        {ra: 17.72972, dec: -7.07944}
  "Hiltner 960":      {ra: 20.39122, dec: 39.34889}
  "HD 204827":        {ra: 21.48269, dec: 58.74000}
  # Asteroids — ephemeris at acquisition overrides these placeholders
  "Melpomene":        {ra: 19.05, dec: -9.60}
  "Juno":             {ra: 20.38, dec: -3.80}

plan:
  - cal: [{bias25: {}}]                    # 19:35 — before sunset science
  - {target: "TwilightFlat", alt: 45, az: 118, lay: [twilightFlatV]}
  # ... standards in timeline order with exposure overrides ...
  - cal: [{darks30: {}}, {darks_short: {}}]  # 03:40
```

**Timeline mapping (bricks per target):**

| Target | Brick(s) | Overrides|
|--------|----------|-----------|
| ? Boo | `polV8` | `exp: 0.3`, `n: 3`|
| HD 154892 | `polV8`, `polR8` | `exp: 30` (default)|
| HD 154445 | `polV8`, `polR8` | `exp: 3`, `n: 2`; **QA gate after**|
| HD 161056 | `polV8` | `exp: 3`|
| BD+32 3739 | `polV8` | `exp: 30`|
| Melpomene | `polV16` | `exp: 30`|
| Hiltner 960 | `polV16` | `exp: 45`|
| Juno | `polV16` | `exp: 30`|
| HD 204827 | `polV8`, `polR8` | `exp: 30`|
| HD 212311 | `polV8` | `exp: 30`|
Dawn Option B bricks appended only if operator enables `dawn: option_b` flag in plan.

### 7.2 Task 2.2 — Extend `night_plans/palette.yaml`

Add bricks:

```yaml
bias25:
  type: cal
  frame: BIAS
  exp: 0
  n: 25
  readout_mode: 0
  gain: 0

polV8_3s:
  type: pol_seq
  filter: "Photometric V"
  angles: 8
  exp: 3
  n: 2

polV8_03s:
  type: pol_seq
  filter: "Photometric V"
  angles: 8
  exp: 0.3
  n: 3

# ... polV16_45s, polR8_3s, etc.

darks_short:
  type: cal_multi     # NEW brick type
  frames:
    - {frame: DARK, exp: 0.2, n: 10}
    - {frame: DARK, exp: 2, n: 10}
    - {frame: DARK, exp: 5, n: 10}
```

### 7.3 Task 2.3 — New brick types

**File:** `obs_utils/night_plan.py`

#### `cal_multi`

Expands to multiple `FramePlan` calibration stacks with different exposures.

#### `pol_flat` (twilight polarimetric flats)

```yaml
twilightFlatV:
  type: pol_flat
  filter: "Photometric V"
  angles: 4              # or 8 — plan uses 4 CasPol angles
  exp: 2.0               # initial; may override per frame in runner
  n: 10
  imagetyp: FLAT
  track: false
  dither_arcsec: 10
  priority: 1            # V first, then R, B in separate bricks
```

Expansion behavior:

1. `TargetPlan` with `alt_deg`/`az_deg` only (no RA/Dec) ? antisolar pointing.
2. For each HWP angle in `{0, 22.5, 45, 67.5}`:
   - Move HWP, capture `n` frames with `IMAGETYP=FLAT`.
   - Optional dither: small alt/az jitter via PWI4 (if implemented) or tracker off.
3. `POLSEQ` id like `"twilight_flat_V"`.
4. Runner hook `on_flat_exposure_suggest(mean_adu, fwc)` to adjust exposure —
   target 15–35 kADU (30–60% of ~51 ke? FWC at gain 1).

**Stretch:** full auto-rescale is Phase 2b; minimum viable is operator adjusts
`exp` in inline override between bands.

### 7.4 Task 2.4 — Filename and directory convention

**File:** `obs_utils/night_session.py`

Add `NightSessionConfig.naming: Literal["legacy", "polite"] = "legacy"`.

**`polite` mode:**

```
FITSDATA/{YYYYMMDD}/
  2026-07-09_{target_slug}_{filter_slug}_{exposure}s_{index:03d}.fits
```

- Date from session, not subfolder per target.
- `filter_slug`: `V`, `R`, `B`, `Dark`, etc. (short map from EFW names).
- Flat structure (no `targets/` subtree) per observation plan §9.
- Optional `cal/` subdirectory for BIAS/DARK/FLAT if cleaner for reduction.

**`legacy` mode:** preserve current behavior for backward compatibility.

### 7.5 Task 2.6 — QA gate hooks in night runner

**File:** `obs_utils/night_session.py`

```yaml
# In 20260709.yaml
qa_gates:
  - after_target: "HD 154445"
    script: first_light_qa
    args: {band: V, abort_on_fail: false}
```

Runner invokes subprocess or Python callable before proceeding.

---

## 8. Phase 3 — Block manifest and environmental logging

**Objective:** Satisfy §9 logging without a paper log. **Estimated effort:** 2–3 days.

### 8.1 Block manifest schema (JSONL)

**File:** `FITSDATA/YYYYMMDD/block_manifest.jsonl`

One JSON object per line per observing block (one brick expansion under one target):

```json
{
  "block_id": "HD154445_polV8",
  "target": "HD 154445",
  "brick": "polV8",
  "ut_start": "2026-07-10T05:10:00Z",
  "ut_end": "2026-07-10T05:48:00Z",
  "filter": "Photometric V",
  "hwp_angles_requested": [0, 22.5, 45, 67.5, 90, 112.5, 135, 157.5],
  "hwp_homed_at_start": true,
  "hwp_residuals_deg": [0.01, -0.02, ...],
  "instrot_deg": 127.45,
  "set_temp_c": -10.0,
  "ccd_temp_c": -9.8,
  "gain_setting": 0,
  "readout_mode": 0,
  "egain_e_per_adu": 1.0,
  "moon_sep_deg": 78.2,
  "airmass": 1.12,
  "seeing_arcsec": null,
  "transparency": null,
  "frames_expected": 16,
  "frames_written": 16,
  "fits_glob": "2026-07-09_HD154445_V_3.0s_*.fits",
  "notes": ""
}
```

**Asteroid extension:**

```json
{
  "phase_angle_deg": 6.0,
  "scattering_plane_pa_deg": 112.3,
  "horizons_query_utc": "2026-07-10T07:15:00Z",
  "horizons_id": "18"
}
```

### 8.2 Implementation — `obs_utils/block_log.py`

- `BlockLogger` context manager: `with logger.block("HD 154445", "polV8") as blk: ...`
- On enter: record `ut_start`, query PWI4, moon separation via `astropy.coordinates`
  `get_moon` + target RA/Dec.
- On exit: frame count glob, `ut_end`, flush JSON line.
- For asteroids: call `obs_utils.horizons.py` before slew.

### 8.3 `obs_utils/horizons.py`

Thin wrapper:

```python
def fetch_ephemeris(
    target_id: str,
    epoch: Time,
    lon_deg: float,
    lat_deg: float,
    elev_m: float,
) -> EphemerisRow:
    """RA, Dec, phase_angle, scat_plane_pa, daily_motion."""
```

Use `astroquery.jplhorizons.Horizons` with site codes or explicit lat/lon.
Cache results per target per night to JSON alongside manifest.

**Site constants (Julian):** lat 33.0701°N, lon ?116.6451°W, elev 1294 m (from plan).

### 8.4 Focus log (optional `focus_log.jsonl`)

After focus block (timeline 20:40):

```json
{"ut": "...", "focuser_position": 12345, "filter": "Photometric V",
 "plate_scale_arcsec_per_px": null, "notes": "both beams round"}
```

---

## 9. Phase 4 — Inline QA gates and end-of-night audits

**Objective:** Automate plan decision points. **Estimated effort:** 3–4 days.

### 9.1 `scripts/bias_qa.py`

**Input:** Directory or list of bias FITS files (?2 for pair statistics).

**Algorithm (§5.4 Lab 1):**

1. Load frames with `caltools.load_frame` (respect `BZERO=32768`).
2. **Histogram test:** min ADU > 0 (no pinned-at-zero pixels); mean bias level
   in ~100–200 ADU range (bench expectation).
3. **RON:** For each pair `(i, j)`, compute difference image;  
   `RON = std(diff) / sqrt(2) * EGAIN` from header.
4. Report mean RON across pairs; pass if |RON ? 3.5| < 0.5 e? (tunable).

**Output:** JSON report + exit code 0/1. Update `SessionCaptureContext.ron_e` if
passed.

**Hook:** Run automatically after `bias25` cal block when `qa_gates` includes
`bias_qa`.

### 9.2 `scripts/first_light_qa.py`

**Input:** Glob of HD 154445 (or configurable standard) `polV8` frames; reference
catalog entry (P, PA, uncertainties).

**Algorithm (§6.5):**

1. Build `PolConfig` from sidecar or headers.
2. `reduce_to_stokes(paths, cfg, method="lsq")` — requires flats or warn.
3. `reduce_to_stokes(paths, cfg, method="double_ratio")`.
4. Compare measured P, PA to reference (HD 154445: P=3.67%, PA=88.6°).
5. Compare lsq vs double_ratio (q, u) — pass if < 0.1%.
6. Compute `efficiency = P_meas / P_ref` for each method; log both.
7. Pass criteria: |?P| < 0.3%, |?PA| < 2°, efficiency in [0.95, 1.02].

**CLI:**

```zsh
python scripts/first_light_qa.py FITSDATA/20260709/2026-07-09_HD154445_*.fits \
  --ref-name "HD 154445" --abort
```

**Dependencies:** Flat fields for lsq path — if no flats yet, run double_ratio
only and warn that lsq gate is deferred.

### 9.3 `scripts/flat_quality_gate.py`

For each polarized standard available:

1. Reduce with lsq and double_ratio.
2. If max |(q_lsq ? q_dr), (u_lsq ? u_dr)| > 0.001 (0.1%), set session flag
   `flats_valid: false` in manifest and recommend `double_ratio` for remainder.

### 9.4 `scripts/sequence_audit.py`

**Input:** `FITSDATA/YYYYMMDD/` directory.

**Algorithm:**

1. Group all LIGHT frames by (`POLSEQ`, `FILTER`).
2. For each group, read `HWPANG` values; compare to expected set (8 or 16 angles).
3. Report missing angles, duplicate angles, wrong count.
4. Exit 1 if any science sequence incomplete.

**Hook:** Call from shutdown checklist or `run_night_session` epilogue.

### 9.5 Integration with `run_night_session`

```python
@dataclass
class QAGate:
    after_target: Optional[str] = None
    after_cal: Optional[str] = None   # e.g. "bias25"
    handler: str = ""                 # "first_light_qa" | "bias_qa" | ...
    abort_on_fail: bool = False
```

Runner dispatches after matching block completes.

---

## 10. Phase 5 — Characterization pipelines

**Objective:** Close §5.4 verification loops with analysis scripts. **Post-first-light
priority** except bias QA and quick PTC sanity.

### 10.1 Photon transfer curve (morning ladder)

**Data:** Option B frames 05:20–05:45 — paired images at increasing sky brightness.

**Script:** `scripts/ptc_from_pairs.py`

1. Group consecutive frame pairs at similar mean ADU (or use manifest block).
2. For each level: `gain = mean(signal) / var(difference)` (FPN cancels in pair diff).
3. Plot mean ADU vs variance; identify linear region and rollover near FWC.
4. Update `pol_config.yaml` `egain_e_per_adu` with measured value.
5. Optionally rewrite `EGAIN` in headers via `fits` header update tool (or store
   correction only in sidecar — prefer sidecar + re-reduce).

**Pass:** gain within 5% of 1.0 e?/ADU; rollover near 51 ke?.

### 10.2 Plate scale and beam geometry

**Script:** `scripts/calibrate_beam_geometry.py`

1. Plate-solve standard fields (astrometry.net or `obs_utils/platesolve.py`).
2. Identify o/e star pairs; fit separation and PA per filter.
3. Write `instrument_calibration.yaml`:

```yaml
filters:
  "Photometric V":
    beam_sep_px: 58.2
    beam_pa_deg: 12.4
    eff_wavelength_nm: 551
    plate_scale_arcsec: 0.225
```

4. Future capture reads this into `BEAMSEP`/`BEAMPA`/`PIXSCALE` headers.

### 10.3 Count-rate zero point (Lab 3)

**Script:** `scripts/flux_zeropoint.py`

1. For each standard: median ADU in aperture / `EXPTIME` ? counts/s.
2. Fit vs catalog V magnitude per filter ? zero point for exposure calculator.
3. Store in `instrument_calibration.yaml` under `zeropoint.V`, etc.

### 10.4 Asteroid motion check

Already specified in §6.4 — implement in `scripts/asteroid_motion_check.py`:

1. Centroid ordinary beam at first and last frame of `polV16` sequence.
2. Compare drift to Horizons predicted rate (arcsec/h).

---

## 11. Phase 6 — Exposure calculator and long-term polish

### 11.1 Exposure calculator

Address §6.2 CONJECTURED tag. Once zero points exist (§10.3):

```python
def suggest_exposure_s(v_mag: float, band: str, target_adu_fraction: float = 0.45) -> float:
    """Return exposure to hit target fraction of FWC in brighter Savart beam."""
```

Integrate into `plan_night.py` dry-run output as advisory column.

### 11.2 Notebook / script unification

Observation plan references Interactive Control notebook cells. Long-term:

- Notebook calls `run_night_session` / `plan_night` under the hood.
- Or generate notebook from YAML plan for manual override steps only (startup
  checklist, focus).

### 11.3 Archive script migration

`scripts/archive/night_session_20260302.py` contains working patterns for flats,
illumination maps, and filename conventions. Mine for:

- Flat capture loop with exposure adjustment.
- Directory layout under `FITSDATA`.

Do not run archive script as-is; port patterns into brick system.

---

## 12. First-light manual fallbacks

If Phase 1–2 are incomplete by 2026-07-09, operators **must** use these manual
steps. The observation plan remains valid; software gaps shift burden to the log.

### 12.1 FITS provenance fallback

For **every block**, record in a paper/electronic log (template below):

| Field | Example|
|-------|---------|
| UT start / end | 05:10–05:48 Z|
| Target | HD 154445|
| readout_mode | 0 (Mode 0)|
| gain_setting | 0|
| offset | 30|
| EGAIN (assumed) | 1.0 e?/ADU|
| SET-TEMP | ?10 °C|
| CCD-TEMP (from FITS or INDIGO) | ?9.8 °C|
| INSTROT (from PWI4 GUI) | 127.45°|
| HWP homed? | Y|
| Filter | Photometric V|
Build `PolConfig` by hand in reduction notebook from these values.

### 12.2 QA gate fallback (§6.5)

After HD 154445 sequence, in Python:

```python
import poltools as pt
from pathlib import Path
paths = sorted(Path("FITSDATA/20260709").glob("*HD154445*V*.fits"))
cfg = pt.PolConfig(...)  # hand-built
r_lsq = pt.reduce_to_stokes(paths, cfg, method="lsq")
r_dr  = pt.reduce_to_stokes(paths, cfg, method="double_ratio")
# compare to P=3.67%, PA=88.6°
```

Do not proceed to HD 161056 until PA offset is understood if gate fails.

### 12.3 Frame-count audit fallback

Before shutdown, run:

```zsh
python -c "
from pathlib import Path
from astropy.io import fits
from collections import defaultdict
groups = defaultdict(set)
for p in Path('FITSDATA/20260709').glob('*.fits'):
    h = fits.getheader(p)
    if h.get('IMAGETYP') == 'LIGHT' and 'HWPANG' in h:
        groups[(h.get('OBJECT'), h.get('POLSEQ'))].add(float(h['HWPANG']))
for k, angs in sorted(groups.items()):
    print(k, len(angs), 'angles', sorted(angs))
"
```

Expect 8 angles per `polV8`, 16 per `polV16`.

### 12.4 Horizons fallback

At asteroid slew, query
[https://ssd.jpl.nasa.gov/horizons.app](https://ssd.jpl.nasa.gov/horizons.app)
with site coordinates; log phase angle and RA/Dec in block log.

---

## 13. Testing and verification matrix

| Test | Type | Command / action | Pass criterion|
|------|------|------------------|----------------|
| Header contract | Unit | `pytest tests/test_fits_header_contract.py` | All §5.1 required keys present|
| Brick expansion | Unit | `plan_night.py night_plans/20260709.yaml` | Frame count matches plan §4|
| HWP modulation | Hardware | `hwp_modulation_test.py --mode hardware` | 4? modulation in Fourier check|
| Bias QA | Integration | `bias_qa.py` on 25 bias frames | RON ? 3.5 e?, histogram OK|
| Sim polarimetry | Integration | `polarimetry_showcase.py` | Unchanged baseline|
| First light QA | Integration | `first_light_qa.py` on sim or standard data | P/PA within tolerances|
| Sequence audit | Integration | `sequence_audit.py FITSDATA/...` | Exit 0 on complete night|
| Reduction e2e | Integration | `reduce_to_stokes` with sidecar only | No manual PolConfig|
| INSTROT stability | Hardware | 8-frame sequence | max ? min < 0.05°|
| Cooler stability | Hardware | Bias at start vs end | CCD-TEMP within 1 °C of SET-TEMP|
---

## 14. File touch map and dependency graph

### 14.1 Files to create

| File | Phase | Purpose|
|------|-------|---------|
| `obs_utils/session_context.py` | 1 | `SessionCaptureContext`|
| `obs_utils/pol_config.py` | 1 | Sidecar read/write|
| `obs_utils/block_log.py` | 3 | JSONL manifest|
| `obs_utils/horizons.py` | 3 | JPL Horizons wrapper|
| `night_plans/20260709.yaml` | 2 | First-light plan|
| `night_plans/first_light_palette.yaml` | 2 | Optional extended palette|
| `scripts/bias_qa.py` | 4 | Bias histogram + RON|
| `scripts/first_light_qa.py` | 4 | §6.5 gate|
| `scripts/flat_quality_gate.py` | 4 | §5.3 gate|
| `scripts/sequence_audit.py` | 4 | End-of-night audit|
| `scripts/ptc_from_pairs.py` | 5 | Morning PTC|
| `scripts/calibrate_beam_geometry.py` | 5 | BEAMSEP/BEAMPA|
| `scripts/flux_zeropoint.py` | 5 | Exposure calculator input|
| `tests/test_fits_header_contract.py` | 1 | Header parity test|
### 14.2 Files to modify

| File | Phases | Changes|
|------|--------|---------|
| `alpyca_tools/fits_writer.py` | 1 | `DetectorCards`, `INSTROT` population support|
| `obs_utils/fits_routine.py` | 1 | Sync with fits_writer|
| `obs_utils/night_session.py` | 1–4 | Context, naming, INSTROT, QA hooks, block log|
| `obs_utils/night_plan.py` | 2 | `cal_multi`, `pol_flat`, camera YAML|
| `obs_utils/imaging.py` | 1 | Pass context to capture|
| `caltools/caltools/io.py` | 1 | `EGAIN` preference|
| `poltools/io.py` | 1 | `load_pol_config_sidecar` helper|
| `night_plans/palette.yaml` | 2 | New bricks|
| `scripts/plan_night.py` | 2 | Print camera + QA summary on dry-run|
### 14.3 Dependency graph (implementation order)

```
Phase 1: session_context + DetectorCards + INSTROT
           ?
Phase 1: pol_config sidecar + caltools EGAIN
           ?
Phase 2: palette + 20260709.yaml + naming mode
           ?
Phase 2: pol_flat brick (can parallelize after Phase 1)
           ?
Phase 3: block_log + horizons (parallel)
           ?
Phase 4: QA scripts (depend on Phase 1 sidecar + Phase 2 plan)
           ?
Phase 5: characterization (depends on data from first light)
```

---

## 15. Priority schedule relative to 2026-07-09

Assuming implementation starts **2026-07-08** (one day before first light):

### Must ship (P0) — before opening the roof

| Item | Phase | Rationale|
|------|-------|-----------|
| `EGAIN`, `READMODE`, `SET-TEMP`, `GAIN` distinction | 1 | All electron math|
| `INSTROT` on every science frame | 1 | PA zero-point / QA|
| `SessionCaptureContext` + YAML `camera:` block | 1 | Operator-proof settings|
| `pol_config.yaml` sidecar | 1 | Reduction without hand config|
| `night_plans/20260709.yaml` with correct exposures | 2 | Timeline execution|
| `FITSDATA` + `polite` naming (or documented legacy + log) | 2 | Data finds reduction|
| Catalog entries for all targets | 2 | Slew automation|
### Should ship (P1) — strongly reduces operator error

| Item | Phase|
|------|-------|
| `bias_qa.py` + hook after bias block | 4|
| `first_light_qa.py` (at least double_ratio path) | 4|
| `sequence_audit.py` at shutdown | 4|
| `block_manifest.jsonl` (even partial) | 3|
| `bias25` + `darks_short` bricks | 2|
### Can be manual first night (P2)

| Item | Phase|
|------|-------|
| `pol_flat` full auto-rescale | 2b|
| Dawn PTC ladder bricks | 2|
| Horizons automation | 3|
| Flat quality gate automation | 4|
| Beam geometry calibration scripts | 5|
| Exposure calculator | 6|
---

## 16. Open questions and decisions

Record decisions here as they are resolved during implementation.

| # | Question | Options | Recommendation|
|---|----------|---------|----------------|
| D1 | Where does PWI4 expose field rotator angle? | Inspect `pwi4.status().mount` fields | Spike in `observatory_smoke_test.py`; document field name|
| D2 | Pyxis on lab Mac: serial vs Alpaca rotator index | `PYXIS_CONFIG` vs `rotator_index` | Observatory Windows uses Alpaca; lab uses serial — both must set `HWPANG`|
| D3 | Flat directory: flat FITSDATA root vs `cal/` subdir | Flat vs nested | `cal/FLAT/` under `FITSDATA/YYYYMMDD/` for clarity|
| D4 | Rewrite FITS `EGAIN` after PTC vs sidecar only | Header mutation vs sidecar | **Sidecar only** — preserve raw headers; reduction reads sidecar|
| D5 | `polite` naming vs legacy default | Breaking change | Default `legacy`; first-light plan sets `naming: polite`|
| D6 | lsq QA before flats exist | Skip lsq gate | Run **double_ratio only** until V-band flats validated|
| D7 | QHY offset bench value | From 2026-03 campaign | Read from bench log; put in `camera.offset` YAML|
| D8 | Consolidate `fits_routine` and `fits_writer` | Single module | Yes in Phase 1.6 — `fits_writer` canonical|
---

## Appendix A — Reference values (QHY268M first light)

From observation plan §5.1 / §5.4 (2026-03 bench — verify in situ):

| Parameter | Value | FITS / config key|
|-----------|-------|-------------------|
| Readout mode | Mode 0 | `READMODE=0`|
| Gain setting | 0 | `GAIN=0`|
| Conversion gain | ~1.0 e?/ADU | `EGAIN=1.0`|
| Read noise | ~3.5 e? | `RON` or sidecar|
| Full well | ~51 ke? | used in exposure targeting|
| Cooler setpoint | ?10 °C | `SET-TEMP`|
| Pixel pitch | 3.76 µm | `XPIXSZ`, `YPIXSZ`|
| Plate scale (expected) | ~0.225?/px | `PIXSCALE` when measured|
**Do not change gain/mode during the night** — invalidates all masters and error
model (plan §5.4).

---

## Appendix B — Standard-star QA reference (HD 154445)

Primary QA gate target (§6.5):

| Quantity | Reference | Tolerance|
|----------|-----------|-----------|
| P | 3.67% ± 0.05% | ±0.3% measured|
| PA | 88.6° ± 0.7° | ±2° measured|
| lsq vs double_ratio (q,u) | — | < 0.1%|
| Efficiency P_meas/P_ref | 0.97–1.00 (Cole 2010 benchmark) | 0.95–1.02 nightly flag|
Catalog source: `julian_obs_plan` / Sch92 (plan §2).

---

## Appendix C — Related project documents

| Document | Relevance|
|----------|-----------|
| `first_light_obs_plan.pdf` | Operational timeline and requirements source|
| `julian_obs_plan.md` | Standard-star catalog and site constraints|
| `startup_shutdown_checklist.md` | Manual startup/shutdown aligned with §4|
| `docs/lab_trial_checklist.md` | Bench procedures (gitignored; on disk at observatory)|
| `docs/polarimetry/08_polarimetric_flats.md` | Flat protocol and lsq/double_ratio gate|
| `docs/polarimetry/07_cmos_error_model.md` | §1.5 gain-mode provenance requirements|
| `scripts/README.md` | Device topology and brick system overview|
| `poltools/README.md` | Reduction methods and `PolConfig` usage|
| `caltools/README.md` | Notes that QHY does not write reliable `GAIN`|
---

*End of implementation plan.*
