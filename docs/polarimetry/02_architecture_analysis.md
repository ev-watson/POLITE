# POLITE Polarimetry Pipeline — Architecture & Integration Analysis (Phase 2)

**Date:** 2026-06-07
**Goal:** Map how a simulator + Stokes-extraction pipeline fits into the existing POLITE
codebase, identify reusable components, and surface gaps that must be filled.

---

## 1. Existing architecture (what's already here)

| Package | Role | Reuse for polarimetry|
|---|---|---|
| `obs_utils/` | Observatory control (PWI4 mount, rotator, imaging, night sessions, FITS write) | **Acquisition** path; FITS header conventions; PWI4 *rotator* control hook|
| `alpyca_tools/` | ASCOM Alpaca camera + FITS writer | FITS write conventions for simulated frames|
| `caltools/` (v0.1.0) | **Detector characterization** library | **Heavy reuse** — see §2|
| `reduction.ipynb` / `workspace.ipynb` | Intensity image reduction (bias/dark/flat) | Reduction idiom to match (origin='upper', BZERO handling)|
| `utils.py` | Legacy helpers | Avoid; superseded by caltools|
### caltools data model (the contract to extend)
- `SensorConfig` (frozen dataclass): `nx, ny, pixel_size_um, gain_e_per_adu, temperature_c, bitdepth, sensor_name`; `.with_gain()`, `.central_roi()`. — `caltools/_types.py`
- `AnalysisResult` (dataclass): `name, scalar_summary:dict, maps:dict, metadata:dict`. **Every analysis returns this.** Polarimetry results should too, for consistency.
- I/O (`caltools/io.py`): `load_frame` (uses `memmap=False` for BZERO=32768 — **mandatory** for these sensors), `load_cube`, `load_cube_chunked`, `sensor_config_from_header`, `group_by_type_and_exposure`, `get_timestamps`, `get_file_index`.
- Stacking (`caltools/stacking.py`): `master_bias/dark/flat(paths, method="median")`.
- Noise model (`caltools/gain.py`, `noise.py`, `dark.py`): PTC gain, RON map, dark current — **the exact detector physics the simulator must inject** (gain ~1 e⁻/ADU, RON ~3.5 e⁻, FWC ~51 ke⁻ for QHY268M).
- Plotting (`caltools/plotting.py`): `image_with_colorbar`, `ptc_plot`, `histogram_gaussian_overlay`, `noise_map_with_histogram`, `summary_table`.

### FITS conventions (must match for simulated frames)
From `obs_utils/fits_routine.py` and real headers: `EXPTIME, EXPOSURE, DATE-OBS, IMAGETYP
("Light Frame"), OBJECT, INSTRUME, XPIXSZ, GAIN, OFFSET, CCD-TEMP, BZERO=32768, BSCALE=1,
FILTER, XBINNING/YBINNING`, WCS (`CRPIX/CRVAL/CD`). **No polarimetry keyword exists yet.**

> WARNING: **Detector reality check:** on-disk science frames (`datafiles/scis/*Jupiter*`) are
> still **SBIG STX-16803** (4096², 9 µm CCD), not QHY268M. The QHY268M migration is in
> progress. ⇒ The simulator and `SensorConfig` must remain **detector-parameterized**;
> default to the QHY268M/IMX571 target chain but allow the legacy CCD.

---

## 2. Gaps (what does NOT exist and must be built)

1. **No polarimetry code at all** (confirmed Phase 0: zero `stokes/polariz/wollaston/HWP`
   hits). Q/U/p/θ extraction, modulation fit, debiasing — all new.
2. **No source detection / aperture photometry.** `caltools` is *pixel/detector* level, not
   *source* level. Dual-beam reduction needs: detect point sources, pair o/e beams, do
   concentric-aperture photometry with sky annulus (SOLVEPOL/DBIP style). → needs
   `photutils` (verify present & non-deprecated per `CLAUDE.md` package rule) or a small
   in-house aperture routine.
3. **No HWP-angle metadata.** FITS writer has no retarder keyword; `io.group_by_*` regex
   (`^(\d{8})(Dark|FlatField)([\d.]+)secs`) only knows Dark/FlatField. Need a polarimetry
   grouping (by HWP angle / sequence) and a header keyword (proposed `HWPANG` / `RET-ANG`).
4. **No forward model / simulator.** Mueller chain, dual-beam PSF placement, detector-noise
   injection, FITS emission.
5. **No Mueller / IP calibration module.** IP subtraction, PA zero-point, efficiency.
6. **No polarization error/statistics module** (MAS debiasing, σ_θ, covariance).

---

## 3. Proposed structure — new sibling package `poltools/`

Rationale: keep **science reduction** (`poltools`) separate from **detector
characterization** (`caltools`), mirroring caltools' clean, dataclass-based, `AnalysisResult`
style and **reusing** caltools.io/stacking/noise. Sibling package (not a caltools submodule)
keeps responsibilities crisp and import graph one-directional (`poltools` → `caltools`).

```
poltools/
  __init__.py        # public API, __version__
  _types.py          # PolConfig (instrument geometry), StokesResult, SourcePair
  mueller.py         # Mueller matrices: M_HWP(θ,φ), M_analyzer, M_rotator(α), M_telescope
  simulate.py        # forward model: Stokes scene → o/e PSFs → detector noise → FITS frames
  photometry.py      # source detection, o/e pairing, concentric-aperture photometry + sky
  modulation.py      # method A (double-difference) + method B (LSQ fit) → q,u,Q,U
  stokes.py          # assemble I,Q,U[,V]; p, θ; covariance
  errors.py          # σ propagation, residual σ_P, σ_θ, MAS debiasing (Plaszczynski/Montier)
  calibration.py     # IP subtraction, PA zero-point, polarization efficiency (standards)
  io.py              # polarimetry FITS keywords, sequence grouping (reuses caltools.io)
  plotting.py        # modulation curves, q-u plane, polarization vector maps
  pipeline.py        # end-to-end orchestrator (raw frames → calibrated StokesResult)
docs/polarimetry/    # this folder (research map, architecture, design Qs, plan, verify)
tests/               # unit + injection-recovery tests
```

### Data-flow (simulation → science)
```
PolConfig + Stokes scene
   │  poltools.simulate
   ▼
simulated raw FITS  ──►  caltools master bias/dark/flat (calibration)
   │  poltools.photometry (detect, pair o/e, aperture phot)
   ▼
per-angle o/e fluxes (+σ)  ──►  poltools.modulation (A/B) ──► q,u (+σ_q,σ_u)
   │  poltools.calibration (IP, PA-zero, efficiency)
   ▼
calibrated Q,U  ──►  poltools.stokes + poltools.errors
   ▼
StokesResult{I,Q,U,p,θ, σ's, p_MAS, covariance, χ²} ──► plotting / catalog
```

### Type extensions
- `PolConfig` (frozen): wraps `SensorConfig` + instrument geometry — α-BBO Savart
  beam separation (px) & PA **per filter** (`filters` registry + `for_filter`),
  HWP angle set, retardance δ, plate scale, **PWI4 field-rotator** angle, active
  filter. Drives both simulator and reducer (single source of truth).
- `StokesResult` (AnalysisResult-compatible): `scalar_summary={I,Q,U,p,theta,p_mas,
  sigma_p,sigma_theta,...}`, `maps={...}`, `metadata={method, n_angles, chi2, source A/B
  provenance}`.

---

## 4. Integration touch-points (non-simulation, for later real-data use)
- **Acquisition:** `obs_utils/night_session.py` + PWI4 rotator (`pwi4_client.rotator_*`)
  could step the HWP if the HWP is on the PWI4 rotator (TBD — design Q). Out of scope for
  the simulation deliverable but the FITS keyword/grouping must be acquisition-compatible.
- **FITS writer:** add `HWPANG` (and analyzer/rotator angles) to `fits_routine.py`
  `FitsHeaderConfig` so real frames carry the same keyword the simulator writes.

---

## 5. External package check (CLAUDE.md "verify package supports feature, not deprecated")
- `photutils` — for `DAOStarFinder`/`aperture_photometry` (source detect + aperture phot).
  **To verify** installed in env `POLITE` and current API (non-deprecated) before use.
- `astropy` — present (used throughout); `astropy.modeling`/`convolution` for PSF.
- `numpy/scipy` — present; scipy for LSQ + special functions (Rice/erf for MAS).
- `numba` — present (caltools uses it); optional for simulator speed.
No non-A/B *analysis method* is introduced; these are implementation libraries only.

---

## 6. Summary
A new `poltools/` sibling package, reusing `caltools` I/O + stacking + noise model and the
`AnalysisResult` contract, cleanly hosts both the **telescope-chain simulator** and the
**Stokes extraction pipeline**. The principal build effort is: forward model (Mueller +
dual-beam + detector noise), **aperture photometry on o/e pairs** (the main missing
capability), modulation→Stokes math (methods A & B), error/debiasing (MAS), and
standard-star calibration. All math is fixed by the Phase 1 research map (Sources A/B).
Open instrument-geometry choices are deferred to Phase 3 design questions.
