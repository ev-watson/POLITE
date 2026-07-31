# POLITE Polarimetry Pipeline — Design Decisions (Phase 3)

**Date:** 2026-06-07
**Goal:** Lock the design choices that shape the build, from user answers + derived defaults.

---

## 1. User-answered design questions

| # | Question | Decision|
|---|---|---|
| Q1 | Polarization scope | **Full-Stokes target, linear (I,Q,U) now.** Telescope currently has only a HWP, so simulate/extract **linear** polarization now, but architect the Mueller chain and Stokes vectors as full 4-vectors `(I,Q,U,V)` so a quarter-wave-plate Stokes-V mode drops in later **without rework**.|
| Q2 | Analyzer / beam-splitter | **HWP modulation + α-BBO Savart birefringent plate (18 mm)**, dual-beam (ordinary `o` + extraordinary `e`). Reduction is flat-field-independent double-ratio (Method A primary). Beam separation & position-angle are `PolConfig` parameters and are **per filter** — see the locked hardware below (Q5).|
| Q3 | Science regimes driving the error budget | **All four:** Stellar/Epsilon Aurigae (sub-%–few %), ISM/dust (~1–5%), Solar-system asteroids/comets/planets (~0.1–1%), Supernovae (sub-%). ⇒ error budget must be valid from **sub-% (low-SNR, MAS-debiasing critical)** to **few-% (high-SNR)**. Verification cases span this whole range.|
| Q4 | Simulator fidelity | **Full 2D imaging FITS frames.** Render o/e PSFs on the detector grid with the full QHY268M noise model (shot, dark, read, PRNU, full-well), bias pedestal + BZERO=32768, and a HWP-angle keyword — **byte-compatible with the real pipeline**.|
| Q5 | **Optical chain locked (2026-06-08)** | Real light path: `Sky → CDK20 → PWI4 Focuser/Rotator (field rotator α) → Astronomik L3 UV/IR-cut → rotating HWP (θ) → ZWO 5-slot EFW (B, V, R, Clear, Dark) → α-BBO Savart 18 mm → QHY268M`. **PWI4 is a separate field rotator** upstream of the L3/HWP (not the HWP stage). The α-BBO Savart is **dispersive ⇒ per-filter beam separation**, sourced from **measured flats/standards (Source A)**, not a Sellmeier model. The EFW sits **before** the analyzer; an ideal filter commutes, so the math is unchanged but per-band `BEAMSEP`/`BEAMPA` and the new `SAVMAT`/`SAVTHK`/`WAVELEN` keywords are added. The Savart has a **small chip on one (orientable) corner** (~1–2 mm edge, ≤1 mm deep) — handled at reduction time by an optional `exclude_regions` mask once characterized from flats; **not** simulated.|
| Q6 | **Default reduction method (2026-06-09)** | **LSQ modulation fit (Method B) is the pipeline default** — `reduce_to_stokes(..., method="lsq")` — run on **flat-fielded** frames (PRNU does not cancel in the fit — `07_cmos_error_model.md` §1.7), since it uses all N angles and yields the (q,u) covariance, χ² goodness-of-fit, and residual σ_P. **Double-ratio (Method A) is the quick, easily-specifiable fallback** (`method="double_ratio"`) **when the night's skyflats are bad/unusable**: flat-field-independent by construction (needs HWP {0, 22.5, 45, 67.5}°). Supersedes the "Method A primary" wording in Q2 and the implication bullet below.|
| Q7 | **FITS header provenance (2026-07-29)** | **A header card records as-acquired state only** — something true of *this frame*, known at write time, and not better obtained elsewhere. Cards that are spec constants, quantities requiring measurement to be trusted, or reduction results are **not written**. This retires `BEAMSEP`/`BEAMPA`, `SAVMAT`/`SAVTHK`, `RETARD`, `POLEFF` and `WAVELEN` from the keyword set added in Q5 and listed in §2 below. Trigger: `BEAMSEP` was being filled from the manufacturer-nominal 0.9 mm (239.4 px on 3.76 µm pixels) while the 2026-07-09 measurement gave 238.4 px (0.896 mm) — an ordinary 0.4 % manufacturing tolerance, but the card gave a reader no way to tell which number they held, and a dispersive plate has no single scalar separation anyway. Beam geometry now lives **only** in the per-session `pol_config.yaml` sidecar, which carries `beam_geometry_characterized` and so can say "nominal, unverified"; a FITS card cannot. Kept: `HWPANG` (commanded), `WPUNCERT`, `INSTROT`, `POLBEAM`, `POLSEQ`/`POLSEQN`. `WPUNCERT` survives the same test that removes the others because it is the uncertainty on `HWPANG` **in the same frame** — the open-loop Optec Pyxis Gen3 steps discretely and never lands exactly on a round angle, and a reduction weighting or fitting the modulation curve needs that alongside the angle. Frames already on disk keep the retired cards; no reduction path reads them.|
| Q8 | **No unmeasured beam geometry can be reported (2026-07-29)** | Q7 removed the cards; this closes the config side, because **the placeholder was the loud part**. The retired 60 px default was ~75 % low, so automatic o/e pairing failed outright — the error announced itself. The only defensible default is the manufacturer spec, and at 239.4 px (vs the 238.4 px measured on 2026-07-09) pairing **succeeds and returns plausible q/u**, which is precisely how unmeasured geometry gets published as a result. So: (1) `nominal_beam_separation_px()` — 0.9 mm ÷ pixel pitch — is the **only** default `poltools` ships; every configured separation is seeded from it and validated against it by `validate_beam_separation`. (2) `polconfig_from_fits_headers` has **no geometry argument at all**; it returns the nominal flagged `characterized=False`. (3) A measurement enters only via `PolConfig.with_beam_geometry(sep, pa)`, which validates against the spec (so a stale placeholder cannot be laundered into a "measurement") and promotes **only the active band**. (4) Sidecars are scrubbed on load — `FITSDATA/` is read-only and five pre-2026-07-29 sidecars still hold 60 px, so an implausible *uncharacterized* value falls back to nominal with a warning, while one declared `characterized: true` is **kept** and warned about (silently overriding a recorded measurement would hide the disagreement instead of surfacing it). (5) `reduce_to_stokes(detect=True)` is **fail-closed** on an uncharacterized band, with `allow_uncharacterized_geometry=True` as the explicit diagnostic opt-out — the same idiom as `allow_mixed_sequences`. Night QA (`obs_utils/qa_lib.py`) passes that flag, because a QA WARN must never block capture, and carries the "nominal, not measured" label in its warnings so the number never travels without it.|

### Implications of the answers
- **V-ready, not V-now.** `mueller.py` builds general retarder Mueller matrices `M_ret(δ, φ)` (δ=retardance, φ=fast-axis angle); HWP is `δ=180°`, future QWP is `δ=90°`. Stokes vectors carry 4 components throughout; the linear pipeline simply leaves V unconstrained/zero and does not solve for it (a single HWP cannot recover V). This is flagged in code + docs so no one mistakes a zero V for a measured V.
- **Savart dual-beam ⇒ Method A is primary**, Method B (LSQ) is the cross-check and the N>4 path. Both implemented (per research map).
- **Four science regimes ⇒** the injection-recovery verification grid covers p ∈ {0.1%, 0.5%, 1%, 3%, 5%} across SNR from ~3 to ~1000, and the **MAS debiasing** path is exercised and shown to remove low-SNR bias (the sub-% SNe / asteroid cases).
- **2D FITS ⇒** simulator emits real `.fits` files through the authoritative POLITE writer, then reads them back through `caltools.io` (`memmap=False`), so the *entire* real reduction chain (detection → pairing → photometry → modulation → Stokes → errors → calibration) is exercised, not just the math.

---

## 2. Decisions made without asking (derived from CLAUDE.md / research map / repo)

| Topic | Decision | Rationale|
|---|---|---|
| **Source detection + aperture photometry** | Use **`photutils`** (Astropy-affiliated): `DAOStarFinder` for detection, `CircularAperture`/`CircularAnnulus`/`aperture_photometry`/`ApertureStats` for concentric-aperture photometry with sky annulus. Install into env `POLITE`; pin + verify non-deprecated API before use. A thin in-house centroid+aperture path is kept as a no-dependency fallback. | CLAUDE.md: "prefer conventions from astronomical/astrophysical journals and their associated software (Astropy, AAS)." photutils is the AAS/Astropy-affiliated standard and is what SOLVEPOL-class pipelines use. CLAUDE.md also requires verifying the package supports the feature and is not deprecated → done at install time.|
| **Package layout** | New sibling package **`poltools/`** (Phase 2 plan), import graph `poltools → caltools` only. | Keeps science reduction separate from detector characterization; reuses `caltools.io/stacking/noise` + `AnalysisResult` contract.|
| **HWP angle set** | Configurable; **default 4 angles {0°, 22.5°, 45°, 67.5°}** for Method A; supports N≥4 (e.g. 8 or 16 at 22.5° steps) for Method B. | Research map §2a/2b; minimal set spans q,u; extensible.|
| **HWP FITS keyword** | `HWPANG` (deg, the **commanded** HWP angle θ); plus `WPUNCERT` (its rotator step-quantization uncertainty), `POLBEAM` ('o'/'e' or 'dual'), `INSTROT` (**PWI4 field-rotator** α), and `POLSEQ`/`POLSEQN` (sequence id + index). Written by the simulator and **mirrored on the real path** by `PolarimetryCards` in `alpyca_tools/fits_writer.py`. **Superseded by Q7 (2026-07-29):** this row previously also listed `RETARD`, `BEAMSEP`/`BEAMPA` and `SAVMAT`/`SAVTHK`/`WAVELEN`; those are no longer written — beam geometry is in the `pol_config.yaml` sidecar and the rest are spec constants or reduction results. | No retarder keyword exists; needs to be acquisition-compatible (Phase 2 §4). `HWPANG` is the common community choice.|
| **Debiasing default** | **MAS estimator** (Plaszczynski 2014 / Montier II) is the reported default; naive `p` and Wardle–Kronberg exposed for comparison tables. | Research map §3.4; Montier II recommends stating estimator choice.|
| **σ_θ** | High-SNR `28.65·σ_P/P`; switch to **Naghizadeh-Khouei & Clarke (1993)** confidence intervals at low SNR. | Research map §3.3.|
| **PSF model** | Gaussian core default (FWHM from a configurable seeing in arcsec → px via plate scale 0.224″/px); Moffat optional. | Standard; seeing is a `PolConfig`/sim parameter.|
| **Detector defaults** | QHY268M/IMX571: gain ~1.0 e⁻/ADU, RON ~3.5 e⁻, FWC ~51 ke⁻, 3.76 µm px, BZERO=32768, but **fully parameterized** via `SensorConfig` (legacy STX-16803 still selectable). | Repo reality check (Phase 2 §1): on-disk data is still legacy CCD; simulator must be detector-parameterized.|
| **Random seed** | All stochastic sim/MC paths take an explicit `numpy.random.Generator`/seed for reproducible verification. | Publication reproducibility.|
---

## 3. Source-A/B compliance note
No new *analysis method* is introduced by these decisions. `photutils` is an
*implementation library* for detection/aperture photometry (a standard operation described
in DBIP/SOLVEPOL — Source B); it introduces no new science method. All reduction/error math
remains exactly as tagged in the Phase 1 research map ledger (Sources A/B). Any deviation
will be flagged per CLAUDE.md.

---

## 4. What Phase 4 will plan
Concrete module-by-module build order, public API signatures, the verification grid
(injection-recovery + pull/coverage + flat-field-independence + (1−cosφ) depolarization +
MAS bias removal), and the end-to-end showcase that produces the publication-grade figures
and tables fulfilling the original report request.
