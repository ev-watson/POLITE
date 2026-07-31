# Simulating the POLITE Telescope Chain and Extracting Stokes Parameters with Publication-Grade Error Metrics

**Project:** POLITE — dual-beam (HWP + Savart) imaging polarimeter
**Software:** `poltools` v0.1.0 (sibling of `caltools`)
**Date:** 2026-06-07
**Status:** implemented, verified (85/85 tests; end-to-end showcase across four science regimes). CMOS-hardened on 2026-06-08 — see `07_cmos_error_model.md`.

> Provenance discipline (CLAUDE.md): every analysis method below is tagged to
> **Source A** (user-provided references) or **Source B** (peer-reviewed
> journals). The provenance ledger is in `01_research_map.md §7`; `photutils` is
> an implementation library (detection/aperture photometry), not a new method.

---

## 1. Summary

We built and validated an end-to-end capability to (i) **simulate** realistic 2-D
imaging FITS frames from the POLITE telescope chain and (ii) **reduce** them to
calibrated Stokes parameters `(I, Q, U)` with a complete, publication-grade error
budget (per-measurement propagation, residual error, Rice-bias debiasing, and
position-angle confidence intervals). The simulator's frames are **byte-compatible**
with the real acquisition path (BZERO=32768, read back pixel-exact by
`caltools.load_frame`), so the entire real reduction chain — source detection,
o/e pairing, aperture photometry, modulation, calibration, error analysis — is
exercised, not just the math.

The pipeline recovers injected polarization across all four target regimes
(stellar few-%, ISM 1–5%, solar-system sub-%, supernova sub-%) within the quoted
uncertainties, with statistically calibrated error bars (Monte-Carlo pull
mean −0.01, std 1.00), and recovers injected instrumental polarization and
modulation efficiency from simulated standard stars.

The architecture is **full-Stokes-ready** (4-vector Stokes, general-retarder
Mueller matrices); the current pipeline solves the **linear** `(I, Q, U)` problem
because the instrument has only a half-wave plate. A future quarter-wave plate
enables Stokes V with no structural rework. **A returned V of zero is not a
measurement** — a single HWP cannot constrain V — and the code/docs flag this.

---

## 2. The telescope chain and its forward model (simulator)

**Optical chain simulated** (research map §1; hardware locked 2026-06-08):
`Sky → CDK20 (D=0.508 m, f/6.8) → PWI4 Focuser/Rotator (field rotator α) →
Astronomik L3 UV/IR-cut → rotating HWP → ZWO 5-slot EFW (B/V/R/Clear/Dark) →
α-BBO Savart plate (18 mm, o/e split) → QHY268M (IMX571, 3.76 µm, 0.224″/px)`.
The EFW sits **before** the α-BBO analyzer; an ideal filter commutes through the
Mueller chain, so it does not alter q,u. The α-BBO split is **dispersive**, so
the o/e beam separation is **per filter**, measured from flats/standards
(Source A — not a dispersion model) and carried in the per-session
`pol_config.yaml` sidecar, not in the frame header (see §"FITS header
provenance" below).

The forward model (`poltools.simulate`) implements, per HWP angle θ:

1. **Mueller propagation** (`poltools.mueller`, Source B): incident Stokes
   `(I,Q,U,V)` → `M_analyzer · M_HWP(θ,δ) · M_rotator(α)`, optionally with an
   injected instrumental-polarization term and modulation efficiency. In the
   ideal-HWP limit this reproduces exactly (Masiero 2007, Source B; DUSTPol,
   Source A):
   `I'_e = ½[I + Q cos4θ + U sin4θ]`, `I'_o = ½[I − Q cos4θ − U sin4θ]`.
   A non-ideal retardance δ introduces the `(1 − cos δ)` depolarization (≤0.2%
   for δ∈[176.4°,183.6°]; Masiero 2007).
2. **Dual-beam imaging:** the two flux-conserving beams are placed on the
   detector grid as PSFs (Gaussian core, seeing→px via the plate scale),
   separated by the **per-filter** α-BBO Savart beam separation and position
   angle (`cfg.for_filter(band).beam`; the split is dispersive).
3. **QHY268M detector physics** (from `caltools`/MEMORY, in-repo PTC-measured):
   photon shot noise, dark current, read noise (3.5 e⁻), **full-well clipping
   (51 ke⁻)**, gain (≈1 e⁻/ADU), bias pedestal, 16-bit quantization, and the
   BZERO=32768 unsigned-int convention.
4. **FITS emission** with a polarimetry header block (`HWPANG` [commanded],
   `WPUNCERT`, `INSTROT` [PWI4 rotator], `POLBEAM`, `POLSEQ/POLSEQN`, plus the
   standard `alpyca_tools/fits_writer.py` cards). These keywords are **mirrored
   on the real acquisition path** by `PolarimetryCards` in
   `alpyca_tools/fits_writer.py`, so acquired frames carry identical metadata.
   Instrument specs and measured or derived quantities are deliberately excluded
   — see §"FITS header provenance".

A correctness consequence verified here: **bright targets that exceed the full
well bias the o/e ratio low** (the brighter beam clips more). This is a real
instrument effect the simulator reproduces; observations must keep targets and
standards below the well.

### FITS header provenance (2026-07-29)

A header card records **as-acquired state**: something true of *this frame*,
known at write time, and not obtainable more reliably elsewhere. A card is not
written when it is a spec constant, a quantity that must be measured to be
trusted, or a reduction result. The same reasoning already excludes conversion
gain and read noise (per-night characterization values the analyst supplies) and
already limits `CRVAL1`/`CRVAL2` to a labelled pointing **seed** with no
`CTYPE`/`CD` matrix.

Applying it consistently retired six cards the June design added: `BEAMSEP`,
`BEAMPA`, `SAVMAT`, `SAVTHK`, `RETARD`, `POLEFF` and `WAVELEN`.

The `BEAMSEP` case is the clearest. Two defensible numbers exist and the card
holds one, unlabelled:

| route | value | |
|---|---|---|
| spec ÷ pixel pitch | 0.9 mm ÷ 3.76 µm = **239.4 px** | manufacturer nominal |
| measured, 2026-07-09 | **238.4 px** = 0.896 mm | 0.4 % low — ordinary tolerance |

Neither is wrong; the card is. The α-BBO plate is dispersive, so the true
separation is per band and a single scalar is the wrong *shape* for the quantity
regardless of its value. And a better home already exists: the per-session
`pol_config.yaml` sidecar carries `beam_separation_px`,
`beam_position_angle_deg` **and `beam_geometry_characterized`** — the
measured-vs-nominal flag a FITS card cannot express. Dropping the card removes a
lossy duplicate rather than losing information. `nominal_beam_separation_px` and
`validate_beam_separation` stay: they guard a *configured* value against being
unphysical (the historical 60 px placeholder was ~75 % low), which is a config
check, not a header.

Removing the card only fixed half of it. That 60 px placeholder was also the *loud*
part: at ~75 % low, automatic o/e pairing failed outright, so the missing
measurement announced itself. The only defensible default is the manufacturer spec,
and at 239.4 px — about 1 px from the 238.4 px measured on 2026-07-09 — pairing
succeeds and returns plausible q/u instead. So the config side is now closed too
(`03_design_decisions.md` **Q8**): `nominal_beam_separation_px()` is the only default
the package ships, `polconfig_from_fits_headers` has no geometry argument at all,
a measurement enters only through `PolConfig.with_beam_geometry()`, pre-2026-07-29
sidecars are scrubbed on load, and `reduce_to_stokes(detect=True)` is **fail-closed**
on an uncharacterized band (`allow_uncharacterized_geometry=True` marks a reduction
as diagnostic). Night QA passes that flag — a QA WARN never blocks capture — and
carries the "nominal, not measured" label with the result.

Kept, as genuine per-frame state: `HWPANG` (the **commanded** angle),
`WPUNCERT`, `INSTROT`, `POLBEAM` (data layout — which beams this array holds),
`POLSEQ`/`POLSEQN`. `WPUNCERT` passes the test where other constant instrument
properties fail it, because it is the uncertainty on `HWPANG` **in the same
frame**: the open-loop Optec Pyxis Gen3 has no encoder, moves in discrete steps,
and does not land exactly on a round commanded angle, so a reduction weighting or
fitting the modulation curve needs it alongside the angle it qualifies.

Before adding a keyword, ask whether a reader would have to measure it to trust
it. If yes it belongs in the sidecar with a `*_characterized` flag, or in the
instrument description — the sidecar can say "nominal, unverified"; a FITS card
cannot. Frames already on disk keep the retired cards; no reduction path reads
them.

---

## 3. Reduction to Stokes parameters

`poltools.reduce_to_stokes` runs: group by **filter** (per-band α-BBO geometry)
→ group by HWP angle → detect (photutils
`DAOStarFinder`) and pair o/e by the beam offset → concentric-aperture
photometry with a sky annulus (photutils; CCD-equation uncertainties) →
modulation → calibration → Stokes assembly.

**Modulation → normalized Stokes** (research map §2):

- **Method B — least-squares modulation fit (N≥4; default since 2026-06-09).**
  `z_i = q cos4ψ_i + u sin4ψ_i` solved by linear least squares (SOLVEPOL /
  Magalhães 1984 — Source B), giving χ², covariance, and the residual σ_P.
  Requires flat-fielded frames (PRNU does not cancel in the fit —
  `07_cmos_error_model.md` §1.7).
- **Method A — double-ratio (flat-field-independent; bad-flat fallback).** With
  the per-angle beam ratio `r(θ)=F_e/F_o`, `RR_q=r(0°)/r(45°)`,
  `q=(√RR_q−1)/(√RR_q+1)`, and `u` from `r(22.5°)/r(67.5°)`. The per-beam
  throughput cancels **exactly** (Tinbergen 1996; DBIP/Masiero 2007; SOLVEPOL —
  Source B), verified invariant under a 22% beam imbalance — select
  `method="double_ratio"` when the night's skyflats are unusable.
- (A first-order double-**difference** comparator is retained as `method="Adiff"`.)

**Stokes assembly** (`poltools.stokes`): `p=√(q²+u²)`, `θ=½atan2(u,q)∈[0,180)`,
with the full error budget below.

---

## 4. Error metrics (everything required to publish)

All from Source B (research map §3):

1. **Per-measurement propagation.** Aperture fluxes carry CCD-equation σ (source
   shot + sky + read noise + sky-estimation). These propagate analytically
   through the double-ratio (`σ_q = s·σ_{lnRR}/(s+1)²`) and through the LSQ
   covariance.
2. **Residual σ_P** (SOLVEPOL/Magalhães): `√[((2/N)Σz² − q² − u²)/(N−2)]` — the
   modulation-scatter error (→0 for a perfect fit; equals the covariance σ_q).
3. **Rice-bias debiasing.** `p=√(q²+u²)` is positive-definite and biased at low
   SNR. The reported default is the **Modified Asymptotic (MAS) estimator**
   (Plaszczynski 2014 eq. 20; Montier II 2015 eq. 19), with the MAS variance
   (Montier II eq. 20). The **naive** and **Wardle–Kronberg** estimators are also
   exposed for comparison, as Montier II recommends stating the estimator choice.
   MAS is verified to remove the Rice bias for SNR≳2 (Monte-Carlo).
4. **Position-angle error.** High-SNR `σ_θ = 28.65·σ_P/P`; at low/moderate SNR the
   **Naghizadeh-Khouei & Clarke (1993)** confidence interval, computed from the
   stable `erfcx` form and validated against Monte-Carlo coverage.
5. **Pull / coverage.** A Monte-Carlo harness confirms `(x̂−x_true)/σ ~ N(0,1)`
   — the error bars are statistically calibrated.

---

## 5. Calibration (Mueller-chain inversion)

`poltools.calibration` (research map §4; Masiero 2007 §3 / DUSTPol §3 /
Serkowski 1974 — Sources B/A):

- **Instrumental polarization** `(q0,u0)` from unpolarized standards; subtracted
  in q–u space.
- **Position-angle zero-point** Δθ from polarized standards (rotate q,u by 2Δθ to
  the equatorial frame).
- **Polarization efficiency** from highly-polarized standards (divide p by eff).

Bundled as `PolCalibration` (apply order IP → efficiency → PA). In the showcase,
the injected IP and efficiency are recovered from simulated standards
(IP fitted (+0.0039,−0.0028) vs injected eff·IP (+0.0039,−0.0029); efficiency
0.974 vs 0.97).

---

## 6. Results (end-to-end showcase)

8-angle HWP sequence, QHY268M 512², injected IP and efficiency, calibrated from
standards. Full table: `showcase_results.md`; figures: `figures/`.

| source | regime | p_inj | p_MAS | σ_p | θ_inj | θ_rec | σ_θ | SNR|
|---|---|---|---|---|---|---|---|---|
| stellar (ε Aur-like) | stellar | 3.000% | 2.950% | 0.033% | 30.0° | 30.2° | 0.32° | 89|
| ISM dust | ISM 1–5% | 1.500% | 1.501% | 0.037% | 120.0° | 119.4° | 0.71° | 41|
| asteroid | solar-system | 0.500% | 0.502% | 0.028% | 75.0° | 75.2° | 1.58° | 18|
| supernova | SNe sub-% | 0.300% | 0.344% | 0.026% | 160.0° | 153.7° | 2.16° | 13|
| faint SNe | SNe low-SNR | 0.500% | 0.440% | 0.159% | 45.0° | 45.1° | 9.99° | 2.9|
All recovered values agree with injection within the quoted errors; the
low-SNR case demonstrates MAS debiasing and the NK&C σ_θ. MC pull: mean −0.01,
std 1.00.

**Figures:** modulation curves; q–u plane (recovered overlaps injected, low-SNR
error ellipse); recovered-vs-injected p on the 1:1 line; pull histogram = N(0,1);
polarization-vector overlay.

---

## 7. Limitations & next steps

- **Linear only (by hardware):** no Stokes V until a QWP is added; architecture
  is V-ready (general-retarder Mueller, 4-vector Stokes).
- **Keep below full well:** saturation biases the ratio; size exposures so
  targets *and* standards stay under ~51 ke⁻. Photometry now **flags** saturated
  apertures (`BeamFlux.saturated`; pipeline warns) rather than reducing silently.
- **CMOS noise model:** the photometric σ now applies the repeat-frame √N
  suppression (Stockmans et al. eq. 16) and optionally a per-pixel
  read-noise map and hot/bad-pixel mask (`07_cmos_error_model.md`).
- **Per-filter α-BBO dispersion:** the o/e separation is now **per band**
  (`PolConfig.filters` + `for_filter`; reduction groups by filter first). The
  per-band separation and position angle must be **measured from flats/standards
  (Source A)** — the registry ships the **manufacturer nominal** (0.9 mm over
  3.76 µm pixels, ≈ 239.4 px; no arbitrary placeholder) in every slot, flagged
  `characterized=False`. A measurement enters only through
  `PolConfig.with_beam_geometry(sep, pa)`, which validates it against the spec and
  promotes just the active band, or from the session sidecar;
  `polconfig_from_fits_headers` has **no** geometry argument at all, and they are
  **not** FITS cards. Because the nominal sits ~1 px from the measured 238.4 px,
  `detect=True` pairing on unmeasured geometry would succeed and return plausible
  q/u, so `reduce_to_stokes` is **fail-closed**: it raises unless
  `allow_uncharacterized_geometry=True` marks the reduction as diagnostic.
- **Savart chipped corner:** a small chip on one (orientable) corner of the
  α-BBO plate is **not** simulated; it is to be characterized from real flats and
  excluded at reduction time via `reduce_to_stokes(..., exclude_regions=[...])`.
- **PWI4 field rotator:** `INSTROT` is now read **per frame**; the pipeline warns
  if it drifts across a sequence (the PA zero-point assumes constant α per
  sequence). A varying derotator angle across a sequence is not yet modelled.
- **Sky polarization, PSF variation, cosmic rays** are not yet modelled in the
  simulator (straightforward extensions); a 2-D master-dark with FPN/hot
  structure is now injectable.
- **Real-data hookup:** the polarimetry keywords are now mirrored on the
  acquisition path (`PolarimetryCards` in `obs_utils/fits_routine.py` and
  `alpyca_tools/fits_writer.py`); a HWP-stepping + EFW-sequencing step in the
  night-session runner remains to be added (the FITS keyword/grouping are already
  acquisition-compatible).
- **Wider validation:** run the injection-recovery grid over more p∈{0.1…5}% ×
  SNR points and publish the coverage/calibration curves.

---

## 8. Reproduce

```
# tests (85 passed)
/Users/blu3/miniforge3/envs/POLITE/bin/python -m pytest tests/ -q
# end-to-end showcase (writes FITSDATA/SIM_20260607/, figures, results table)
/Users/blu3/miniforge3/envs/POLITE/bin/python scripts/polarimetry_showcase.py
```

**Code:** `poltools/` (modules: `_types, mueller, simulate, io, photometry,
modulation, errors, stokes, calibration, plotting, pipeline`).
**Docs:** `docs/polarimetry/01_research_map.md` … `06_report.md` (this file);
CMOS error model in `07_cmos_error_model.md`.
**Dependencies added:** `photutils` 3.0.0 (AAS/Astropy-affiliated), `pytest`.
