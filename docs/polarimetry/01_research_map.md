# POLITE Polarimetry Pipeline — Research Map (Phase 1)

**Date:** 2026-06-07
**Goal:** Establish the verified, literature-grounded methodology for (a) simulating data
from the POLITE telescope chain and (b) reducing it to Stokes parameters with
publication-grade error metrics.

All methods below are sourced **only** from Source A (user-provided reference materials)
and Source B (peer-reviewed journals), per `CLAUDE.md`. Each method is tagged with its
source. Converted markdown of every source lives in `reference-sheets/_md/`.

---

## 1. What POLITE is (instrument lineage)

Evidence assembled from the reference set establishes that **POLITE is a small-telescope,
dual-beam, single-detector linear imaging polarimeter** in the direct lineage of:

- **Cole (2010)** "Developing a Polarimeter to Support the Epsilon Aurigae Campaign" (SAS) — *Source A* (scanned, no text layer; method reproduced via DUSTPol below)
- **DUSTPol** (Wolfe 2014, U. Denver, advisor R. Stencel) — *Source A* — dual-beam HWP + Savart plate on an 8" SCT.
- **DBIP** (Masiero, Hodapp, Harrington & Lin 2007, PASP 119, 1126) — *Source B* — the canonical small-telescope dual-beam imaging polarimeter design + reduction.
- The `Standard_Stars.md` reference is literally Gary Cole's calibration star list (`/Users/garycole/...`), confirming the lineage.

**Canonical optical chain (the real "telescope chain" to simulate):**

```
Sky → PlaneWave CDK20 OTA → PWI4 Focuser/Rotator (field rotator, angle α)
     → Astronomik L3 UV/IR-cut filter → rotating Half-Wave Plate (HWP, angle θ)
     → ZWO 5-slot EFW (Photometric B, V, R, Clear, Dark/blocking)
     → α-BBO Savart plate (18 mm; polarizing beam splitter, o + e beams; small
       chip on one orientable corner) → QHY268M (IMX571 mono CMOS) detector
```

Each point source appears **twice** on the detector (ordinary `o` + extraordinary `e`
beams, orthogonally polarized, separated by ~few–30 px). The HWP is stepped through a
set of angles; the relative o/e intensities modulate with HWP angle, encoding Q and U.

> **Resolved hardware (confirmed with the user, 2026-06-08; see `03_design_decisions`):**
> α-BBO **Savart** plate (18 mm), **not** a Wollaston. The **PWI4 rotator** is a
> separate field rotator *upstream* of the L3/HWP (distinct from the HWP's own
> rotation stage). Filters: an **Astronomik L3 UV/IR-cut** (fixed, after the
> rotator) plus a **ZWO 5-slot EFW** (B, V, R, Clear, Dark). The filter wheel sits
> **before** the α-BBO analyzer. Because α-BBO is **dispersive**, the o/e beam
> separation is **per filter** — measured from flats/standards (Source A), not from
> a dispersion model. An ideal filter commutes through the Mueller chain, so the
> filter order does not change the *math*, but it does change simulator geometry
> (per-band beam separation, carried in the session `pol_config.yaml` sidecar) and
> the FITS keywords.

### Telescope-chain numbers (for the simulator)
| Quantity | Value | Source|
|---|---|---|
| Aperture D | 0.508 m (CDK20) | README / PlaneWave CDK20 spec|
| Focal length | 3454 mm (f/6.8) | PlaneWave CDK20 spec|
| Detector | QHY268M, IMX571 mono, 3.76 µm px | MEMORY.md, README|
| Plate scale | 206265·3.76e-6/3.454 ≈ **0.224″/px** | derived|
| Gain (mode0,g0) | ~1.0 e⁻/ADU | MEMORY.md (PTC-measured)|
| Read noise | ~3.5 e⁻ | MEMORY.md|
| Full well | ~51 000 e⁻ | MEMORY.md|
| BZERO | 32768 (→ `memmap=False`) | MEMORY.md, caltools.io|
---

## 2. The measurement equation (dual-beam HWP modulation)

For an ideal half-wave plate at angle θ followed by an analyzer, the two emergent beams
carry intensities (DBIP eq.; Masiero 2007 — *Source B*; identical in DUSTPol §2.1 — *Source A*):

$$ I'_{\parallel}(\theta) = \tfrac12\big[I + Q\cos 4\theta + U\sin 4\theta\big], \qquad
   I'_{\perp}(\theta) = \tfrac12\big[I - Q\cos 4\theta - U\sin 4\theta\big]. $$

The factor **4θ** (not 2θ) is the HWP signature. A full Mueller-matrix treatment for a
*real* retarder of retardance φ (Masiero 2007 — *Source B*) shows the depolarization /
V-leak terms scale as (1−cos φ); for φ within 176.4°–183.6° this is ≤0.2% — the basis of
the systematic error budget and a verification check for the simulator.

### 2a. Reduction method A — double-difference ratio (point sources) — **bad-flat fallback**
DBIP (Masiero 2007, *Source B*); DUSTPol "Pickering's method" (Wolfe 2014, *Source A*).

> **Default change (2026-06-09):** Method B (LSQ, §2b) is now the **project
> default** (`reduce_to_stokes(..., method="lsq")`), run on flat-fielded frames
> (PRNU does not cancel in the fit — `07_cmos_error_model.md` §1.7). Method A
> remains the quick,
> easily-specifiable fallback (`method="double_ratio"`) for nights whose
> skyflats are unusable: per-beam throughput cancels exactly, no flat needed.

Define the per-angle normalized difference (the "polarization ratio") from the o/e fluxes
`F_o, F_e` measured by aperture photometry:

$$ R(\theta) = \frac{F_{e}(\theta) - F_{o}(\theta)}{F_{e}(\theta) + F_{o}(\theta)}. $$

Using the four HWP angles {0°, 45°, 22.5°, 67.5°}:

$$ q = \tfrac12\big[R(0^\circ) - R(45^\circ)\big] = Q/I, \qquad
   u = \tfrac12\big[R(22.5^\circ) - R(67.5^\circ)\big] = U/I. $$

**Why this method:** taking the o/e ratio at each angle cancels transparency/seeing/
extinction (T(t)); taking the *difference* of complementary angles cancels the flat-field /
transmission gain ratio between the two beams. Result: **flat-field-independent,
seeing-independent** normalized Stokes — ideal for the QHY268M dual-beam point-source case.

### 2b. Reduction method B — least-squares modulation fit (many sources / N angles)
SOLVEPOL (Ramírez et al. 2017, PASP 129, 055001 — *Source B*), formulation of Magalhães,
Benedetti & Roland (1984).

For N HWP positions ψ_i = (i−1)·22.5°, with modulation
`z_i = (F_{e,i}−F_{o,i})/(F_{e,i}+F_{o,i})` (corrected by the total-flux ratio FT_e/FT_o):

$$ z_i = Q\cos 4\psi_i + U\sin 4\psi_i, \quad
   Q = \tfrac{2}{N}\sum_i z_i\cos 4\psi_i,\quad U = \tfrac{2}{N}\sum_i z_i\sin 4\psi_i. $$

This generalizes to any N≥4 and yields a residual-based error (next section). We will
implement **both** A and B; A is the default for point sources, B is used when N>4 angles
are taken and for cross-checking.

---

## 3. Error metrics (publication-grade)

### 3.1 Per-measurement propagation (photon + detector noise)
σ on each flux from Poisson(source+sky) + read noise + dark, via the QHY268M noise model
already characterized in `caltools` (gain, RON, dark). Propagate through the q,u formulae
analytically (and cross-check with Monte-Carlo). σ_Q, σ_U follow from standard error
propagation of the ratio/difference expressions.

### 3.2 Residual-based error (SOLVEPOL / Magalhães 1984 — *Source B*)
For the LSQ method, assuming σ_Q=σ_U=σ_P:

$$ \sigma_P = \sqrt{\tfrac{2}{N-2}\Big(\sum_i z_i^2 - Q^2 - U^2\Big)}. $$

### 3.3 Polarization-angle error
High-SNR (Serkowski; used in SOLVEPOL — *Source B*):

$$ \sigma_\theta\,[\deg] = 28.65\,\frac{\sigma_P}{P}. $$

Low/moderate SNR: use the proper confidence intervals of **Naghizadeh-Khouei & Clarke
(1993)** (referenced by SOLVEPOL) / Montier II (*Source B*).

### 3.4 Polarization bias (the critical publication issue)
P = √(Q²+U²) is **positive-definite ⇒ biased** at low SNR (Rice distribution).
We adopt the **Modified Asymptotic (MAS) estimator** — Plaszczynski et al. (2014, MNRAS
439, 4048 — *Source B*), eq. 20; restated in Montier et al. (2015 II — *Source B*) eq. 19:

$$ \hat p_{\mathrm{MAS}} = \hat p - b^2\,\frac{1 - e^{-\hat p^2/b^2}}{2\hat p}, $$

with noise-bias `b²` (Montier II eq. 15; canonical equal-variance case `b²=σ²`), and the
MAS variance (Montier II eq. 20):

$$ \sigma^2_{\hat p,\mathrm{MAS}} = \frac{\sigma_Q'^2\cos^2(2\psi-\theta) + \sigma_U'^2\sin^2(2\psi-\theta)}{I_0^2}. $$

Classical alternatives to report/compare (all *Source B*): Wardle & Kronberg (1974)
`p_db=√(p²−σ²)`; Simmons & Stewart (1985) estimator family; Ricean treatment per
Clarke & Stewart (1986) (SOLVEPOL flags P/σ_P ≲ 3 as bias-affected).

**Decision:** MAS is the modern default (continuous, unbiased for SNR≳2, analytic
variance); we will also expose the naive and Wardle–Kronberg debiasing for comparison
tables, as Montier II recommends reporting estimator choice explicitly.

---

## 4. Instrumental polarization & calibration (Mueller chain)

Required for absolute, publishable Stokes (DBIP §3, DUSTPol §3 — *Sources B/A*):

1. **Instrumental polarization (IP)** `q₀,u₀`: from **unpolarized standard stars**
   (`Standard_Stars.md` list — *Source A*; Serkowski 1974, Gehrels ed.). Subtract in q–u
   space: `q_corr = q − q₀`, `u_corr = u − u₀`.
2. **Position-angle zero-point** Δθ: from **polarized standard stars** with known θ_lit
   (e.g. DBIP found 9.23°±0.32°). Rotate q,u by 2Δθ to the equatorial (N→E) frame.
3. **Telescope/mirror IP & rotator dependence**: characterize vs instrument-rotator angle
   (DBIP Figs. 3–4 — *Source B*); separates sky-fixed (real) from instrument-fixed (IP)
   signal. Relevant because POLITE has a PWI4 instrument rotator.
4. **Polarization efficiency** (depolarization factor) from highly-polarized standards;
   divide measured p by efficiency.

These define the **Mueller matrix chain** the simulator must inject and the pipeline must
invert: `M_total = M_telescope · M_rotator(α) · M_HWP(θ,φ) · M_analyzer`.

---

## 5. Simulator requirements (forward model)

To "generate simulated data from the current telescope chain," the simulator must, per the
forward model above, produce **FITS frames byte-compatible with the real pipeline**:

1. Input astrophysical Stokes vector (I, Q, U[, V]) per source (+ optional polarized sky).
2. Apply Mueller chain → I'_o(θ), I'_e(θ) for each HWP angle θ in the chosen set.
3. Place o & e PSFs on the detector grid (plate scale 0.224″/px, beam separation, field
   rotation), realistic seeing PSF.
4. Inject detector physics from `caltools`/QHY268M: photon shot noise, read noise (3.5 e⁻),
   dark, bias pedestal/BZERO=32768, gain (e⁻/ADU), full-well clipping (51 ke⁻), PRNU/flat.
5. Write FITS with proper headers incl. a **HWP-angle keyword** + IMAGETYP/EXPTIME/etc.
   matching `obs_utils/fits_routine.py` conventions.

**Verification target (Phase 6):** inject known (Q,U,p,θ); confirm the pipeline recovers
them within the quoted error bars (pull distribution ~N(0,1)); confirm flat-field
independence of method A; confirm MAS removes low-SNR bias; reproduce the (1−cos φ)
depolarization scaling for a non-ideal HWP.

---

## 6. Source inventory (converted to markdown)

**Methods (reduction):** `solvepol.md` (B), `DUSTPol_Commissioning_v1.md` (A),
`Hawaii_88inch_polarimeter_commissioning.md` = DBIP/Masiero 2007 (B),
`Reno_polarimeter_commissioning.md` (A).
**Error metrics / statistics:** `arxiv/Plaszczynski2014_MAS_estimator.md` (B),
`arxiv/Montier2015_II_best_estimators.md` (B), `arxiv/Montier2015_I_instrumental_systematics.md` (B).
**Calibration:** `Standard_Stars.md` (A).
**Science context:** `wang_and_wheeler_2008_-properties_of_polarization_in_Supernovae.md` (B),
`James_Wiley_PhD_Dissertation.md` (A), lecture/lab notes (A).
**Detector (alt. architecture, DoFP micro-polarizer arrays — NOT the POLITE chain):**
`imx-sensors-polarimetry-characterization.md` (B) — kept for reference only.
**Failed to convert (scanned, no text layer):** `Small_Telescope_Spectropolarimetry.pdf`,
`Developing_a_Polarimeter...Epsilon_Aurigae.pdf` (Cole 2010), `kemp_wolstencroft_1972.pdf`
— their methods are covered by DBIP/DUSTPol/SOLVEPOL above.

---

## 7. Method-provenance ledger (CLAUDE.md Sources A/B compliance)

| Pipeline element | Method | Source | Tag|
|---|---|---|---|
| Measurement eqn (HWP 4θ) | I'∥,⊥ | Masiero 2007; DUSTPol | B / A|
| Stokes q,u (point src) | double-difference | Masiero 2007; Wolfe 2014 | B / A|
| Stokes Q,U (N angles) | LSQ modulation | Magalhães 1984 / SOLVEPOL 2017 | B|
| σ_P residual | modulation residual | Magalhães 1984 / SOLVEPOL | B|
| σ_θ | 28.65·σ_P/P; NK&C 1993 | SOLVEPOL; Montier II | B|
| p debias | MAS estimator | Plaszczynski 2014; Montier II 2015 | B|
| IP / PA-zero / efficiency | standard-star calib | Masiero 2007; DUSTPol; Serkowski 1974 | B / A|
| Detector noise model | PTC/RON/dark | caltools (in-repo, PTC-measured) | in-repo|
No method outside Sources A/B is used. If any is later needed, it will be flagged per
`CLAUDE.md`.
