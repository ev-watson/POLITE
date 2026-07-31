# CMOS error model — QHY268M / IMX571 noise handling in `poltools`

**Date:** 2026-06-08 (consolidated into docs 2026-06-09)
**Status:** implemented and verified (`tests/test_cmos_error_model.py`; suite green).
**Scope:** how the photometric error model and the simulator account for the
measured behaviour of the QHY268M / IMX571 CMOS detector, and the literature
basis for the repeat-frame (√N) statistics.

> Provenance discipline (CLAUDE.md): every method below is tagged **Source A**
> (user-provided materials) or **Source B** (peer-reviewed literature).

---

## 1. Detector-driven elements of the error model

### 1.1 Per-pixel read-noise map (RTN tail)

The project characterization (`reference-sheets/_md/reduction.md` §3.3–3.8,
Source A/B) measures a 2-D read-noise map — median 7.286 ADU, RMS 7.590 ADU,
MAD-σ 0.699 ADU — a tight Gaussian core with a long high-σ tail (902,975 px
>3σ = 3.4%; random-telegraph "kite" scatter, Alarcón et al. 2023, Source B). A
scalar read noise describes the core but understates RTN/hot pixels inside an
aperture and cannot represent o-vs-e spatial variation; a differential o/e RON
misestimate maps directly into a wrong σ_q/σ_u.

**Implementation:** `measure_fluxes(..., ron_map=)` accepts a per-pixel
read-noise map (electrons). The read term becomes `Σ_ap RON_i²` over aperture
pixels and `⟨RON²⟩` over the annulus — the standard CCD equation (Howell;
Mortara & Fowler 1981, Source B) with the RON the detector actually has. A
constant map reproduces the scalar result exactly (regression-tested); the
scalar `cfg.read_noise_e` remains the default. Threaded through
`photometer_sequence` and `reduce_to_stokes`.

### 1.2 Bad/hot-pixel repair

With 3.4% of pixels hot at >3σ, the chance that ≥1 hot/RTN pixel lands in a
~150-px aperture is `1 − 0.966¹⁵⁰ ≈ 99%`. The per-angle median-combine rejects
these only for N≥3; at **N=1** there is no rejection, and a hot pixel in the o-
but not e-aperture is a fake polarization signal.

**Implementation:** `measure_fluxes(..., bad_pixel_mask=)` repairs flagged
pixels with `astropy.convolution.interpolate_replace_nans` (PSF-weighted local
estimate) before the aperture sum and sky median. Tested: a 60 kADU hot pixel
that inflated an N=1 flux by >20% is brought back within ~3% of the clean
value. The mask source is the `reduction.md` hot-pixel census (Source A/B).
Recommendation retained: prefer ≥3 exposures per HWP angle so the median also
engages. (Single-frame CR rejection à la L.A.Cosmic is outside Sources A/B as
cited — not adopted.)

### 1.3 Effective-N variance suppression after frame combining

Combining N equal-exposure frames at one HWP angle suppresses the per-pixel
(hence aperture-summed) variance — by `N` for a mean, `N/(π/2)` for a median
(Kenney & Keeping 1962, large-N; exact for N≤2). Before this was threaded
through, the pipeline reported the single-exposure σ after a median combine:
σ over-reported by ≈√N, SNR under-reported, MAS over-debiased, MC pulls <1.

**Implementation:** `measure_fluxes(..., n_combined=, combine=)` divides the
variance by the effective N (`_effective_n`); `reduce_to_stokes` records the
per-angle frame count and passes it automatically. Validated by Monte Carlo:
at N=5 median, the reported σ matches the empirical flux scatter to <15%.
Literature basis in §2.

### 1.4 Dark current

The CCD equation needs **no explicit `n_ap·D·t` term**: the background is
measured empirically from the annulus, whose pixels accumulate dark too, so
`bg_e` already carries the dark shot noise — an explicit term would
double-count it. (At 0.005 e⁻/s, dark is ~0.075 e⁻ over 15 s anyway.)

The simulator-fidelity side: `render_frame(..., dark_frame=)` accepts an
optional 2-D master-dark (electrons, with its FPN / hot-pixel structure) in
place of the scalar `D·t`, matching the `reduction.md` dark handling.

### 1.5 Gain-mode provenance

`read_noise_e` and `gain_e_per_adu` are valid for **one readout mode / gain
setting only** (QHY268M RON/gain change strongly across HCG/LCG), and the
POLITE writer records the camera gain setting while gain is measured via PTC (`reduction.md` §3.1,
Source A/B). `PolConfig` therefore carries `readout_mode`, `gain_setting`, and
`linearity_limit_e` provenance; `with_hwp_angles` uses `dataclasses.replace`
so it can never silently drop them.

### 1.6 Saturation / linearity guard

A saturated brighter beam truncates more flux than its partner → spurious
polarization. `aperture_peaks` + `BeamFlux.saturated` flag apertures whose
peak reaches `(peak − bias)·gain ≥ sat_limit_e()` (linearity limit or full
well); `reduce_to_stokes` warns and records `metadata["saturated"]` /
`["saturated_angles"]` — flag-and-surface rather than silently reduce.

### 1.7 PRNU and flat-fielding across reduction methods

Pixel-to-pixel response (PRNU) interacts differently with each reduction:

- **`double_ratio`** — rigorously flat-field-independent: per-beam throughput
  appears as a common factor in every per-angle ratio and cancels exactly in
  the ratio-of-ratios (Tinbergen 1996; DBIP/Masiero 2007; Wolfe 2014 —
  Sources A/B).
- **`double_difference`** — cancels it to first order only.
- **`lsq_modulation`** (the **project default** since 2026-06-09, design
  decisions Q6) — assumes matched / pre-normalized o/e beams and does **not**
  cancel PRNU. On real frames, **flat-field first** (master skyflat per EFW
  band; the `reduction.md` PRNU map). If the night's flats are unusable, fall
  back to `double_ratio`.

---

## 2. Repeat-frame statistics (the √N question)

The photometric error is the analytic single-exposure CCD equation. Do modern
polarimetry pipelines apply the textbook √N when repeat frames are combined?
Two error paradigms appear in our sources:

| Pipeline (source) | Per-measurement error model | Explicit √N? | Detector match to POLITE|
|---|---|---|---|
| **`poltools`** | Analytic CCD equation (shot+sky+RON) | **Yes — effective-N applied** | —|
| Stockmans et al. (B; `imx-sensors-polarimetry-characterization.md`) | Analytic photon noise, eq. 16 `σ_q≈√g/√(2S)` | **Yes — explicit**: "suppressed by √N for N exposures to average over" (lines 408–411) | **Same IMX/CMOS family**|
| DBIP / Masiero 2007 (B; `Hawaii_88inch_polarimeter_commissioning.md`) | Analytic Poisson noise per measurement | Yes — implicit (0.03% → 0.015% over 4 sets, lines 188–190) | CCD, dual-beam HWP (closest architecture)|
| SOLVEPOL / Magalhães et al. 1984 (B; `solvepol.md`) | **Residual** σ_P from the modulation fit (eq. 6–7) | No — N enters via dof + median combine | CCD imaging polarimeter (closest pipeline)|
| Reno / Cole 2010 (B; `Reno_polarimeter_commissioning.md`) | **Empirical** set variance (~5 repeats) | Yes — empirically (s²/N) | CCD, Savart dual-beam|
| DUSTPol / Wolfe 2014 (A; `DUSTPol_Commissioning_v1.md`) | Qualitative photon-count | — | small-telescope imaging polarimeter|
**Reading:** pipelines with an **analytic** σ (like ours) pair it with √N
(Stockmans eq. 16 — same IMX sensor family, so it transfers directly; DBIP at
the set level). Those that omit it switched paradigm — fit-residual (SOLVEPOL)
or empirical (Reno) — which already absorbs the combination.

**Decision:** thread the effective N into the CCD equation (the only option
that also serves the 4-angle `double_ratio`, where the residual route has
unstable 2-dof residuals; no change to the observing pattern). The SOLVEPOL
residual σ_P is additionally surfaced for `lsq` (`sigma_p_resid` in
`scalar_summary`) as the literature-faithful cross-check. Residual-only and
empirical-set-variance headlines were considered and not adopted.

---

## 3. Verified invariants (unchanged by the above)

- BZERO/unsigned handling: `io.read_pol_frame` → `caltools.load_frame` with
  `memmap=False`; uint16 + BZERO=32768, pixel-exact.
- Median-combine of repeats matches SOLVEPOL (B) and the project CMOS RTN /
  salt-and-pepper rationale (A/B).
- Error statistics: MAS (Plaszczyński 2014 / Montier II), Wardle–Kronberg,
  NK&C low-SNR PA interval, Serkowski high-SNR σ_θ — all Source B.
- Fail-closed validation: non-positive/non-finite fluxes, missing HWPANG,
  names/positions mismatch, unknown method all raise.
- Plot origin upper-left throughout.

**Tests:** `tests/test_cmos_error_model.py` (effective-N scaling + MC scatter
match, RON-map scalar regression + RTN patch, hot-pixel repair, dark-FPN
injection, provenance fields, saturation flag end-to-end).
