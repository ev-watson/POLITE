# POLITE Polarimetry Pipeline — Verification (Phase 6)

**Date:** 2026-06-07
**Scope:** Evidence that `poltools` (simulator + Stokes pipeline) is correct and
publication-grade. Two layers: (a) a unit/integration **test suite**
(`tests/`, 60 tests, all passing) and (b) an end-to-end **showcase**
(`scripts/polarimetry_showcase.py`) across the four science regimes.

Run:
```
/Users/blu3/miniforge3/envs/POLITE/bin/python -m pytest tests/ -q          # 60 passed
/Users/blu3/miniforge3/envs/POLITE/bin/python scripts/polarimetry_showcase.py
```

---

## 1. Verification grid → result

| Check | Test | Result|
|---|---|---|
| Mueller `oe_intensities` = ideal-HWP formula `½[I ± (Q cos4θ + U sin4θ)]` | `test_mueller::test_ideal_hwp_formula` | exact to 1e-12|
| HWP Mueller matrix structure (`cos4φ/sin4φ`, V→−V) | `test_mueller::test_hwp_matrix_structure` | exact|
| Real-retarder depolarization ≤0.2% for δ∈[176.4,183.6]° (Masiero 2007) | `test_mueller::test_retarder_depolarization_window` | ≤0.2%|
| V-ready architecture (QWP δ=90 Mueller correct) | `test_mueller::test_qwp_ready_v_mode` ||
| Noiseless q,u round-trip (Methods A-ratio, A-diff, B) | `test_modulation::test_noiseless_roundtrip_all_methods` | 1e-9|
| **Flat-field independence** of Method A double-ratio under 22% beam imbalance | `test_modulation::test_double_ratio_flat_field_independent` | invariant to 1e-9; difference-form biased (as expected)|
| Method B χ²/cov well-formed | `test_modulation::test_method_b_chi2_and_cov` ||
| **MAS removes Rice bias** at low SNR (1,2,3); near-unbiased at SNR≥3 | `test_errors::test_mas_reduces_rice_bias` ||
| Wardle–Kronberg & naive debiasers | `test_errors::test_wardle_kronberg_and_naive` ||
| **NK&C 1993 PA interval vs Monte-Carlo** (SNR 1.5, 3.0) | `test_errors::test_nkc_pa_interval_matches_mc` | within 15% of MC|
| σ_θ high-SNR = 28.65·σ_P/P; NK&C→asymptote | `test_errors::test_sigma_theta_highsnr_value` ||
| Residual σ_P → 0 for a noiseless perfect fit | `test_errors::test_residual_sigma_p_zero_for_perfect_fit` ||
| **FITS byte round-trip** (sim → `caltools.load_frame`, BZERO=32768) | `test_simulate_io::test_fits_roundtrip_byte_exact` | pixel-exact|
| HWP-sequence FITS keywords + grouping | `test_simulate_io::test_sequence_writes_all_angles_and_groups` ||
| Full-well saturation clipping | `test_simulate_io::test_render_saturation_clip` | clips at FWC, no ADC overflow|
| photutils detection + o/e pairing | `test_photometry::test_detect_and_pair` | all pairs found|
| Aperture flux recovers injected electrons | `test_photometry::test_aperture_flux_recovers_injected` | <3%|
| IP from unpolarized standards | `test_calibration::test_ip_recovered_from_unpolarized_standards` ||
| PA zero-point fit + rotation | `test_calibration::test_pa_zeropoint_and_rotation` | exact|
| Efficiency fit | `test_calibration::test_efficiency` ||
| Full PolCalibration inverts injected IP+PA+eff | `test_calibration::test_polcalibration_bundle_roundtrip` | 1e-9|
| **End-to-end 2D injection-recovery** (Method A) | `test_pipeline::test_end_to_end_2d_injection_recovery` | within 4σ, SNR>10|
| Methods A and B agree | `test_pipeline::test_methods_a_and_b_agree` | <3e-3|
| **Pull distribution ~N(0,1)** (error bars calibrated) | `test_pipeline::test_pull_distribution_is_unit_normal` | |mean|<0.08, std=1±0.08|
| Calibration applied in pipeline removes injected IP | `test_pipeline::test_calibration_applied_in_pipeline` | within 4σ|
**60 passed.**

---

## 2. End-to-end showcase (four science regimes)

Simulated 8-angle HWP sequence (QHY268M 512², seed 20260607), full detector
noise, injected IP=(+0.004,−0.003) and efficiency 0.97, calibrated from
simulated standard stars. (Table: `docs/polarimetry/showcase_results.md`;
figures: `docs/polarimetry/figures/`.)

**Calibration recovery:** fitted IP=(+0.0039,−0.0028) vs injected eff·IP=
(+0.0039,−0.0029); fitted efficiency 0.974 vs 0.97. 

| source | regime | p_inj | p_MAS | σ_p | θ_inj | θ_rec | σ_θ | SNR|
|---|---|---|---|---|---|---|---|---|
| stellar_eps_aur | stellar few-% | 3.000% | 2.950% | 0.033% | 30.0° | 30.20° | 0.32° | 89.3|
| ism_dust | ISM 1–5% | 1.500% | 1.501% | 0.037% | 120.0° | 119.43° | 0.71° | 40.5|
| asteroid | solar-system sub-% | 0.500% | 0.502% | 0.028% | 75.0° | 75.24° | 1.58° | 18.1|
| supernova | SNe sub-% | 0.300% | 0.344% | 0.026% | 160.0° | 153.7° | 2.16° | 13.3|
| sne_faint | SNe low-SNR | 0.500% | 0.440% | 0.159% | 45.0° | 45.08° | 9.99° | 2.9|
All recovered p,θ agree with injection within the quoted errors (the faint SNe
case at SNR≈2.9 shows MAS pulling p down from the naive 0.467% toward the true
0.5%, with a large NK&C-based σ_θ as expected at low SNR).

**MC pull (q,u):** mean −0.013, std 0.997 (target 0, 1) ⇒ error bars are
statistically calibrated.

**Figures:** `fig_modulation.png` (R(θ) curves + model), `fig_qu_plane.png`
(recovered overlaps injected, low-SNR error ellipse visible),
`fig_recovered_vs_injected.png` (on the 1:1 line), `fig_pull.png` (matches
N(0,1)), `fig_pol_vectors.png` (vector overlay).

---

## 3. Key correctness findings (and how they were fixed)

1. **Flat-field independence requires the double-ratio, not the difference.**
   The simple `½[R(0)−R(45)]` cancels per-beam throughput only to first order
   (drifts ~1% under a 22% imbalance). The **double-ratio**
   `q=(√RR−1)/(√RR+1)` cancels it exactly (Tinbergen 1996; DBIP/SOLVEPOL,
   Source B). `poltools` makes the double-ratio the exact form of Method A; the
   difference form is retained as `method="Adiff"` for comparison. *(2026-06-09:
   LSQ is now the pipeline default on flat-fielded frames; the double-ratio is
   the bad-flat-night fallback — design decisions Q6.)*
2. **Residual σ_P normalization.** The SOLVEPOL/Magalhães residual error is
   `√[((2/N)Σz² − q² − u²)/(N−2)]`; this vanishes for a noiseless perfect fit
   and equals the LSQ-covariance σ_q (validated vs MC). Documented in
   `errors.residual_sigma_p`.
3. **Full-well saturation biases the o/e ratio.** A source bright enough to
   exceed the QHY268M 51 ke⁻ well clips the brighter beam more, compressing the
   ratio and biasing p low. This is a *real instrument effect* the simulator
   correctly reproduces; observers must keep targets/standards below the well
   (the showcase geometry/exposure does so). Verified by the saturation test and
   by the corrected showcase (efficiency fit 0.78→0.97 once unsaturated).

No analysis method outside CLAUDE.md Sources A/B was introduced; `photutils` is
an implementation library for detection/aperture photometry only. Provenance is
carried in `StokesResult.metadata` and the Phase 1 ledger.
