# Polarimetric flat fields — requirements, difficulty, and the LSQ/double-ratio decision

**Date:** 2026-06-09
**Status:** analysis / observing-prep guidance (no code change). Companion to
design decision Q6 (`03_design_decisions.md`): **`lsq` default requires
flat-fielded frames; `double_ratio` is the bad-flat fallback.**

> Provenance: **Source A** = user-provided materials (Wolfe 2014 DUSTPol),
> **Source B** = peer-reviewed literature. Two Source-B papers were added to
> the ledger here: Sosa et al. 2019 (CasPol, *J. Astron. Telesc. Instrum.
> Syst.* 5, 028002 — Savart + HWP dual-beam, the closest published analogue to
> POLITE) and González-Gaitán et al. 2020 (FORS2 imaging polarimetry, *A&A*
> 634, A70; arXiv:1912.08684).

## 1. Why this matters

Wolfe 2014 (Source A, §2.1) motivates DUSTPol's double-ratio reduction
precisely by the flat problem: with the gain factor G(α) cancelling, *"flat
field images are no longer needed … as polarimetric flat fields are
time-consuming and difficult to reduce."* POLITE's default since 2026-06-09 is
instead the LSQ modulation fit, which does **not** cancel pixel response —
so the night plan must actually deliver usable polarimetric flats, or the
default falls back to `double_ratio`.

## 2. How a polarimetric flat differs from a regular flat

A regular flat (twilight sky or dome/panel "white frame") needs only
**spatially uniform illumination**; any polarization of the source is
irrelevant because nothing downstream analyzes it. A polarimetric flat has
three extra requirements:

1. **Full optical train in the beam.** The flat must be taken through L3 →
   HWP → EFW filter → α-BBO Savart in the science configuration, because
   vignetting, dust shadows, and the o/e-dependent throughput of those
   elements are part of the response being calibrated. FORS2 (Source B):
   *"Since the beam is split after optical components like … the HWP, flats
   need to be taken with these elements in the light path."* Consequence:
   the detector-only PRNU map from the characterization campaign (white
   frames, no polarimetry optics; `reduction.md`) corrects pixel response but
   **not** optics-level structure.
2. **Effectively unpolarized (or polarization-scrambled) illumination.**
   Any source polarization is modulated by the HWP and split by the Savart
   into an o/e imbalance that is **imprinted into the flat** as if it were
   pixel response; dividing science frames by it injects spurious q,u.
   This is the hard part: the twilight sky is strongly linearly polarized
   (Rayleigh single scattering reaches ~80–94% at 90° from the Sun; the
   twilight zenith is strongly polarized — Ugolnikov et al. 2004, Source B),
   and dome screens/internal lamps are partially polarized by oblique
   reflection. FORS2 (Source B): *"sources like internal flat screens and
   twilight sky may introduce polarization that is difficult to eliminate."*
3. **Per-configuration multiplicity.** One flat set per EFW band (the α-BBO
   split is dispersive → per-band geometry) and — for the robust strategy —
   per HWP angle, redone whenever the optical train is touched (CasPol,
   Source B: dust-grain shadow positions changed every time the instrument
   was remounted, so only same-campaign flats were usable).

The dual-beam geometry also means a flat frame is the detector's response to
**two superposed, displaced images** of the illumination; in the field
interior of a uniform source this is benign, but edges/vignetted zones differ
between o and e — another reason `exclude_regions` (Savart chipped corner)
must come from these same flats.

## 3. Acquisition strategies in the literature (Source B)

| Strategy | How | Caveat|
|---|---|---|
| **Continuously rotating HWP** during each flat exposure (rotation ≫ faster than exposure) | the 4θ modulation of the source polarization averages to zero | needs hardware capable of continuous rotation; "in principle" the best (FORS2 paper), not possible on FORS2 itself|
| **Average flats over the HWP angle set** into one master | polarized illumination cancels in the sum over {0, 22.5, 45, 67.5}° | breaks if the source intensity drifts between angles — CasPol: *"not always effective because the intensity of the source … is not really stable"* (twilight!)|
| **Master flat per HWP angle** (≥10 flats at each angle; divide each science frame by the matching-angle flat) | CasPol's adopted method; also captures HWP-angle-dependent throughput | heaviest program: bands × angles × ≥10 frames, per campaign|
| **Sum all HWP angles of combined+matched o/e beams** | FORS2's depolarized flat construction | requires the beam-matching step first|
## 4. How bad is it really? (quantitative anchors)

- FORS2 (Source B): the flat correction reaches **~2% at the field edge**,
  yet applying it produced *"no significant difference in the maps of
  polarization degree and angle"* for their science case.
- CasPol (Source B): comparing the same standard-star frames reduced **with
  and without** flat-fielding, a Z-test and K–S test showed **no significant
  difference at their ~0.2–0.3% precision**.
- Why point-source work is forgiving: aperture photometry averages
  pixel-scale PRNU over ~150 px (≈ σ_PRNU/√n_eff, sub-0.1% for ~1% PRNU);
  what does **not** average is **large-scale structure (vignetting
  gradients, dust donuts) differing between the o and e aperture positions**
  over the ~239 px beam separation (0.9 mm α-BBO split on 3.76 µm pixels).
  That differential is the component that
  maps into q,u — and the component a mediocre flat handles worst.

## 5. Recommendation for POLITE

1. **Keep `lsq` default, but make the flat program explicit** in the night
   plan: per EFW band, skyflats at each HWP angle of the science set
   (≥10 each, CasPol-style), pointed at the **low-polarization twilight sky**
   (near the solar/antisolar meridian neutral points, away from 90°
   scattering); build one master per (band, angle). If the HWP stage supports
   smooth continuous rotation, rotating-HWP flats are the cleaner single-set
   alternative.
2. **Validate empirically every night that has standards:** reduce the same
   standard sequences with `method="lsq"` (flat-fielded) and
   `method="double_ratio"` (no flat). The (q,u) difference is a direct
   measurement of the night's flat quality — the pipeline already supports
   both, so this is one extra call.
3. **Decision rule (extends Q6):** if the lsq−double-ratio discrepancy on
   standards exceeds the target σ_q (sub-% science ⇒ ~0.1%), the night's
   flats are "bad" — use `double_ratio` (cost: only the 4-angle subset is
   used, no per-fit χ²/covariance). The CasPol/FORS2 results suggest that
   for **point sources** a reasonable flat program will pass this gate;
   Wolfe's warning bites hardest for extended-field/edge work.
4. The detector PRNU map (white frames) remains useful as a pixel-scale
   prior, but is **not** a substitute for through-the-optics flats.

## References

- Wolfe 2014, DUSTPol commissioning (Source A;
  `reference-sheets/_md/DUSTPol_Commissioning_v1.md` §2.1).
- Sosa et al. 2019, *JATIS* 5(2), 028002, §3.1/§4.5 (CasPol; Source B;
  arXiv:1903.01475).
- González-Gaitán et al. 2020, *A&A* (FORS2 imaging-polarimetry tips;
  Source B; arXiv:1912.08684).
- Ugolnikov, Postylyakov & Maslov 2004, twilight-sky polarization
  (Source B; arXiv:physics/0306070).
- Masiero et al. 2007 (DBIP), Ramírez et al. 2017 (SOLVEPOL) — local
  reference sheets (Source B).
