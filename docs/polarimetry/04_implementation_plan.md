# POLITE Polarimetry Pipeline — Implementation Plan (Phase 4)

**Date:** 2026-06-07
**Scope:** Build `poltools/` — a telescope-chain **simulator** + **Stokes-extraction
pipeline** with publication-grade error metrics — and an **end-to-end showcase** that
fulfils the original report request. Linear (I,Q,U) now, full-Stokes-ready architecture.

All math is fixed by the Phase 1 research map (Sources A/B). All design choices are fixed by
Phase 3. This document is the executable contract for Phase 5 (execute) and the source of
the Phase 6 verification grid.

---

## 1. Package layout & build order

```
poltools/
  __init__.py     # public API, __version__ = "0.1.0"
  _types.py       # (1) PolConfig, BeamGeometry, PointSource, StokesResult, BeamFlux
  mueller.py      # (2) Stokes 4-vectors; M_rot, M_retarder(δ,φ), M_polarizer; oe_intensities()
  simulate.py     # (4) forward model → 2D FITS frames (uses caltools SensorConfig + io idiom)
  io.py           # (3) pol FITS keywords (HWPANG…), sequence grouping (reuses caltools.io)
  photometry.py   # (5) detect (photutils DAOStarFinder), pair o/e, aperture phot + sky σ
  modulation.py   # (6) Method A double-difference; Method B LSQ; → q,u (+σ)
  errors.py       # (7) σ-propagation, residual σ_P, σ_θ (hi/lo SNR), MAS/WK/naive debias
  stokes.py       # (8) assemble I,Q,U[,V], p, θ, covariance → StokesResult
  calibration.py  # (9) IP subtraction, PA zero-point, polarization efficiency (standards)
  plotting.py     # (10) modulation curves, q-u plane, polarization-vector map
  pipeline.py     # (11) end-to-end orchestrator: frames → calibrated StokesResult
tests/
  test_mueller.py, test_simulate.py, test_photometry.py, test_modulation.py,
  test_errors.py, test_calibration.py, test_pipeline.py
scripts/
  polarimetry_showcase.py   # (12) end-to-end run → figures + tables (the report deliverable)
docs/polarimetry/
  05_verification.md, 06_report.md  # Phase 6 / Phase 7 deliverables
```

**Build order** (dependency-respecting): _types → mueller → errors → modulation → simulate →
io → photometry → stokes → calibration → plotting → pipeline → tests → showcase. (errors &
modulation are pure-math and unit-testable before the heavier simulate/photometry, so we
front-load them and lock their tests.)

Import rule: `poltools → caltools` only (one-directional). No `poltools` ← `caltools` edge.

---

## 2. Module contracts (public API signatures)

### (1) `_types.py`
```python
@dataclass(frozen=True)
class BeamGeometry:
    separation_px: float       # o↔e split on detector
    position_angle_deg: float  # split direction (detector frame)
    # e position = o position + sep*(sin PA, cos PA) in (x,y)? -> documented convention

@dataclass(frozen=True)
class PolConfig:
    sensor: caltools.SensorConfig
    beam: BeamGeometry
    plate_scale_arcsec: float = 0.224
    hwp_angles_deg: tuple = (0.0, 22.5, 45.0, 67.5)
    retardance_deg: float = 180.0     # HWP; 90.0 ready for QWP/V mode
    instrument_rotator_deg: float = 0.0   # PWI4 field rotator (alpha)
    filter_name: str = "Clear"            # active EFW slot
    filters: tuple = ()                   # FilterConfig registry (per-band α-BBO geometry)
    def with_hwp_angles(self, angles) -> "PolConfig": ...
    def for_filter(self, name) -> "PolConfig": ...   # select an EFW band

@dataclass(frozen=True)
class PointSource:
    x: float; y: float                # detector px (origin upper-left per CLAUDE.md)
    flux_e: float                     # total source flux (e-) at I (both beams summed)
    stokes: tuple = (1.0,0.0,0.0,0.0) # normalized (1, q, u, v); q,u,v in [-1,1]
    name: str = ""

@dataclass
class BeamFlux:           # one source, one HWP angle, measured
    hwp_deg: float; f_o: float; f_e: float; sig_o: float; sig_e: float

@dataclass
class StokesResult:       # AnalysisResult-compatible (name, scalar_summary, maps, metadata)
    name: str
    scalar_summary: dict   # I,Q,U,(V),p,p_mas,theta_deg,sigma_p,sigma_theta_deg,snr,chi2…
    maps: dict
    metadata: dict         # method, n_angles, estimator, source-A/B provenance
```

### (2) `mueller.py`  (Source B: Mueller formalism; HWP eqn Masiero 2007 / DUSTPol)
```python
def stokes_vector(I,q,u,v=0.0) -> np.ndarray            # 4-vector (I, Q, U, V)
def M_rotator(theta_deg) -> 4x4                          # frame rotation
def M_retarder(delta_deg, fast_axis_deg) -> 4x4         # general retarder (HWP=180, QWP=90)
def M_linear_polarizer(transmission_axis_deg) -> 4x4    # ideal analyzer
def M_hwp(theta_deg, retardance_deg=180.0) -> 4x4       # convenience
def oe_intensities(stokes, hwp_deg, retardance_deg=180.0,
                   eff=1.0, ip=(0.0,0.0)) -> (I_o, I_e)
    # returns the two orthogonal analyzed intensities; ideal-HWP limit reproduces
    #   I'∥ = ½[I + Q cos4θ + U sin4θ],  I'⊥ = ½[I − Q cos4θ − U sin4θ]   (research map §2)
def system_mueller(hwp_deg, rotator_deg, retardance_deg, M_tel=None) -> 4x4
```
**V-ready:** everything is 4×4 / 4-vector; linear pipeline never *solves* for V (single HWP
cannot), but the matrices are correct for adding a QWP later.

### (7) `errors.py`  (Source B: SOLVEPOL/Magalhães; Plaszczynski 2014; Montier II; NK&C 1993)
```python
def propagate_qu_sigma(beam_fluxes, method) -> (sigma_q, sigma_u)   # analytic from photon σ
def residual_sigma_p(z, Q, U, N) -> float                          # √(2/(N−2)(Σz²−Q²−U²))
def debias_naive(p) -> p
def debias_wardle_kronberg(p, sigma_p) -> sqrt(max(p²−σ²,0))
def debias_mas(p, sigma_p, b2=None) -> p − b²(1−e^{−p²/b²})/(2p)    # Plaszczynski eq.20
def sigma_p_mas(sigma_q, sigma_u, theta_deg, I0) -> float           # Montier II eq.20
def sigma_theta_highsnr(p, sigma_p) -> 28.65*σ_p/p   [deg]
def sigma_theta_nkc(p_over_sigma, conf=0.6827) -> (lo,hi) [deg]     # Naghizadeh-Khouei&Clarke
```

### (6) `modulation.py`  (Source B/A: Masiero 2007 double-difference; SOLVEPOL LSQ)
```python
def ratio_R(f_o, f_e) -> (f_e−f_o)/(f_e+f_o)
def method_a_double_difference(beam_fluxes) -> dict(q,u,sigma_q,sigma_u)   # angles {0,45,22.5,67.5}
def method_b_lsq(beam_fluxes) -> dict(Q,U,sigma_Q,sigma_U,sigma_p_resid,chi2,z,model)
```

### (8) `stokes.py`
```python
def assemble_stokes(qu, errs, I0, name, method, provenance, estimator="mas") -> StokesResult
    # computes p=√(q²+u²), theta=½atan2(u,q) [0..180), p_mas, σ_p, σ_θ, snr, covariance
```

### (9) `calibration.py`  (Source B/A: Masiero 2007 §3; DUSTPol §3; Serkowski 1974)
```python
def fit_instrumental_polarization(unpol_standards) -> (q0,u0,cov)   # mean q,u of unpol stds
def apply_ip(q,u, q0,u0) -> (q−q0, u−u0)
def fit_pa_zeropoint(pol_standards) -> (dtheta_deg, sigma)          # vs literature θ
def apply_pa_zeropoint(q,u, dtheta_deg) -> rotate by 2·dθ
def fit_efficiency(highpol_standards) -> eff                        # measured p / literature p
def apply_efficiency(p, eff) -> p/eff
```

### (4) `simulate.py`  (forward model; research map §5)
```python
def render_frame(sources, cfg, hwp_deg, *, seeing_arcsec, sky_e_per_px, exptime_s,
                 dark_e_per_s, read_noise_e, bias_adu, rng, psf="gaussian",
                 prnu=None, retardance_deg=None) -> np.ndarray(uint16-equivalent ADU)
    # per source: oe_intensities → o,e total flux (e-) → 2D PSF at o,e positions
    # + sky; Poisson(shot) + dark + N(0,RON); /gain; +bias; +BZERO; clip to full-well/ADU max
def simulate_sequence(sources, cfg, *, out_dir, seeing_arcsec, sky, exptime_s, rng, …)
    -> list[Path]   # writes one FITS per HWP angle with HWPANG etc.; returns paths
```

### (3) `io.py`
```python
POL_KEYWORDS = dict(HWPANG=…, WPUNCERT=…, INSTROT=…, POLBEAM=…, POLSEQ=…, POLSEQN=…)
def write_pol_fits(path, data_adu, header_cfg, hwp_deg, cfg, **extra) -> Path  # reuses fits_routine idiom
def read_pol_frame(path, roi=None) -> (data, hwp_deg, header)                  # caltools.io load_frame
def group_by_hwp_angle(paths) -> dict[float, list[Path]]
def group_pol_sequence(paths) -> OrderedDict[hwp_deg → path]
```

### (5) `photometry.py`  (photutils — verified API; DBIP/SOLVEPOL aperture method)
```python
def detect_sources(image, fwhm_px, threshold_sigma=5.0) -> Table       # DAOStarFinder
def pair_oe(detections, beam: BeamGeometry, tol_px=2.0) -> list[SourcePair]
def measure_pair(image, pair, r_ap_px, r_in_px, r_out_px, gain, read_noise_e)
    -> BeamFlux   # CircularAperture sum − sky(ApertureStats median); σ from photon+sky+RON
def photometer_sequence(frames_by_angle, cfg, …) -> dict[source → list[BeamFlux]]
```

### (11) `pipeline.py`
```python
def reduce_to_stokes(frame_paths, cfg, *, method="A", estimator="mas",
                     calibration=None, …) -> list[StokesResult]
    # group by HWP → detect → pair o/e → aperture phot → modulation → calibration
    #   → stokes assembly + errors.  One StokesResult per source.
```

---

## 3. Verification grid (Phase 6 — drives `tests/` + showcase)

| Check | What it proves | Pass criterion|
|---|---|---|
| **Mueller unit** | `oe_intensities` reduces to research-map §2 ideal-HWP formula | exact to 1e-12 across θ|
| **Mueller V-leak** | real retarder depolarization scales as (1−cosφ) | matches Masiero 2007 ≤0.2% for φ∈[176.4,183.6]°|
| **Round-trip (math)** | inject (q,u) → oe fluxes → Method A/B → recover (q,u) | within 1e-6 (noiseless)|
| **Injection-recovery (2D)** | full sim→pipeline on FITS | recovered p,θ within quoted σ across p∈{0.1,0.5,1,3,5}%|
| **Pull distribution** | MC over noise realizations: (p̂−p_true)/σ_p | mean≈0, std≈1 (N(0,1)) at high SNR|
| **Flat-field independence** | Method A invariant to per-beam gain ratio + PRNU | q,u unchanged to <1e-3 when beam gains perturbed|
| **MAS bias removal** | low-SNR (SNe/asteroid) p̂ bias | naive biased high; MAS removes bias for SNR≳2|
| **σ_θ low-SNR** | NK&C interval vs MC | MC coverage matches NK&C CI|
| **Calibration** | IP subtraction + PA zero-point recover injected systematics | residual q0,u0,Δθ ≈ injected within σ|
| **FITS byte-compat** | sim frames read by `caltools.io` (memmap=False, BZERO) | round-trip pixel-exact|
MC uses fixed seeds. The showcase runs the realistic four-regime cases end-to-end.

---

## 4. End-to-end showcase (the report deliverable — `scripts/polarimetry_showcase.py`)
1. Build a `PolConfig` (QHY268M + per-filter α-BBO Savart geometry via
   `default_efw_filters`/`for_filter`) + a synthetic field with sources at
   **known** (I,q,u) spanning the four science regimes.
2. `simulate_sequence` → real FITS per HWP angle (with HWPANG) in `FITSDATA/SIM_<date>/`.
3. `pipeline.reduce_to_stokes` → per-source `StokesResult` (Method A, MAS).
4. Cross-check with Method B; run the MC pull test; apply mock standard-star calibration.
5. Emit publication figures (modulation curves, q-u plane with error ellipses, recovered-vs-
   injected, pull histogram) + a results table (p, σ_p, θ, σ_θ, p_MAS, SNR, χ²).
6. Write `docs/polarimetry/06_report.md` with methods (A/B-tagged), results, and the
   verification summary — the deliverable the original request asked for.

---

## 5. Risks / mitigations
- **photutils API drift** → pinned + API verified at install (Phase 5 first step). (checked) checked.
- **Origin convention** (CLAUDE.md: upper-left) → enforce `origin='upper'` in all imshow;
  document (x,y)↔(col,row) once in `_types`.
- **BZERO=32768** → all reads via `caltools.io` (`memmap=False`); sim writes BZERO=32768.
- **Scope creep to V** → architecture is V-ready but pipeline deliberately does not solve V;
  asserted/tested + documented so a zero-V is never read as a measurement.
- **Source A/B discipline** → method-provenance carried in `StokesResult.metadata`; no new
  analysis method introduced (photutils is an implementation library only).
