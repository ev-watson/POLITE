# Roadmap from POLITE commissioning to the EB-draft science program

Date: 2026-07-10  
Vision source: `reference-sheets/polarimetry/papers/EB-draft.pdf`, converted and
read as `EB-draft.md` using the required PDF-to-Markdown workflow.

## What the draft is asking this project to become

`EB-draft` is written as a mature AAS-style science paper, not a first-light or
instrument note. Its intended endpoint is:

1. repeatable, calibrated B–I eclipsing-binary photometry;
2. exposure timestamps traceable to UTC and converted to BJD_TDB;
3. a physical, band-aware eclipse model;
4. explicit treatment of correlated noise and astrophysical chromatic effects;
5. a distance-scaling test for an inter-band propagation delay;
6. a coverage-validated upper limit and a complete physical-systematics budget.

**ESTABLISHED:** The current POLITE system is not at that endpoint. It has just
demonstrated partial detector-frame dual-beam tracking under a mount failure.
That is a useful commissioning datum, but it is several validation layers below
publishable eclipse timing.

The recommended project structure is therefore two connected but distinct
tracks:

```text
Track A — instrument qualification
  detector -> timing -> mount/guiding -> filter/optics -> repeatability

Track B — EB science
  target pilot -> multi-epoch light curves -> joint physical model
  -> blinded validation -> distance regression -> paper
```

Track B must not begin producing physics constraints until Track A’s hard gates
are satisfied.

## Immediate gap analysis

| Draft requirement | Current state | Gap | Gate to close it |
|---|---|---|---|
| Reliable pointing/tracking | DEC drive disconnects mount | No target identity or full eclipse tracking | Repair DEC; home, slew, guide, and plate-solve repeatedly |
| B and I photometry | Wheel is Clear, B, V, R, Dark | No I band | Install a characterized Cousins-I path; preferably expand wheel or use simultaneous dual-band optics |
| BJD_TDB timestamps | 116/278 dates invalid; no `TIME-SRC` | Timing provenance failed | External timing validation plus fail-closed header contract |
| Stable detector state | Science near −15 °C; bias near −19 °C | Calibration mismatch | Temperature value and slope gate |
| Measured conversion gain/linearity | EGAIN/RON are placeholders | S/N and saturation unverifiable | PTC, RON, linearity, full-well characterization in the exact operating mode |
| Science flats | Super-flat has zero signal | No flat correction | Uniform high-ADU flats per filter/optical state |
| Stationary sources | 66 px/s drift | Pair exits before full sequence | Tracking/guiding and cadence budget |
| Calibrated optical geometry | V split placeholder wrong by ≈4× | Pairing/configuration invalid | Per-filter beam separation/PA and distortion map |
| Simultaneous/controlled colors | Mechanical cycling only | Color sampling and atmosphere are confounded with time | Simultaneous B/I preferred; otherwise measured wheel latency and interleaving model |
| Physical eclipse model | Prototype text only; packages absent | No executable model or tests | Versioned EB analysis package with synthetic recovery |
| Multi-epoch/systematics controls | No EB data | Astrophysical timing shifts unmeasured | Repeat eclipses, filter-order reversals, null/control targets |

The current environment has Astropy 7.2 and Photutils 3.0. `ccdproc`, `ellc`,
`emcee`, `dynesty`, and `linmix` are not installed. This is not evidence that
those packages are unsuitable; it means dependency support and Python-version
compatibility must be checked before the implementation is committed.

## Scientific corrections required before treating EB-draft as a result

The PDF is a useful target form, but its current numerical result should be
treated as a prototype. Several internal issues must be resolved.

### 1. Propagation-delay sign

The draft defines Δc = c_B − c_I and ΔT = T_B − T_I, then writes
ΔT = d Δc / c². First-order expansion gives:

```text
t_B − t_I = d(1/c_B − 1/c_I) ≈ −d(c_B − c_I)/c²
            = −(d/c)(Δc/c)
```

**ESTABLISHED:** Under the draft’s stated signs, the equation needs a minus sign.
An absolute upper limit is unchanged, but fitted slope interpretation and every
signed table/plot must follow one convention.

### 2. Residual table uses the opposite subtraction

The draft defines:

```text
ΔT_resid = ΔT_obs − ΔT_astro
```

but the displayed examples use the opposite sign. For Algol, 46.89 − 27.25 =
+19.64 s, while the table shows −19.65 s. V505 Sgr similarly gives +28.42 s
under the written equation but is tabulated as −28.42 s.

**ESTABLISHED:** The implementation, table, and sign prose are inconsistent.
The future pipeline must generate tables directly from one tested data model,
not from manually copied numbers.

### 3. Mid-eclipse versus ingress timing is conceptually blurred

The draft motivates timing with ingress morphology, but fits per-band
mid-eclipse times. In an ideal symmetric eclipse with a correct band-dependent
surface-brightness model, limb darkening changes shape and depth; whether it
creates a nonzero best-fit mid-time offset depends on model misspecification,
surface asymmetry, cadence, and what parameters are shared.

**OPEN:** A deterministic 20–50 s “limb-darkening timing correction” is not yet
established for this program. It must be demonstrated by forward simulations
and checked against the literature before being subtracted from data.

Recommended approach: fit one joint physical model in which geometry and event
time are shared, band-specific intensities/limb darkening are explicit, and a
putative propagation term enters the photon arrival time directly. Do not fit
independent times and then subtract a posterior-median template correction
unless injection-recovery proves that estimator unbiased.

### 4. Parameter count and identifiability

The draft calls the model “19-parameter” while listing four per-band quantities
(`J`, two limb-darkening coefficients, and `T0`) for four bands, plus shared
geometry, period, and reference epoch. The listed count and parameter list do
not agree, and independent per-band `T0` values can be redundant with a shared
reference epoch.

**ESTABLISHED:** The model specification needs a machine-readable parameter
schema and identifiability tests.

### 5. MCMC convergence claims

The draft says a longer production chain should reduce a physical timing
uncertainty by a factor of 3–5. Longer chains reduce Monte Carlo error when the
posterior is already sampled; they do not shrink the data-conditioned posterior.

**ESTABLISHED:** A result whose posterior width changes materially with chain
length is not converged. Use multiple independent initializations, rank/split
diagnostics appropriate to the sampler, autocorrelation-aware effective sample
size, and simulation-based calibration.

RZ Cas should not simply be excluded because walkers found a phase alias. Fix
the ephemeris/parameterization, demonstrate multimodality, or apply a
pre-registered data-quality exclusion independent of the desired result.

### 6. Red-noise treatment

The draft uses cyclic residual permutation and an additional arbitrary ×1.2
safety factor. A scalar inflation can be a diagnostic, but it cannot replace a
generative noise model when cadence, airmass, seeing, guiding, and filter are
correlated with eclipse phase.

**OPEN:** Select the final correlated-noise method only after synthetic and
out-of-eclipse validation. Candidate approaches include a Gaussian process over
time plus measured covariates, block/bootstrap validation, and time-averaging
diagnostics. Any chosen method must be traceable to a peer-reviewed Source B.

### 7. Regression cross-check is not yet convincing

The displayed Bayesian slope uncertainty is 0.0226 s/pc while the BCES
uncertainty is 0.74 s/pc, a factor ≈33 apart. Agreement of central values near
zero does not establish agreement of inference.

**ESTABLISHED:** The two methods are not equivalent as presented. Coverage and
upper-limit calibration must be tested using the actual five-target design,
distance uncertainties, intercept, intrinsic scatter, and selection function.

### 8. Systematics cannot be set to zero because they were not measured

The table assigns zero starspot variability because only one epoch exists and
uses one Algol residual as the limb-darkening floor. It also may count distance
uncertainty both inside the errors-in-variables regression and in a separate
systematic sum.

**ESTABLISHED:** “Unmeasured” is not “zero.” Single-epoch activity, third light,
period/apsidal changes, filter-dependent detector latency, and source-model
misspecification must remain open nuisance terms or motivate additional data.

### 9. Target list and hardware details need reconciliation

- The draft selection says declination >−20°, but V505 Sgr is listed at about
  −29°45′.
- The draft requires Cousins I, which the installed wheel does not contain.
- It claims GPS verification, while the QHY268M has no GPS receiver and this
  night demonstrated false in-buffer GPS parsing.

These are correctable planning inconsistencies, but they must be fixed before
observing proposals or target schedules are generated.

## Recommended final measurement model

### Observation-level time model

For exposure `j`, band `λ`, target `i`, and epoch `k`:

```text
t_recorded = t_true_mid_exposure
             + camera_latency(filter, exposure, readout_mode)
             + clock_offset(time)
             + timestamp_measurement_error
```

Convert the corrected UTC mid-exposure to BJD_TDB using site location and a
plate-solved target coordinate.

### Joint light-curve model

Use one event-time parameter per physical eclipse epoch, not one unrelated time
per filter:

```text
flux(i,k,λ,j) = eclipse_model(
    time = BJD_TDB(i,k,λ,j) − propagation_shift(i,λ),
    shared_geometry = {inclination, radii, eccentricity, omega, ...},
    band_surface_brightness = J[i,λ],
    limb_darkening = LD[i,λ],
    third_light = L3[i,λ],
    exposure_integration = actual_EXPTIME[j],
)
+ baseline(time, airmass, seeing, centroid, sky, comparison_color)
+ correlated_noise
```

Choose a reference band, for example I:

```text
propagation_shift(i, I) = 0
propagation_shift(i, B) = −(d_i/c) ξ
ξ = (c_B − c_I)/c
```

This places the sign convention in one function and allows a direct global
posterior/likelihood for ξ rather than a correction-then-regression chain.

### Source-level chromatic nuisance model

Astrophysical chromatic shifts should be explicit:

```text
δ_chromatic(i,k) = target_mean[i]
                 + epoch_activity[i,k]
                 + modeled_spot_or_asymmetry[i,k]
```

The propagation term scales with distance. Target/epoch astrophysical terms do
not automatically do so, but selection effects can create correlations; the
hierarchy and target controls must be simulated.

## Staged work program

## Phase 0 — Restore safe observatory operation

Deliverables:

- repaired DEC drive with documented root cause;
- 30-minute tracking test at three declinations;
- repeated home/slew/plate-solve/guide cycles;
- emergency stop and no-home recovery procedure;
- mount state recorded in every science FITS.

Hard gate:

```python
assert mount.connected
assert mount.declination_axis_healthy
assert plate_solve_rms_arcsec < requirement
assert guiding_rms_arcsec < photometry_aperture_budget
```

No EB data should be scheduled before this gate passes.

## Phase 1 — Detector operating point

Acquire at the exact proposed science mode/gain/offset/temperature:

1. ≥50 bias frames after thermal equilibrium;
2. darks across planned exposure times and at least three temperatures;
3. uniform flat pairs across a signal ramp;
4. linearity sequence through saturation onset;
5. repeated sequence on multiple nights.

Products:

```text
conversion gain [e-/ADU]
read-noise map [e-]
linearity curve and maximum allowed ADU
full-well/ADC behavior
bad/RTN/warm pixel masks
dark-current model versus temperature and exposure
PRNU map per filter
```

Exposure selection must target integrated precision and linearity:

```python
def acceptable_exposure(frame, calibration):
    peak = robust_source_peak(frame)
    return (
        peak > calibration.minimum_precision_adu
        and peak < calibration.linearity_limit_adu
        and saturated_fraction(frame) == 0
    )
```

Do not copy `EGAIN=1.0` or `RON=3.5` from a plan into a science header unless
those values are measured and versioned.

## Phase 2 — Timing metrology

The science signal is a time difference, so timestamp calibration is an
instrument calibration, not metadata polish.

Required tests:

1. disable QHY GPS parsing unless a physical receiver is installed;
2. record NTP source, phase offset, last-sync age, and uncertainty;
3. expose a GPS-disciplined LED or other externally timed optical signal;
4. measure requested-start to photon-integration start, exposure duration, and
   readout-mode/filter dependence;
5. repeat across exposure times, temperatures, and computer load;
6. test midnight/date rollover and leap-second policy;
7. reject any timestamp outside the current observing window.

Header contract:

```text
DATE-OBS  actual exposure start UTC
DATE-END  actual exposure end UTC
MJD-OBS   derived start
TIMESYS   UTC
TIME-SRC  measured source, e.g. NTP/GPS-disciplined host
TIMEUNC   validated 1σ or conservative bound [s]
NTP-OFFS  measured clock offset [s]
NTP-AGE   time since last synchronization [s]
```

Fail-closed code sketch:

```python
def validate_timing(header, now_utc, requirement_s):
    start = Time(header["DATE-OBS"], scale="utc")
    end = Time(header["DATE-END"], scale="utc")
    assert abs((start - now_utc).sec) < 3600
    assert header["TIME-SRC"] in approved_sources
    assert header["TIMEUNC"] <= requirement_s
    assert end > start
    assert abs((end - start).sec - header["EXPTIME"]) < duration_tolerance
```

## Phase 3 — Optical/filter architecture

### Minimum path

Install and characterize Cousins I. With a five-slot wheel, sacrificing Dark or
Clear is operationally awkward; a larger wheel is preferable.

### Recommended path

**CONJECTURED engineering recommendation:** For a differential timing experiment,
a dichroic or other simultaneous B/I two-channel system is materially safer than
mechanical filter cycling. It removes atmospheric evolution and eclipse slope
from the B-versus-I sampling order and eliminates wheel transition latency as a
first-order confounder.

If cycling is retained:

- measure filter-wheel settle latency on every transition;
- alternate the starting band between cycles/epochs;
- use a symmetric sequence such as B–I–I–B when cadence permits;
- model each exposure at its actual integration interval;
- include filter order as a null-test covariate.

The polarimetric HWP/Savart chain is not required for EB timing and splits flux.
Evaluate a bypass/imaging mode. If the Savart remains in the path, differential
photometry must combine both beams and validate that HWP/beam throughput does
not create band-dependent time structure.

## Phase 4 — Photometric reduction package

Create a package separate from notebooks, for example:

```text
ebtiming/
  manifest.py        immutable raw-file/header inventory
  validation.py      timing, temperature, counts, filter, pointing gates
  calibration.py     master bias/dark/flat and bad-pixel masks
  astrometry.py      plate solution and source identity
  photometry.py      ensemble differential photometry
  time.py            UTC midpoint -> BJD_TDB
  detrend.py         covariates and out-of-eclipse baseline
  eclipse.py         exposure-integrated joint multi-band model
  inference.py       optimization/sampling and convergence checks
  systematics.py     injection, filter-order, activity, timing tests
  population.py      distance-scaled hierarchical model
  products.py        provenance-rich tables/figures
tests/
  test_time_sign.py
  test_exposure_integration.py
  test_injection_recovery.py
  test_filter_order_null.py
  test_population_coverage.py
```

Core reduction sketch:

```python
manifest = ingest(raw_paths)
manifest.require_single_instrument_configuration_per_epoch()
manifest.require_valid_timing()

masters = build_calibrations(
    bias=manifest.bias,
    dark=manifest.dark.group_by(actual_exposure, detector_temperature),
    flat=manifest.flat.group_by(filter_name),
)

for frame in manifest.science:
    calibrated = masters.apply(frame)
    wcs = solve_or_validate_wcs(calibrated, frame.header)
    sources = fixed_catalog_positions(wcs)
    fluxes = aperture_grid_with_growth_curve(calibrated, sources)
    store(frame.id, fluxes, diagnostics={
        "sky": sky,
        "fwhm": fwhm,
        "centroid": centroid,
        "airmass": airmass,
        "linearity_margin": linearity_margin,
    })

light_curve = ensemble_differential_photometry(
    target,
    comparisons=select_stable_color_matched_comparisons(),
)
light_curve.time = utc_midpoint_to_bjd_tdb(
    light_curve.utc_midpoint,
    target_coord=plate_solved_coord,
    site=observatory_location,
)
```

Every rejected exposure must retain a reason code; no manual silent deletion.

## Phase 5 — Single-target pilot

Before a distance sample, observe one bright, well-characterized EB over
multiple complete eclipses.

Each epoch must include:

- ≥30 minutes out-of-eclipse baseline on both sides, preferably more;
- full ingress and egress;
- no saturation/nonlinearity;
- stable guiding and defocus;
- several color-matched comparison stars;
- alternating filter order or simultaneous B/I;
- weather/seeing/sky/centroid covariates;
- external timing check before and after.

Pilot success criteria:

```text
same-band split-half timing null passes
filter-order reversal gives consistent inter-band offset
aperture-radius grid gives consistent timing
comparison-star leave-one-out gives consistent timing
injected offsets are recovered without bias
repeat epochs agree after modeled astrophysical variability
posterior predictive residuals show no eclipse-phase structure
```

## Phase 6 — Model validation and blinding

Build synthetic data with real cadence, noise, missing frames, covariates, and
known injected ξ. Validate:

1. no-injection false-positive rate;
2. interval coverage at several ξ values;
3. recovery under starspots/third light/limb-darkening mismatch;
4. robustness to filter-order and timestamp offsets;
5. distance-regression coverage with the actual target distances;
6. target-exclusion and jackknife behavior.

Recommended blind protocol:

```python
secret_offset = blinding_service.per_target_band_offset()
analysis_time = physical_time + secret_offset

# Freeze selection, reduction, model, and quality gates.
freeze_pipeline_commit()

# Reveal only after injection recovery and null tests pass.
physical_result = blinding_service.unblind(frozen_result)
```

This prevents iterative choices from following the sign or size of the desired
effect.

## Phase 7 — Target sample and observation design

For every candidate, record:

```text
coordinates and observability
Gaia distance posterior and quality flags
period/ephemeris uncertainty at observation date
eclipse depth and ingress/egress duration
third light/crowding
spectral types and atmosphere parameters
activity/spot history
expected B and I count rates
comparison-star field quality
```

Reject/retain criteria must be frozen before inspecting inter-band residuals.
V505 Sgr must either be admitted by a revised declination criterion or removed
from a sample claiming Dec>−20°.

Use an exposure/cadence simulator before scheduling:

```python
for target in candidates:
    counts = throughput_model(target.sed, filter, aperture, detector)
    cadence = exposure + readout + measured_filter_transition
    timing_precision = inject_and_fit(
        physical_eclipse(target),
        counts=counts,
        cadence=cadence,
        scintillation=site_model,
        red_noise=pilot_noise_model,
    )
    retain_if(timing_precision, full_eclipse_visibility, comparison_field)
```

## Phase 8 — Population inference and constraint

Two acceptable architectures should be compared on simulations:

1. a fully joint hierarchical light-curve model with global ξ;
2. validated per-epoch timing likelihoods followed by a hierarchical
   distance-scaling model.

The second is computationally simpler but must propagate full, possibly
non-Gaussian timing likelihoods rather than only a point and symmetric error.

Do not label an interval “Feldman–Cousins” unless a Neyman construction with the
actual nuisance-handling and ordering rule has demonstrated coverage. A basic
coverage harness is:

```python
for xi_true in xi_grid:
    covered = 0
    for simulation in range(n_simulations):
        dataset = simulate_full_program(xi=xi_true, nuisance_draw=True)
        interval = analysis_pipeline(dataset).confidence_interval
        covered += interval.low <= xi_true <= interval.high
    assert abs(covered / n_simulations - 0.95) < coverage_tolerance
```

## Physical error budget required for the paper

The final table should contain a measured bound or nuisance model for every row.

### Timing chain

| Effect | Measurement/control | Propagation |
|---|---|---|
| Host clock offset/drift | NTP/GPS status plus external optical timing | Per-exposure correlated time nuisance |
| Camera start latency | Timed-light laboratory test versus exposure/mode | Filter/mode-dependent offset |
| Exposure-duration error | Timed-light test and DATE-END | Midpoint and integration-kernel uncertainty |
| Filter-wheel latency | Transition telemetry | Actual cadence; possible band-dependent offset |
| UTC→TDB and barycentric conversion | Tested against trusted examples | Deterministic with site/coordinate uncertainty |
| Coordinate/plate-solve error | WCS residuals | Barycentric correction uncertainty |

### Detector and reduction

| Effect | Measurement/control | Propagation |
|---|---|---|
| Nonlinearity/saturation | PTC/linearity ramp; peak gate | Flux-shape distortion injection tests |
| PRNU/flat error | Independent flats and flat splits | Re-reduce with flat realizations |
| Dark/bias/temperature mismatch | Thermal model and matched masters | Calibration ensemble |
| Bad/RTN pixels | Temporal mask | Mask sensitivity and aperture tests |
| Guiding/centroid drift | Per-frame centroids | Covariate and aperture-grid test |
| Seeing/defocus changes | FWHM/shape metrics | Variable-aperture or PSF-model comparison |
| Background gradients | Local/2D background alternatives | Reduction-method spread |
| Comparison-star instability/color | Ensemble leave-one-out | Hierarchical comparison ensemble |

### Atmosphere and sampling

| Effect | Measurement/control | Propagation |
|---|---|---|
| Extinction/color extinction | Airmass and comparison colors | Band baseline nuisance |
| Differential chromatic refraction | WCS centroids versus airmass/color | Aperture-loss/time-shape nuisance |
| Scintillation | Site/aperture/exposure model plus residuals | Correlated noise |
| Clouds/transparency | Ensemble comparison flux | Common-mode correction and frame flag |
| Non-simultaneous colors | Simultaneous optics or filter-order controls | Explicit cadence forward model |

### Astrophysical model

| Effect | Measurement/control | Propagation |
|---|---|---|
| Limb darkening/model grid | Multiple atmosphere grids/priors | Model averaging or nuisance hierarchy |
| Starspots/activity | Multi-epoch/color residuals | Epoch-dependent chromatic term |
| Third light/crowding | Catalog, high-resolution image, model | Band-specific dilution nuisance |
| Period/ephemeris error | Joint multi-epoch timing | Shared orbital model |
| Apsidal motion/LTTE/period change | Literature plus multi-epoch fit | Orbital timing nuisance |
| Exposure smearing | Integrate model over actual exposure | Deterministic forward model |
| Target selection | Frozen criteria and simulation | Selection-aware population validation |

### Population/physics conversion

| Effect | Measurement/control | Propagation |
|---|---|---|
| Distance posterior | Gaia posterior and quality filtering | Sample distance in hierarchy |
| Intercept/intrinsic scatter | Explicit nuisance | Joint posterior/profile construction |
| Dominant distant target | Jackknife and design simulation | Report sensitivity; add distant targets |
| Sign/unit conversion | Unit-tested single function | Generated tables and plots only |
| Interval coverage | Monte Carlo Neyman/coverage tests | Determines reportable confidence statement |

No row should be assigned zero solely because the program lacks repeat data.

## Verified literature basis

The following works cited by the draft were verified to exist through OpenAlex
on 2026-07-10. Verification establishes bibliographic existence, not automatic
endorsement of every use in the draft.

| Use | Verified work |
|---|---|
| UTC/BJD_TDB timing | Eastman, Siverd & Gaudi (2010), “Achieving Better Than 1 Minute Accuracy in the Heliocentric and Barycentric Julian Dates,” DOI [10.1086/655938](https://doi.org/10.1086/655938), [OpenAlex W1999652311](https://openalex.org/W1999652311) |
| Detached-EB forward model | Maxted (2016), “ellc: A fast, flexible light curve model for detached eclipsing binary stars and transiting exoplanets,” DOI [10.1051/0004-6361/201628579](https://doi.org/10.1051/0004-6361/201628579), [OpenAlex W2332990190](https://openalex.org/W2332990190) |
| Limb-darkening tables | Claret & Bloemen (2011), “Gravity and limb-darkening coefficients…,” DOI [10.1051/0004-6361/201116451](https://doi.org/10.1051/0004-6361/201116451), [OpenAlex W2135634498](https://openalex.org/W2135634498) |
| Red-noise motivation | Pont, Zucker & Queloz (2006), “The effect of red noise on planetary transit detection,” DOI [10.1111/j.1365-2966.2006.11012.x](https://doi.org/10.1111/j.1365-2966.2006.11012.x), [OpenAlex W2160374032](https://openalex.org/W2160374032) |
| Regression with measurement error | Kelly (2007), “Some Aspects of Measurement Error in Linear Regression of Astronomical Data,” DOI [10.1086/519947](https://doi.org/10.1086/519947), [OpenAlex W2063617102](https://openalex.org/W2063617102) |
| Unified confidence intervals | Feldman & Cousins (1998), “Unified approach to the classical statistical analysis of small signals,” DOI [10.1103/PhysRevD.57.3873](https://doi.org/10.1103/PhysRevD.57.3873), [OpenAlex W2128947363](https://openalex.org/W2128947363) |

The user-provided polarimeter commissioning and reduction materials remain
Source A for the current dual-beam methods. Any additional scientific method
must be tied to a peer-reviewed Source B and verified before implementation.

## Definition of the project’s final position

The project is ready to claim the EB-draft vision only when all of the following
are true:

1. the telescope reliably points, tracks, guides, and plate-solves;
2. B and I are both characterized, preferably simultaneous;
3. timing latency and clock error are externally measured and meet a frozen
   requirement;
4. detector gain, RON, nonlinearity, flats, and thermal behavior are measured in
   the science configuration;
5. the complete reduction/model pipeline passes blinded synthetic recovery and
   null tests;
6. at least one multi-epoch pilot target produces repeatable inter-band timing;
7. target selection and exclusions are pre-registered;
8. the distance-scaling interval has demonstrated coverage;
9. every physical-systematics row has data, a bound, or an explicit nuisance
   model;
10. all paper tables and figures are generated from frozen, versioned products.

Until then, the appropriate publication target is an instrument/commissioning
report, not a vacuum-dispersion constraint.
