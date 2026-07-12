# 2026-07-09 salvage instrument-commissioning reduction

Date reduced: 2026-07-10  
Raw data: `FITSDATA/20260709/`  
Provisional products: `generated/salvage_first_light_20260709/`

## Correct objective and executive verdict

This night was not a failed attempt to obtain a science target. The mount could
not be trusted, so the correct objective was to salvage instrument calibration:
bias/dark behavior, flat-field feasibility, beam geometry, HWP modulation,
whole-instrument rotation, plate scale if possible, focus evidence, and a
complete failure record. The drifting stellar exposures were deliberate
calibration exposures: stars were used as moving probes of the dual-beam
geometry, not as identified astrophysical targets.

| Commissioning question | Verdict | Evidence |
|---|---|---|
| HWP modulation test | **Inconclusive; no significant fourth-harmonic signal detected.** | The usable partial stellar calibration track gives amplitude 0.0168 with conservative S/N=0.66. The moving probe is not a known polarized source, and only 10/16 frames contain both beams. This does not falsify the HWP. |
| Beam geometry measurement | **Successful and the strongest result of the night.** | Independent drifting blocks give 5 and 10 matched pairs. Their measured vectors agree: `driftA_polV8` has Δx=−123.4±3.0 px, Δy=+202.6±1.1 px, separation 237.24 px; `driftA_polV8_rep2` has Δx=−126.3±3.8 px, Δy=+203.5±2.0 px, separation 239.52 px. This directly replaces the 60 px/0° placeholder for provisional V-band commissioning. |
| Frame-by-frame drift tracking | **Successful over the interval in which both beams are visible.** | The pair track is recovered in 5/16 frames in the first block and 10/16 in the repeat. The repeat track is linear to 1.73 px RMS. Later loss is useful edge-of-field/drift information, not a target-acquisition failure. |
| +45° whole-polarimeter rotation test | **Inconclusive/not testable.** | No comparable pair survives in the rotated block; all light frames lack `INSTROT`; the untracked field moved many detector widths during manual rotation. |
| V-band flat-field calibration | **Failed to produce a usable flat.** | At every angle, the 20-frame calibrated median is 0.0 ADU with ≈2.46 ADU scatter. There is no positive illumination pedestal to normalize. |
| Plate-scale measurement | **Not independently measured.** | 0.224 arcsec/px is header/configuration input. The measured drift is consistent with sidereal motion under that assumed scale, but no astrometric solution exists. |
| Focus measurement | **Not performed.** | No deliberate focus sweep or focus metric was recorded. The stellar probes are compact enough to measure geometry, but that is not a focus calibration. |

The scientifically honest first-light outcome is therefore:

- **ESTABLISHED:** the camera recorded a geometrically stable dual image and the
  HWP command sequence completed at all requested angle labels;
- **ESTABLISHED:** the moving stellar probes provide a usable provisional V-band
  beam vector, detector drift velocity, and edge-of-field track;
- **ESTABLISHED:** the detector calibration stack and defect population are
  measurable, although the flat illumination was inadequate;
- **INCONCLUSIVE:** HWP modulation, because the probe polarization is unknown,
  the usable angle coverage is partial, and the fourth-harmonic fit is below
  detection;
- **OPEN:** modulation efficiency, instrumental polarization, absolute PA, HWP
  zero point, independent plate scale, focus, and the +45° transformation.

No calibrated P or PA should be extracted from this night. That restriction does
not reduce the value of the data: the primary deliverable is instrument
characterization, not sky characterization.

## Data inventory and provenance

### Frames

| Type | Count | Temperature range | Notes |
|---|---:|---:|---|
| BIAS | 25 | −19.6 to −18.0 °C | Filter header says Photometric R; 8 timestamps invalid. |
| DARK | 45 | −17.4 to −15.5 °C | V-band blocking configuration; requested 0.05, 0.2, 2, and 5 s groups; 14 timestamps invalid. |
| LIGHT | 208 | −15.9 to −14.8 °C | All Photometric V; 94 timestamps invalid; all lack pointing and `INSTROT`. |
| Total | 278 | −19.6 to −14.8 °C | 116 invalid/missing 2026 `DATE-OBS`; zero `TIME-SRC` cards. |

The camera set point was −20 °C, but stellar/light frames were near −15 °C. The
bias stack is therefore 3–5 °C colder than the light/dark data.

### Polarimetric sequences

| POLSEQ | Frames | Distinct angles | Frames/angle | Valid / invalid timestamps |
|---|---:|---:|---:|---:|
| `superflatV_drift` | 160 | 8 | 20 | 85 / 75 |
| `driftA_polV8` | 16 | 8 | 2 | 9 / 7 |
| `driftA_polV8_rep2` | 16 | 8 | 2 | 10 / 6 |
| `driftA_polV8_rot45` | 16 | 8 | 2 | 10 / 6 |

**ESTABLISHED:** Angle-set completeness is real: every block contains 0°, 22.5°,
45°, 67.5°, 90°, 112.5°, 135°, and 157.5°. This is a metadata/acquisition
success, not proof of a valid modulation measurement.

### Calibration objective coverage

| Requested salvage product | Data taken | Reduction status | Commissioning value |
|---|---|---|---|
| Bias ×25 | 25 full-frame biases | **Usable provisionally** | Bias pedestal, fixed-pattern defects, pinned pixels, and a first read-noise map |
| Matched darks | 45 darks at four nominal exposure groups | **Usable provisionally** | Dark/bias behavior and exposure-group inventory; temperature mismatch remains |
| V flats through full train | 160 V light frames, 20 per HWP angle | **Not usable as a normalized flat** | Proves the chosen 0.05 s sky-flat exposure had no measurable illumination; does not test flat construction at adequate counts |
| HWP modulation test | Three drifting 8-angle stellar sequences | **Inconclusive** | Provides moving beam probes and a partial fourth-harmonic test; no known-polarization standard |
| Beam geometry | Same stellar sequences | **Successful provisionally** | Direct measurement of the V-band pixel split and its stability |
| Plate scale | No plate-solving/astrometric frames | **Not measured** | 0.224 arcsec/px remains an assumed configuration value |
| Focus | No focus sweep | **Not measured** | Stellar frames demonstrate detectability, not an optimum focus value |
| Whole-instrument +45° test | One rotated 8-angle block | **Inconclusive/not testable** | Rotation block exists but lacks source continuity and `INSTROT` metadata |
| Failure documentation | FITS metadata plus known DEC/timestamp faults | **Partly complete** | The failure modes are reconstructable; second-copy verification and operator log were not present in the FITS set |

This table is the correct interpretation of the night. “Not usable” refers to a
calibration product’s ability to calibrate later data; it does not mean the
frames were pointless or should be discarded.

### Exposure-time details

- Science blocks contain 47 frames reported as 0.050 s and one rotated frame as
  0.051 s.
- The super-flat contains 148 frames at 0.050 s and 12 at 0.051 s.
- The matched-dark set contains 14 frames within microseconds of 0.050 s and one
  frame at 0.051 s.

The 1 ms difference is negligible for dark current at these temperatures, but
it is a 2% illumination difference if the frames contained appreciable sky
signal. A real flat builder must normalize each illuminated input before
combination and must not group 0.050 and 0.051 s merely because filenames look
similar.

## Reduction method

### 1. Master bias and matched dark

The reduction uses row-streamed median combination:

```python
bias_paths = frames(IMAGETYP="BIAS")
dark_paths = frames(IMAGETYP="DARK", EXPTIME≈0.050)

master_bias = median_stack_chunked(bias_paths)
master_dark_005 = median_stack_chunked(
    load(dark_paths) - master_bias
)

corrected_light = raw_light - master_bias - master_dark_005
```

Measured master properties:

| Product | Median | Sigma-clipped σ | Important tails |
|---|---:|---:|---|
| Master bias | 491.0 ADU | 1.51 ADU | 0 to 59,649 ADU fixed defects |
| 0.05 s master dark minus bias | 0.0 ADU | 2.39 ADU | −2,560 to +2,944 ADU unstable/temperature-sensitive defects |
| Frame-difference read-noise map | 5.62 ADU | spatial clipped σ 1.11 ADU | maxima >20,000 ADU at pathological pixels |
| Temporal standard-deviation map | 5.55 ADU | spatial clipped σ 0.92 ADU | maxima >21,000 ADU |

The master-bias scatter is smaller than single-frame read noise because 25
frames were median combined.

### 2. Bad-pixel mask

The conservative commissioning mask is the union of 10-MAD outliers in:

```python
bad = (
    outlier(master_bias)
    | outlier(master_dark_005)
    | outlier(read_noise_map)
    | outlier(temporal_std_map)
)
```

It flags 97,054 / 26,438,800 pixels = **0.3671%**. This includes the 16,841
pixels pinned at zero in every sampled raw frame plus high/unstable pixels.

### 3. Super-flat feasibility test

The intended method is scientifically valid only when calibrated sky signal is
positive and large compared with the combined detector noise:

```python
for angle in hwp_angles:
    cube = load(superflat_frames[angle])
    calibrated = cube - master_bias - master_dark_005

    # A production night-sky flat would first scale each exposure by its robust
    # sky level, mask sources, and sigma-clip before combination.
    sky_levels = robust_median(calibrated, per_frame=True)
    require(all(level > minimum_flat_signal for level in sky_levels))
    scaled = calibrated / sky_levels[:, None, None]
    master = sigma_clipped_median(scaled, axis=0)
    require(master_median > 0)
    normalized = master / robust_median(master)
```

For this night, the precondition fails before normalization:

| HWP angle | N | Combined median | Combined clipped σ |
|---:|---:|---:|---:|
| 0.0° | 20 | 0.0 ADU | 2.460 ADU |
| 22.5° | 20 | 0.0 ADU | 2.462 ADU |
| 45.0° | 20 | 0.0 ADU | 2.461 ADU |
| 67.5° | 20 | 0.0 ADU | 2.459 ADU |
| 90.0° | 20 | 0.0 ADU | 2.459 ADU |
| 112.5° | 20 | 0.0 ADU | 2.460 ADU |
| 135.0° | 20 | 0.0 ADU | 2.459 ADU |
| 157.5° | 20 | 0.0 ADU | 2.460 ADU |

Dividing by this product would divide by noise and imprint detector residuals
onto the science. No normalized flat files were produced.

### 4. Frame-by-frame pair tracking

Image combination before tracking is invalid for these data because the moving
stellar calibration probe moves hundreds of pixels between exposures. The
usable sequence was reduced as:

```python
for path in one_polseq:
    image = load(path) - master_bias - master_dark_005
    peaks = find_smoothed_peaks(image, mask=bad, threshold=8_sigma)
    pair = match_vector(peaks, expected_dx=-127, expected_dy=+203,
                        tolerance=20)
    if pair:
        f1 = masked_local_aperture(image, pair.first, radius=20)
        f2 = masked_local_aperture(image, pair.second, radius=20)
        ratio = (f2 - f1) / (f2 + f1)
        retain(filename, hwp_angle, positions, f1, f2, ratio)
```

The measured detector split is consistent across the two independently acquired
unrotated blocks:

```text
driftA_polV8:     Δx = −123.4 ± 3.0 px, Δy = +202.6 ± 1.1 px
                  separation = 237.24 px, detector PA = 328.66°
driftA_polV8_rep2: Δx = −126.3 ± 3.8 px, Δy = +203.5 ± 2.0 px
                  separation = 239.52 px, detector PA = 328.17°
```

The values are not averaged into one formal calibration uncertainty because the
peak locations are detection centroids from different drifting probes and the
instrument was not independently illuminated. Their agreement at the few-pixel
level is nevertheless strong evidence that the V-band beam geometry is stable.

The sign labels “beam 1” and “beam 2”; ordinary versus extraordinary has not
been independently established. Swapping them changes the signs of q and u but
does not create a modulation detection.

### 5. Drift measurement

Seven tracked frames have valid 2026 timestamps. A straight-line fit gives:

```text
vx = +32.96 px/s
vy = −57.34 px/s
speed = 66.14 px/s
speed = 14.82 arcsec/s if PIXSCALE=0.224 arcsec/px
track residual RMS = 1.73 px
predicted trail during 0.050 s = 3.31 px
```

**CONJECTURED:** If the motion is pure sidereal drift and the plate scale is
correct, 14.82″/s is consistent with a field near the celestial equator because
the sidereal image rate is approximately 15 cos(Dec) arcsec/s. This does not
identify the field and is not a pointing solution.

The plan required <2 px trail. The measured maximum exposure satisfying that
criterion would be approximately:

```python
max_exposure_s = 2.0 / 66.14  # ≈0.030 s
```

Shortening the exposure, however, would further reduce an already poor source
S/N. The correct fix is tracking/guiding, not ever-shorter exposures.

### 6. Modulation diagnostic

For the ten frames with two positive beam fluxes, the diagnostic model was:

```text
R(θ) = a₀ + q cos(4θ) + u sin(4θ)
R = (F₂ − F₁)/(F₂ + F₁)
```

The free `a₀` absorbs first-order beam-throughput imbalance. This is not a
standard-star calibration.

The two unrotated blocks were deliberately kept separate. They are independent
drifting calibration probes, not repeated measurements of one fixed source at
the same detector coordinates. The primary fit below uses `driftA_polV8_rep2`
because it contains ten matched pairs spanning 0°–90°; the first block contains
only five matched pairs beginning at 67.5° and is used for geometry consistency,
not pooled into the modulation fit.

Fit result:

```text
a₀ = +0.0042
q_detector = +0.00185 ± 0.01631 (empirical)
u_detector = −0.01669 ± 0.01930 (empirical)
fourth-harmonic amplitude = 0.01679
conservative amplitude S/N = 0.66
ratio residual RMS = 0.0323
```

The reported flux uncertainties are detector/background lower bounds. A source
shot-noise term in electrons cannot be trusted because `EGAIN=1.0 e⁻/ADU` was a
plan placeholder, not a measured Mode-0/gain-0 conversion gain.

The beam-summed flux is also not constant: it falls from roughly 14.1 kADU to
9.8 kADU over the tracked portion. Transparency, vignetting, aperture loss,
source identity/blending, and the detector-edge trajectory are confounded with
HWP phase. This further prevents a clean optical interpretation.

## Physical and operational error ledger

| Error | Status | Consequence | Recoverable from these data? | Required correction |
|---|---|---|---|---|
| DEC drive failure; no pointing/tracking | **ESTABLISHED** | Unknown target; rapid drift; no sky PA; no complete pair track | No | Repair mount/DEC before science. |
| 0.05 s exposure retained without console tuning | **ESTABLISHED** | 3.31 px predicted trail; source signal near detector-noise regime | Partly | Use tracking; then tune counts, not merely trail length. |
| 0.05 s sky-flat exposure chosen to match the stellar probes | **ESTABLISHED** | Flat received zero usable sky signal | No | Flats need high, linear ADU and matched optical configuration; they do not need the stellar exposure time. |
| Night-sky super-flat median=0 ADU after calibration | **ESTABLISHED** | No flat-field correction possible | No | Illuminated twilight/dome screen or much longer tracked sky flats with validated signal. |
| Beam separation/PA left at 60 px/0° placeholder | **ESTABLISHED** | Automatic pairing fails | Yes for provisional V geometry | Measure per filter with illuminated/standard-star data; promote only after review. |
| Calibration probe leaves detector before sequence completion | **ESTABLISHED** | Only 10/16 frames usable for the HWP ratio test; beam geometry remains measurable | No | Hold the field fixed for a complete HWP test, or use a bench source. |
| Manual +45° rotation after untracked delay | **ESTABLISHED** | Original source moved many detector widths | No | Track continuously or reacquire/plate-solve the same source. |
| `INSTROT` missing in all light frames | **ESTABLISHED** | Rotation amount is not machine-verifiable | No | Record commanded and achieved whole-instrument angle per frame. |
| Mixed correct and 1996 `DATE-OBS` | **ESTABLISHED** | Absolute timing and cadence invalid for many frames | Only relative track using valid subset | Hardware-test GPS-disabled server; require `TIME-SRC`, clock offset, and valid-year gate before capture. |
| No `TIME-SRC` card | **ESTABLISHED** | Cannot demonstrate clock provenance | No | Fail preflight for timing science. |
| Cooler not stabilized | **ESTABLISHED** | Bias stack colder than lights/darks; defect behavior changes | Partly via matched dark | Wait for temperature and slope tolerance before every calibration/science block. |
| EGAIN and RON are placeholders | **ESTABLISHED** | Electron noise, saturation, and S/N are uncalibrated | No | Photon-transfer and read-noise characterization in exact read mode/gain/offset. |
| 0.050/0.051 s duration mixture | **ESTABLISHED** | 2% potential illumination scaling difference | Yes if signal existed | Store requested and actual durations; normalize illuminated flats individually. |
| Bias frames say Photometric R | **ESTABLISHED** | Misleading provenance, though bias is optically insensitive | Header only | Use blocking Dark slot for bias/dark and record shutter/cover state. |
| No RA/Dec/WCS | **ESTABLISHED/intentional salvage** | No field identity, airmass, sky orientation, or independent plate scale | No | Plate solve for astrometric calibration; this was not required to measure a pixel beam vector. |
| HWP angle is a header label without an independent achieved-error card | **OPEN** | Cannot bound angle-settle/zero error | No | Record requested angle, achieved angle, error, home state, and settle timestamp. |

## What this night can support

### Defensible commissioning statements

1. **ESTABLISHED:** A stable two-image geometry exists in Photometric V across
   two independent drifting stellar calibration blocks.
2. **ESTABLISHED:** The measured split is approximately 238–240 px at detector
   PA≈328°, not the 60 px placeholder.
3. **ESTABLISHED:** The detected images share an extremely linear detector track,
   supporting their association as a physical pair rather than unrelated random
   peaks.
4. **ESTABLISHED:** The detector contains a non-negligible population of pinned,
   high, and temporally unstable pixels that the reduction must mask.
5. **ESTABLISHED:** Bias and matched-dark stacks are constructible as provisional
   detector calibration products, with temperature/gain caveats.
6. **ESTABLISHED:** All four programmed blocks visited all eight HWP angle labels.
7. **INCONCLUSIVE:** The moving stellar probes do not show a significant HWP
   fourth harmonic; a known-polarization source is required to turn this into an
   efficiency/modulation test.
8. **ESTABLISHED:** The timestamp failure persisted intermittently during the
   night and must be treated as a hard preflight gate for timing science.

### Claims these data cannot support

- calibrated linear polarization fraction;
- sky-referenced polarization position angle;
- modulation efficiency or HWP retardance;
- instrumental polarization;
- HWP angle zero point;
- correct q/u transformation under +45° instrument rotation;
- target/catalog identification;
- a science-quality flat or PRNU map;
- an electron-calibrated error budget;
- any eclipsing-binary timing or vacuum-dispersion result.

The data do support a provisional V-band beam-geometry calibration, a drift/edge
behavior characterization, and a detector defect/read-noise inventory. Those are
the intended salvage products and should be retained even though the flat and
rotation tests are incomplete.

## Recommended follow-up calibration session

The next session should continue the instrument-commissioning objective even if
the DEC repair is not yet complete. A dead mount blocks sky science and a
sky-referenced rotation test, but it does not block detector calibration,
illuminated flats, bench HWP tests, beam geometry, or focus measurements.

### Gate A: detector operating point

```text
temperature within ±0.5 °C of set point
temperature slope <0.1 °C over 10 min
measured EGAIN and RON loaded for exact mode/gain/offset
peak signal between configured linear lower/upper limits
bad-pixel mask loaded
```

Take the bias and dark stacks only after thermal equilibrium. Take matched darks
for every actual exposure time used, grouping by measured duration rather than
filename. Run a small photon-transfer/linearity ramp so ADU values become
physical detector quantities.

### Gate B: illuminated V flats through the complete train

Use a twilight field, dome screen, or stable laboratory illuminator. A flat does
not need the short stellar-probe exposure time; it needs a high, uniform,
unsaturated signal in the same optical state, filter, HWP angle, focus regime,
gain, offset, readout mode, and detector temperature.

```python
for theta in hwp_angles:
    raw = [capture_uniform_V(hwp=theta, n=20,
                             target_median_adu=25_000)
           for _ in range(1)]
    calibrated = [frame - master_bias - matched_dark(frame) for frame in raw]
    assert all(15_000 < robust_median(frame) < 35_000 for frame in calibrated)
    assert all(saturated_fraction(frame) < 1e-5 for frame in calibrated)
    flat[theta] = sigma_clip_median(
        [frame / robust_median(frame) for frame in calibrated], axis=0
    )
```

The required output is one normalized flat per HWP angle, plus a flat-quality
report showing illumination level, source rejection, residual beam geometry, and
inter-angle throughput changes.

### Gate C: controlled HWP modulation test

Use a known unpolarized source and a known polarized source (laboratory source,
polarizer, or characterized standard). The drifting stars were useful geometry
probes but cannot establish modulation efficiency because their polarization is
unknown.

```python
for source in (unpolarized_source, polarized_source):
    hold_field_fixed()
    for theta in hwp_angles:
        achieved = move_hwp(theta)
        frame = expose_to_linear_regime()
        write_metadata(
            requested_hwp=theta,
            achieved_hwp=achieved,
            hwp_error=achieved - theta,
        )
    fit_R_of_4theta(source)
```

The unpolarized source measures instrumental polarization and the polarized
source measures efficiency and phase. Repeat after a known HWP zero-point
offset to check the transformation convention.

### Gate D: beam geometry and focus

The present night already demonstrated that the V split is about 239.5 px. The
next measurement should make that value a real calibration product:

```python
for filter_name in filters:
    for focus_position in focus_grid:
        frame = short_exposure_bright_source(filter_name, focus_position)
        pairs = detect_and_pair_without_placeholder_geometry(frame)
        record(pair_separation, pair_PA, centroid_residual, FWHM, focus_position)
```

Select focus from a measured FWHM/ellipticity metric, not visual compactness. Do
this per filter because the Savart geometry is dispersive. Measure plate scale
from a plate solve or a known astrometric field; the current 0.224 arcsec/px is
only a configuration input.

### Gate E: whole-instrument rotation

Use a fixed laboratory/bench source or a tracked astronomical source. Rotate the
whole polarimeter by a machine-recorded +45°, retain the source in the detector,
and record both commanded and achieved rotator angles in every frame.

```python
q0, u0 = reduce_detector_frame_sequence(angle_set_0)
rotate_whole_polarimeter(+45.0)
q45, u45 = reduce_detector_frame_sequence(angle_set_rotated)
assert metadata_has("INSTROT", all_frames=True)
compare_rotation(q0, u0, q45, u45, expected_angle=45.0)
```

This is the calibration experiment the salvage night could not complete. It is
separate from the HWP test and must remain a separate `POLSEQ`.

### Gate F: mount/timing recovery

The mount and timing defects remain hard gates for sky commissioning:

```text
DEC engages without disconnecting
mount homes and tracks for >=30 min
plate solve succeeds
DATE-OBS valid current UTC for 20/20 test frames
TIME-SRC present for 20/20
external timing check meets the science requirement
```

These gates are not prerequisites for the bench calibration products above, but
they are prerequisites for sky-referenced PA, identified standards, and the EB
timing program described in the separate roadmap.

## Evidence map

| Artifact | Purpose |
|---|---|
| `metrics.json` | Machine-readable full-session metrics |
| `header_inventory.csv` | Every relevant FITS card for all 278 frames |
| `sequence_inventory.csv` | Per-`POLSEQ` angle and timestamp completeness |
| `master_bias.fits` | Provisional median master bias |
| `master_dark_0.05s.fits` | Provisional matched dark minus bias |
| `read_noise_map.fits` | Frame-difference detector-noise map in ADU |
| `bad_pixel_mask.fits` | Conservative commissioning mask |
| `superflat_roi_metrics.csv` | Per-angle proof that flat signal is zero |
| `representative_corrected_frames.png` | Full-detector corrected overview, origin upper-left |
| `tracked_pair_photometry.csv` | Per-frame positions, fluxes, ratios, and timing validity |
| `tracked_pair_summary.json` | Geometry, drift, and modulation fit |
| `tracked_pair_diagnostics.png` | Detector track, ratio versus HWP, and total flux |
| `driftA_polV8/` | Same pair-tracking diagnostic applied independently to the first unrotated block |

All are provisional AI-generated analysis products and remain in `generated/`
pending human review.
