# POLITE code and reduction audit

Date: 2026-07-10  
Scope: repository architecture, offline reducibility, detector calibration,
polarimetric provenance, and the 2026-07-09 salvage-analysis path.

## Outcome

**ESTABLISHED:** The repository can now be imported and tested on an offline
reduction machine without an installed Alpaca camera driver. The full test suite
passes: **144 passed**.

**ESTABLISHED:** Master-frame construction is now genuinely row-streamed. The
old implementation called `load_frame()` for every file in every row chunk. For
25 QHY268M frames and 50-row chunks, that reread roughly 100 GB to produce a
1.3 GB input-stack result. The replacement opens each FITS once, memory-maps the
unscaled signed storage, applies `BSCALE`/`BZERO` to only the requested strip,
and reads each pixel once.

**ESTABLISHED:** Polarimetric reduction now fails closed if frames from multiple
`OBJECT/POLSEQ` experiments are passed to one reduction. This prevents a target,
repeat, and +45° rotation from being median-combined into a scientifically
meaningless result.

**ESTABLISHED:** The salvage dataset has a reproducible, non-destructive analysis
path. Raw FITS are unchanged; all provisional products are under
`generated/salvage_first_light_20260709/`.

## Highest-risk defects found and fixed

### 1. Pure analysis imports pulled in the complete hardware stack

Before:

```text
import obs_utils.qa_lib
  -> execute obs_utils/__init__.py
  -> import imaging
  -> import alpyca_tools.camera_ops
  -> import alpaca.camera
  -> ModuleNotFoundError on an offline reduction computer
```

The same eager-import problem existed in `alpyca_tools/__init__.py` and in the
otherwise pure FITS-header builders.

Fix:

- `obs_utils` and `alpyca_tools` now expose their public APIs through lazy
  `__getattr__` mappings.
- Hardware classes used only for typing are under `TYPE_CHECKING`.
- Camera operations are imported inside the functions that actually touch
  hardware.

Why it matters:

- **ESTABLISHED:** `scripts/sequence_audit.py` now runs without Alpaca.
- **ESTABLISHED:** timing, QA, Pyxis protocol, and FITS-header tests collect on
  the reduction machine.
- Observatory commands retain the old public import surface; hardware errors
  occur only when hardware is requested.

### 2. “Chunked” FITS stacking was memory-bounded but I/O-explosive

Old pseudocode:

```python
for row_chunk in detector:
    for path in paths:
        full_50_mb_frame = fits.getdata(path, memmap=False)
        cube.append(full_50_mb_frame[row_chunk])
    master[row_chunk] = median(cube)
```

New algorithm:

```python
with open_all_fits_once(memmap=True, do_not_scale_image_data=True) as hdus:
    for row_chunk in detector:
        for hdu in hdus:
            raw_strip = hdu.data[row_chunk, selected_columns]
            physical_strip = raw_strip * BSCALE + BZERO
            cube.append(physical_strip)
        master[row_chunk] = median(cube, axis=0)
```

Additional safeguards:

- Empty path lists and non-positive chunk sizes raise named errors.
- Reducers accept only `median` or `mean`; typos no longer silently select mean.
- Frame-shape mismatches raise with the offending filename.
- A calibrated flat with non-finite or non-positive median signal cannot be
  normalized. This is the gate that correctly rejects this night’s zero-signal
  “super-flat.”
- A path-based `read_noise_map_from_paths()` computes temporal and
  frame-difference maps without allocating the full 25×4210×6280 cube.

### 3. Repeated HWP frames could be overwritten

`group_pol_sequence()` formerly returned an `OrderedDict[angle, path]`. A second
frame at an already-present angle replaced the first.

It now returns `OrderedDict[angle, list[path]]`. Repeats remain explicit and are
tested.

### 4. Different experiments could be silently merged

`reduce_to_stokes()` grouped only by filter. Passing the session directory would
therefore combine:

```text
driftA_polV8
driftA_polV8_rep2
driftA_polV8_rot45
superflatV_drift
```

at each HWP angle. This destroys repeat scatter, instrument-rotation information,
source identity, and drift history.

The reducer now inspects `(OBJECT, POLSEQ)` before image combination. More than
one sequence raises unless `allow_mixed_sequences=True` is explicitly supplied.
The intended multi-sequence API is:

```python
import poltools as pt

by_sequence = pt.reduce_pol_sequences(
    frame_paths,
    cfg,
    o_positions=ordinary_positions,
    method="double_ratio",
)

for (object_name, polseq, band), results in by_sequence.items():
    persist(object_name, polseq, band, results)
```

`allow_mixed_sequences=True` remains available only for an intentional repeat
stack whose registration and physical equivalence have already been established.

### 5. The standard-star QA did not select the named standard

`run_first_light_qa(..., ref_name="HD 154445", band="V")` previously did not use
`ref_name` to select files. A directory could reduce the first detected source
from another target or calibration block and compare it to HD 154445.

It now requires all of:

```text
IMAGETYP = LIGHT
OBJECT   = requested reference name
FILTER   = requested Photometric band
```

No match is a named QA failure, not a fallback to arbitrary session FITS.

### 6. Beam geometry was treated as real when it was a placeholder

The saved sidecar said `beam_separation_px: 60.0`, PA=0° implicitly. The observed
V-band pair is approximately 239.5 px apart at detector PA 328.2°. Automatic
pairing at the placeholder geometry necessarily failed.

Changes:

- Session detector configuration now carries separation, detector PA, and a
  `beam_geometry_characterized` flag.
- FITS fallback reads `BEAMSEP` and `BEAMPA`; geometry is characterized only if
  both cards exist.
- Sidecar snapshots include separation, PA, and characterization state.
- Automatic detection emits a warning when active-filter geometry is a
  placeholder.
- Default filter order now matches the installed carousel:
  `Clear, Photometric B, Photometric V, Photometric R, Dark`.

**OPEN:** The current sidecar schema still has one detector-level fallback
geometry. The final multi-band implementation should serialize a complete
per-filter geometry registry. The V-band value measured from this night remains
in `generated/` until reviewed; it has not been promoted to an instrument
calibration.

### 7. Invalid efficiency silently disabled calibration

`apply_efficiency(q, u, efficiency=0)` previously returned uncorrected values.
Zero, negative, infinite, and NaN efficiency now raise. A physically undefined
calibration can no longer masquerade as a calibrated result.

### 8. Floating exposure artifacts fragmented calibration groups

Header values such as `0.2000000000109` and `0.2` were separate keys in
`group_by_type_and_exposure()`. Exposure values are now canonicalized to
microsecond precision. A real 0.050 versus 0.051 s difference remains separate.

This distinction is present in the salvage data: 12 of 160 super-flat frames are
0.051 s while the others are 0.050 s.

### 9. Duplicate detector package removed

There were two divergent implementations:

```text
caltools/*.py             # imported from the repository root and packaged
caltools/caltools/*.py    # stale duplicate imported only from some CWD layouts
```

The nested duplicate was removed. `caltools/pyproject.toml` explicitly maps the
single root implementation as the package.

## New reproducible analysis commands

```bash
python scripts/analyze_salvage_first_light.py FITSDATA/20260709
python scripts/reduce_salvage_drift_sequence.py FITSDATA/20260709
```

The first command produces:

- a 278-frame header inventory;
- per-`POLSEQ` completeness inventory;
- master bias and matched 0.05 s master dark;
- frame-difference read-noise map;
- conservative bad-pixel mask;
- representative corrected images;
- per-angle super-flat feasibility metrics.

The second command:

- detects and follows the observed pair frame by frame;
- measures the actual split vector;
- measures drift speed from the subset of valid timestamps;
- performs masked local aperture photometry;
- fits `a₀ + q cos(4θ) + u sin(4θ)` as an uncalibrated diagnostic;
- never reports calibrated P or PA.

## Reduction architecture after the audit

```text
raw FITS (immutable)
  ├─ header_inventory.csv
  ├─ sequence_inventory.csv
  └─ detector calibration
       ├─ master_bias.fits
       ├─ master_dark_0.05s.fits
       ├─ read_noise_map.fits
       └─ bad_pixel_mask.fits

each OBJECT/POLSEQ/FILTER independently
  ├─ stationary sequence -> poltools.reduce_to_stokes
  └─ drifting commissioning sequence
       ├─ detect each frame
       ├─ match using measured (Δx, Δy)
       ├─ local masked photometry before combination
       ├─ retain per-frame fluxes and track
       └─ diagnostic modulation fit
```

The key simplification is that image stacking is no longer treated as the first
operation for every use case. Stationary repeats may be image-combined; drifting
frames must be registered or photometered individually first.

## Review findings intentionally not hidden by code

### Repository-wide lint backlog

**ESTABLISHED:** Focused Ruff checks pass on every file changed or added by this
audit. A repository-wide scan still reports 72 findings in 34 other files. Most
are import-order warnings caused by script-local `sys.path` bootstrapping and
unused imports, but the backlog also includes legacy bare-except/type-comparison
findings in `pwi4_client.py` and undefined-name findings in the concurrently
modified checklist generator. Those unrelated paths were not mechanically
rewritten because the worktree contains user-owned changes. They should be a
separate cleanup with behavior-specific tests.

### Calibration uncertainty propagation

**OPEN:** `PolCalibration` applies instrumental-polarization subtraction,
efficiency, and PA rotation, but the pipeline does not yet propagate the
calibration covariance into science `q,u`. The final API should carry a
calibration parameter covariance and add its Jacobian-propagated term:

```python
science_cov = J_measurement @ cov_qu @ J_measurement.T
calib_cov = J_calibration @ cov_calibration @ J_calibration.T
total_cov = science_cov + calib_cov
```

This omission does not affect the current report because no calibrated P/PA is
claimed.

### Per-frame registration in the general polarimetry API

**OPEN:** The salvage tracker is deliberately a separate commissioning tool.
The production `poltools` API still assumes one source position across the
sequence. A future general reducer should accept a table keyed by filename and
source ID:

```text
filename, source_id, ordinary_x, ordinary_y, extraordinary_x, extraordinary_y
```

and should photometer each exposure before optional robust combination.

### Acquisition/header duplication

**OPEN:** `alpyca_tools.fits_writer` and `obs_utils.fits_routine` still contain
overlapping FITS-header construction logic. Their imports are safe now, but a
future consolidation should make one schema/card builder authoritative and
leave each capture backend responsible only for acquiring pixels.

### Notebook role

**ESTABLISHED:** `notebooks/reduction.ipynb` is an exploratory detector notebook
for older datasets, not a reproducible first-light reducer. The two new scripts
are the executable record for this night. Future notebooks should consume their
CSV/JSON/FITS products rather than reimplement ingestion, grouping, or master
construction in cells.

## Verification

Commands run:

```bash
pytest -q tests/
python scripts/sequence_audit.py FITSDATA/20260709
python scripts/analyze_salvage_first_light.py FITSDATA/20260709
python scripts/reduce_salvage_drift_sequence.py FITSDATA/20260709
```

Results:

- **144 tests passed**.
- Four science sequences are structurally complete at 8/8 distinct HWP angles.
- Analysis completed on 278 FITS files without editing raw data.
- Generated products remain unreviewed and therefore remain under `generated/`.

## Worktree boundary

Pre-existing or concurrent deletions outside the explicitly listed `caltools`
duplicate-package removal were not created, restored, or otherwise altered by
this audit. They remain user-owned worktree changes.
