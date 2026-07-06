# poltools memory index

Router embedded in `.cursor/skills/analyst-poltools/SKILL.md` for auto-load.
**Edit both files** when modules or dependencies change.

Read target leaf before answering. Do not load all leaves.

## Modules

| Module | Role | Key symbols | Depends |
|--------|------|-------------|---------|
| `_types.py` | Config + result containers | `PolConfig`, `BeamGeometry`, `FilterConfig`, `StokesResult`, `default_efw_filters` | caltools.SensorConfig |
| `mueller.py` | Forward Mueller model | `system_mueller`, `oe_intensities`, `M_hwp`, `M_retarder` | — |
| `simulate.py` | Synthetic FITS frames | `render_frame`, `simulate_sequence`, `make_scene` | mueller, io, caltools |
| `io.py` | FITS I/O + grouping | `read_pol_frame`, `write_pol_fits`, `group_pol_sequence` | caltools, _types |
| `photometry.py` | Detect, pair O/E, aperture flux | `detect_sources`, `pair_oe`, `measure_pair`, `photometer_sequence` | _types |
| `modulation.py` | HWP fit ? q,u | `lsq_modulation`, `double_ratio`, `double_difference`, `ratio_r` | _types, errors |
| `stokes.py` | Assemble StokesResult | `assemble_stokes`, `polarization_fraction_angle` | _types, errors |
| `errors.py` | Uncertainties + debias | `residual_sigma_p`, `debias_mas`, `sigma_theta_nkc` | — |
| `calibration.py` | Standard-star cal | `PolCalibration`, `fit_instrumental_polarization` | — |
| `pipeline.py` | End-to-end reducer | `reduce_to_stokes` | all above |

## Data flow

```
frames ? io.group_* ? photometry (pair O/E) ? modulation (q,u)
       ? calibration ? stokes.assemble_stokes ? StokesResult
```

## Question routing

| Question type | Read first | Then |
|---------------|------------|------|
| variable / default | INDEX row ? target `.py` | INSTRUMENT.md if hardware default |
| literature / method | SCIENCE.md | target module docstring |
| non-obvious / interactions | INDEX deps column | QUIRKS.md + caller module |
| prior Q overlap | LEDGER.md (grep) | only if miss, read source |

## Leaf files

- `SCIENCE.md` — methods, papers, estimator choices
- `INSTRUMENT.md` — POLITE hardware, PolConfig defaults
- `QUIRKS.md` — coords, FITS keywords, gotchas
- `LEDGER.md` — answered atoms (append-only)
