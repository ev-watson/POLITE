# caltools

Detector characterization library for astronomical imaging sensors.

Developed for the POLITE QHY268M (Sony IMX571) but usable for any FITS-based imaging detector. The library builds master calibration frames, measures read noise and dark current, determines conversion gain from the **photon transfer curve**, tests linearity, and maps **pixel-to-pixel sensitivity variation** (photo-response non-uniformity).

Every analysis returns an `AnalysisResult` — a dataclass with `name`, `scalar_summary` (floats), `maps` (arrays), and `metadata`. Reductions consume that uniform container; `summary_table()` renders a list of them.

## Quick start

```python
import glob

import caltools as ct

paths   = sorted(glob.glob("FITSDATA/20260717/**/*.fits", recursive=True))
groups  = ct.group_by_type_and_exposure(paths)          # {(IMAGETYP, exptime): [path]}
config  = ct.sensor_config_from_header(paths[0], gain=1.0, pixel_size_um=3.76)

bias = ct.master_bias(groups[("BIAS", 0.0)])
rn_map, temporal_std = ct.read_noise_map(ct.load_cube(groups[("BIAS", 0.0)]))

# Conversion gain: the PTC over a full flat ladder is the definitive measurement.
flat_groups = {exp: p for (typ, exp), p in groups.items() if typ == "FLAT"}
ptc  = ct.photon_transfer_curve(flat_groups, bias)
fwc  = ct.full_well_capacity(ptc, config)

print(ct.summary_table([ptc, fwc]))
```

`sensor_config_from_header` reads geometry from the header; supply `gain` and `pixel_size_um` explicitly, because POLITE headers deliberately carry no measured `EGAIN`/`RON` (those are per-night reduction inputs, not as-acquired state).

## Package layout

| Module | Role |
|--------|------|
| `_types.py` | `SensorConfig`, `AnalysisResult`, `Frame`/`FrameCube`/`ROI` aliases |
| `io.py` | `load_frame`, `load_cube`, `load_cube_chunked`, `sensor_config_from_header`, `group_by_type_and_exposure`, `get_timestamps` |
| `stacking.py` | `master_bias`, `master_dark`, `master_flat` (chunked for memory) |
| `stats.py` | `WelfordVariance`, `mad_sigma`, `outlier_mask`, `gaussianity_tests`, `sigma_vs_mean_2d` |
| `noise.py` | `read_noise_map`, `read_noise_map_from_paths`, `read_noise_spatial`, `row_column_noise`, `dsnu`, `fpn`, `detect_rtn_pixels` |
| `dark.py` | `dark_current_vs_exposure`, `dark_current_vs_temperature`, `arrhenius`, `dark_spatial_structure`, `warm_pixel_map` |
| `flat.py` | `photon_transfer_curve`, `photon_transfer_curve_with_ron`, `conversion_gain_from_flat_pair`, `full_well_capacity`, `noise_decomposition`, `momsdom` |
| `linearity.py` | `linearity_test`, `linearity_error` |
| `prnu.py` | `prnu_map` |
| `plotting.py` | `image_with_colorbar`, `quick_view`, `ptc_plot`, `momsdom_twilight_plot`, `dark_current_vs_exposure_plot`, `dark_current_vs_temperature_plot`, `histogram_gaussian_overlay`, `noise_map_with_histogram`, `summary_table` |

## Measuring conversion gain

Two routines, and they are not interchangeable:

- **`photon_transfer_curve(flat_groups, bias)`** — the result that counts. Fits variance against signal across a flat ladder, so it resolves the read-noise intercept instead of assuming it away. `photon_transfer_curve_with_ron` returns the intercept explicitly; `full_well_capacity(ptc, config)` consumes the PTC.
- **`conversion_gain_from_flat_pair(flat_a, flat_b, bias_a, bias_b)`** — a single-point Janesick check for use at the telescope while frames are still landing. It assumes equal illumination and a linear regime, and reports `preliminary: True` plus `flat_pair_stable` (False when the pair drifted more than 2%). Use `central_fraction` to skip vignetted corners. Never cite it as the night's gain.

## Dark current vs temperature

`dark_current_vs_temperature` takes one fitted slope per setpoint (from `dark_current_vs_exposure`) rather than averaging `master_dark / exptime`, because offsets at long exposures otherwise dominate the short ones. The Arrhenius fit is **opt-in** (`arrhenius_fit=True`) and needs at least three temperatures spanning ≥ 1 °C; when it is off, under-determined, or non-convergent, the result carries `arrhenius_fit_failed` instead of the coefficients, and `dark_current_vs_temperature_plot` draws the measured points without an overlay. With only three setpoints the fit is a diagnostic, not a precise activation-energy measurement.

## Conventions

- Detector arrays use `[row, col] == [y, x]` with origin upper-left (row 0 at the top).
- FITS frames with `BZERO=32768` are loaded with `memmap=False` so astropy applies scaling.
- Header metadata is authoritative for grouping: `IMAGETYP` and `EXPTIME` are required. Detector temperature is read from `DET-TEMP`, with `CCD-TEMP` accepted as a legacy fallback.
- Supply conversion gain and pixel size explicitly; there are no camera-model defaults.

## Requirements

- Python 3.12+
- numpy, scipy, astropy, matplotlib
