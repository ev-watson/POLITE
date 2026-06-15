# poltools

Imaging polarimetry simulator and Stokes-extraction pipeline for dual-beam small-telescope polarimeters (rotating half-wave plate + ?-BBO Savart analyzer).

Developed for the POLITE observatory (PlaneWave CDK20 + QHY268M), but usable for any instrument that records **ordinary** and **extraordinary** beams on a single detector while stepping a half-wave plate.

## Optical model

```
Sky ? telescope ? field rotator ? half-wave plate ? filter wheel ? ?-BBO Savart analyzer ? detector
```

Each point source appears twice on the detector. As the half-wave plate rotates, the ratio of extraordinary to ordinary flux modulates with **4?**, encoding linear Stokes parameters *Q* and *U*.

The library provides:

- **Forward model** — Mueller matrices, 2-D point-spread functions, detector noise (via `caltools`)
- **Reduction** — source detection, beam pairing, aperture photometry, modulation fit, standard-star calibration
- **Uncertainties** — Modified Asymptotic debiasing (Plaszczynski et al. 2014), Naghizadeh–Khouei & Clarke (1993) position-angle intervals, SOLVEPOL residual polarization uncertainty

Linear (*I*, *Q*, *U*) reduction is implemented now. The Mueller layer uses full 4-vector Stokes for possible future circular-polarization work.

## Installation

`poltools` depends on `caltools` (FITS I/O and `SensorConfig`). Install both from the POLITE repository or add them to `PYTHONPATH`:

```bash
pip install numpy scipy astropy photutils matplotlib
```

## Quick start

```python
import numpy as np
import poltools as pt
from caltools import SensorConfig

sensor = SensorConfig(nx=6280, ny=4210, pixel_size_um=3.76, gain_e_per_adu=1.0,
                      temperature_c=-10.0, sensor_name="QHY268M")
cfg = pt.PolConfig(sensor=sensor, beam=pt.BeamGeometry(separation_px=60.0))

rng = np.random.default_rng(0)
scene = pt.make_scene([(320, 240)], [(0.03, 0.02)], [5e6], names=["star"])
paths = pt.simulate_sequence(scene, cfg, out_dir="sim", rng=rng, shape=(512, 512))

results = pt.reduce_to_stokes([str(p) for p in paths], cfg,
                              o_positions=[(320, 240)], method="lsq")
print(results[0])
```

When flat fields are poor or missing, use `method="double_ratio"` (Tinbergen 1996; Masiero et al. 2007) — flat-field errors cancel in the ratio-of-ratios.

## Package layout

| Module | Role |
|--------|------|
| `_types.py` | `PolConfig`, `BeamGeometry`, `FilterConfig`, `StokesResult` |
| `mueller.py` | Mueller matrices, ordinary/extraordinary intensity model |
| `simulate.py` | Synthetic 2-D FITS frames |
| `io.py` | Polarimetry FITS keywords, filter and angle grouping |
| `photometry.py` | Detection, beam pairing, aperture photometry |
| `modulation.py` | Half-wave plate sequence ? normalized *q*, *u* |
| `stokes.py` | Polarization fraction, position angle, error budget |
| `errors.py` | Debiasing and uncertainty estimators |
| `calibration.py` | Instrumental polarization, efficiency, position-angle zero-point |
| `pipeline.py` | `reduce_to_stokes` end-to-end reducer |
| `plotting.py` | Diagnostic figures |

## Reduction methods

| Method | When to use |
|--------|-------------|
| `lsq` (default) | Flat-fielded frames; at least four half-wave plate angles; returns covariance and chi-squared |
| `double_ratio` | Bad or missing flats; requires {0°, 22.5°, 45°, 67.5°} |
| `double_difference` | Cross-check; first-order flat-field cancellation |

The Savart plate is dispersive: beam separation changes with filter. Register each filter slot with `default_efw_filters()` and reduce with `cfg.for_filter(name)`.

## Requirements

- Python 3.12+
- numpy, scipy, astropy, photutils, matplotlib
- caltools (sibling package)

## Citation

If you use this software in published work, cite the underlying polarimetry methods (Masiero et al. 2007, PASP 119, 1126; Plaszczynski et al. 2014, MNRAS 439, 4048; Ramírez et al. 2017, MNRAS 472, 2793) and acknowledge the POLITE observatory software.
