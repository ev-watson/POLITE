# caltools

Detector characterization library for astronomical imaging sensors.

Developed for the POLITE QHY268M (Sony IMX571) but usable for any FITS-based imaging detector. The library builds master calibration frames, measures read noise and dark current, determines conversion gain from the **photon transfer curve**, tests linearity, and maps **pixel-to-pixel sensitivity variation** (photo-response non-uniformity).

## Quick start

```python
import caltools as ct

config = ct.sensor_config_from_header("frame.fits", gain=0.5)
bias = ct.master_bias(bias_paths)
read_noise_map, temporal_std_map = ct.read_noise_map(bias_cube)
ptc = ct.photon_transfer_curve(flat_groups, bias, config)
```

## Package layout

| Module | Role|
|--------|------|
| `_types.py` | `SensorConfig`, `AnalysisResult`, frame/cube aliases|
| `io.py` | FITS loading, grouping, header parsing|
| `stacking.py` | Master bias, dark, flat (chunked for memory)|
| `stats.py` | Online mean/variance, robust scatter, gaussianity tests|
| `noise.py` | Read noise, dark non-uniformity, fixed pattern noise, telegraph noise|
| `dark.py` | Dark current vs exposure/temperature, warm pixels|
| `gain.py` | Photon transfer curve, full-well capacity, noise decomposition|
| `linearity.py` | Signal-vs-exposure linearity and fractional error|
| `prnu.py` | Pixel sensitivity map from flat fields|
| `plotting.py` | Diagnostic figures|
## Conventions

- Detector arrays use `[row, col] == [y, x]` with origin upper-left (row 0 at the top).
- FITS frames with `BZERO=32768` are loaded with `memmap=False` so astropy applies scaling.
- Header metadata is authoritative for grouping: `IMAGETYP` and `EXPTIME` are required.
- Supply conversion gain and pixel size explicitly when the FITS header does not contain `EGAIN` or `XPIXSZ`; there are no camera-model defaults.

## Requirements

- Python 3.12+
- numpy, scipy, astropy, matplotlib
