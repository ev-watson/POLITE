# POLITE instrument context

Active research; intent to publish. Methods must trace to SCIENCE.md refs.

## Hardware

| Component | Model | Notes |
|-----------|-------|-------|
| Telescope | PlaneWave CDK20 | |
| Camera | QHY268M | 6280×4210, 3.76 µm pixels |
| Analyzer | ?-BBO Savart plate | 18 mm; dispersive ? per-filter beam sep |
| Modulator | Rotating half-wave plate | 4-angle sequence |
| Filter wheel | ZWO 5-slot EFW | B, V, R, Clear, Dark |

## PolConfig defaults (POLITE-tuned)

| Field | Default | Rationale |
|-------|---------|-----------|
| `plate_scale_arcsec` | 0.224 | CDK20 + QHY268M plate scale |
| `hwp_angles_deg` | (0, 22.5, 45, 67.5) | Standard 4-angle modulation set |
| `retardance_deg` | 180.0 | Ideal HWP |
| `read_noise_e` | 3.5 | Mode0 @ gain 0 |
| `dark_rate_e_per_s` | 0.005 | Cooled operation |
| `full_well_e` | 51000 | QHY268M well depth |
| `beam.separation_px` | 60.0 (placeholder) | Must calibrate per filter (Savart dispersion) |
| `default_efw_filters()` | same sep all slots | Placeholder until per-band calibration |

## Savart / FITS metadata (io.py)

- `SAVMAT` = "alpha-BBO"
- `SAVTHK` = 18.0 mm
- Polarimetry keywords: `HWPANG`, `RETARD`, `INSTROT`, `POLBEAM`, `POLSEQ`, `POLEFF`, `PIXSCALE`

## SensorConfig example (README)

```python
SensorConfig(nx=6280, ny=4210, pixel_size_um=3.76, gain_e_per_adu=1.0,
             temperature_c=-10.0, sensor_name="QHY268M")
```

## Per-filter requirement

Savart split shifts with wavelength. Register `default_efw_filters()` then reduce with `cfg.for_filter(name)`. Using one separation for all bands mis-pairs O/E beams.

## Reduction defaults (pipeline)

- `method="lsq"` — flat-fielded data
- `estimator="mas"` — Plaszczynski debias for reported p
