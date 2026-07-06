# poltools quirks

## Coordinates

- Positions: `(x, y)` pixels, origin **upper-left**
- Array indexing: `data[y, x]` — source at `(x,y)` ? `array[y,x]`
- Plots: origin upper-left (project rule)

## Stokes conventions

- 4-vector `(I, Q, U, V)` throughout; pipeline measures linear pol only
- Returned V=0 is **not** a measurement
- Extraordinary beam carries analyzer +Q? axis
- PA zero-point set by standard-star calibration, not intrinsic to ratio R

## Beam pairing

- E position = O position + `BeamGeometry.offset_xy()`
- `position_angle_deg` default 0 ? offset along +y
- Pairing tolerance tied to `separation_px`; wrong sep ? wrong Stokes

## Detector / CMOS

- `caltools.load_frame` with `memmap=False` (uint16 BZERO=32768)
- Optional per-pixel read-noise map (RTN tails)
- Bad-pixel mask + local interpolation before apertures
- Saturation flag when peak hits `sat_limit_e()`

## Warnings to know

- `INSTROT` drift across sequence smears PA (pipeline warns if >0.05°)
- Unregistered filter ? falls back to `cfg.beam` with warning
- `double_ratio` / `double_difference` require exact {0,22.5,45,67.5}°

## Package boundary

- `caltools`: FITS I/O, SensorConfig, detector noise model
- `poltools`: polarimetry-specific only
- Tests in `tests/test_*.py` encode expected behavior

## Test conventions

- RNG seeded in simulate tests
- Method cross-checks: lsq vs double_ratio vs double_difference
- Mueller round-trip: forward model ? modulation fit
