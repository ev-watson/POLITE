# POLITE

Observatory automation and detector characterization for a PlaneWave CDK20 + QHY268M (IMX571) system, controlled via ASCOM Alpaca and PlaneWave PWI4.

## Overview

POLITE provides end-to-end observatory control — mount startup, slewing, automated imaging, and calibration acquisition — alongside two analysis libraries:

- **`caltools`** — bias/dark/flat stacking, read noise, gain (PTC), linearity, PRNU, dark current
- **`poltools`** — dual-beam imaging-polarimetry simulator and Stokes-extraction pipeline

The instrument is a dual-beam imaging polarimeter (rotating HWP + α-BBO Savart analyzer):

```
Sky → PlaneWave CDK20 → PWI4 Focuser/Rotator → Astronomik L3 UV/IR-cut
    → rotating Half-Wave Plate → ZWO 5-slot EFW (Clear, Photometric B, V, R, Dark)
    → α-BBO Savart plate (18 mm) → QHY268M (IMX571)
```

Mount and rotator control use the PWI4 HTTP API; camera and filter wheel use ASCOM Alpaca. Night sessions are declarative YAML *brick* plans — previewed with `scripts/plan_night.py` and executed with `scripts/execute_night.py`.

## Project Structure

```
obs_utils/               Observatory control modules
  config.py              Dataclass configs (Alpaca, PWI4, sky limits)
  startup.py             Observatory startup sequencing
  mount.py               PWI4 mount control (slew, home, tracking)
  imaging.py             ASCOM Alpaca camera capture
  night_session.py       Declarative night session runner
  autoguide.py           Autoguiding and dithering
  platesolve.py          Plate solving interface
  pointing.py            Pointing model utilities
  fits_routine.py        Compatibility wrapper for the authoritative FITS writer
  logging.py             Session logging configuration
  user_config.py         Site-specific hardware configuration

alpyca_tools/            ASCOM Alpaca camera interface layer
  camera_device.py       Camera device abstraction
  camera_ops.py          Exposure control and readout
  fits_writer.py         FITS file writing with acquisition metadata
  discovery.py           Alpaca device discovery
  schema.py              Camera state schema
  telemetry.py           Telemetry collection
  scripts/               Diagnostic and snapshot scripts

caltools/                Detector characterization (v0.1.0)
  README.md              Package overview and quick start
  io.py                  FITS I/O, cube loading, header parsing
  stacking.py            Master bias, dark, flat generation
  stats.py               Welford accumulator, MAD sigma, outlier masking
  noise.py               Read noise maps, DSNU, FPN, RTN detection
  dark.py                Dark current vs exposure/temperature, warm pixels
  flat.py                Master flat, photon transfer curve, full well, noise decomposition
  linearity.py           Linearity testing and error characterization
  prnu.py                Photo-response non-uniformity mapping
  plotting.py            Diagnostic plots

poltools/                Imaging polarimetry simulator + Stokes pipeline (v0.1.0)
  README.md              Package overview and quick start
  mueller.py             Mueller forward model (HWP, PWI4 rotator, analyzer)
  simulate.py            2-D FITS forward model of the telescope chain
  _types.py              PolConfig, BeamGeometry, FilterConfig (per-band α-BBO)
  io.py                  Polarimetry FITS keywords + filter/HWP grouping
  photometry.py          Detection, o/e pairing, aperture photometry
  modulation.py          double-ratio / double-difference / LSQ → q,u
  errors.py              MAS debiasing, σ_θ (NK&C), residual σ_P
  stokes.py              Stokes assembly (p, θ)
  calibration.py         IP / efficiency / PA-zeropoint from standards
  pipeline.py            reduce_to_stokes (per-filter, end-to-end)
  plotting.py            Polarimetry diagnostic plots

scripts/                 Operator scripts (server start, plan preview, execute, QA gates)
night_plans/             palette.yaml + per-night brick plans (YYYYMMDD.yaml)
notebooks/               Jupyter notebooks
  templates/               Control-notebook palette — copy night.ipynb, then pull cells
    night.ipynb              Starter, provenance card, non-moving connect/status
    devices.ipynb            Explicit device operation (camera, EFW, HWP, PWI4 axes)
    capture.ipynb            Supervised diagnostic/calibration/modulation captures
    inspect.ipynb            Read-only frame, group, trend, and QA inspection
  observatory_control.ipynb  Interactive night control (superseded by templates/)
  lab_control.ipynb        Interactive lab-bench control (persistent kernel)
  polite.ipynb             Main analysis notebook
  reductions/              Dated reduction notebooks (reduction_YYYYMMDD.ipynb)

FITSDATA/                Raw FITS data organized by date (YYYYMMDD)
```

## Usage

Night sessions are declarative YAML brick plans under `night_plans/` (bricks
defined once in `palette.yaml`, laid under targets per night). Preview a plan,
then execute it (settings banner, mount / filter-wheel / HWP gates, cooler gate,
per-invocation output subdirs):

```bash
python scripts/plan_night.py    night_plans/example.yaml            # preview (no hardware)
python scripts/execute_night.py night_plans/example.yaml --run      # execute
```

### Detector operating point

The default is **Mode 5, gain 56, offset 20** on the QHY268M — QHY's reported
lowest-read-noise mode, at the lowest gain within it that still reaches the noise
floor, so the low noise costs as little dynamic range as possible. Dynamic range is a
first-order constraint on polarimetry, not a nicety: Cole (2010, SAS) defocuses the
PSF over ~6×6 px because the per-pixel well caps single-exposure signal, and his
acquisition loop restarts the *entire* waveplate sequence when a max pixel saturates.
**Mode 3, gain 0** is the alternative — highest full well of any mode, high but lower
dynamic range — for observations where well depth binds.

A plan inherits the default; set `camera:` in the plan only to deviate. A plan with no
`camera:` block, or with a key the loader does not recognize, logs a WARN and
continues on the defaults. Conversion gain (e⁻/ADU) and read noise (e⁻) are never plan
inputs — they are per-night reduction results, measured in a notebook cell.

Each device is connected only when the loaded plan actually uses it. The Pyxis
HWP rotator comes up when the plan steps it (`--hwp auto`, the default), and
`--hwp off` is refused on a plan that does — frames at an unknown plate angle are
unreducible. The mount comes up when the plan names a sky position (`--mount
auto`, the default), and `--mount off` is likewise refused on such a plan unless
`--unpointed` is also given, because nothing in the resulting FITS would record
that the telescope was never aimed at the object. Cal-only plans touch neither.
Shared safety gates live in `obs_utils/night_safety.py`.

> **Mount status.** POLITE's DEC drive (PWI4 axis 1) does not currently engage.
> The pointing path above is complete and wired but has not been exercised
> against a working mount — treat its first pointed night as commissioning.
> Nothing is special-cased to the fault: `night_safety.verify_mount` energizes
> both axes *with a deadline* and refuses to slew one that will not come up,
> naming axis1 when that is the one that failed. When the drive is repaired the
> same gate passes and pointed science runs with no code change.

While a plan runs, watch it from a control notebook. Copy a template from
`notebooks/templates/` and point `SESSION_DIR` at the night's directory:

```python
from obs_utils import live

live.session_table("FITSDATA/20260717")        # what has landed so far
live.watch("FITSDATA/20260717", every=10)      # block; report each new frame
live.frame_report(live.latest_frame(d))        # image + histogram + stats
live.qa_print(live.sequence_audit(d))          # end-of-night completeness
```

`obs_utils.live` is read-only — it only reads FITS off disk — and uses the same
sigma-clipping as `obs_utils/qa_lib.py`, so a number printed live is the number
the gate will judge.

Detector characterization:

```python
import caltools as ct

config = ct.sensor_config_from_header("frame.fits", gain=0.5)
bias = ct.master_bias(bias_paths)
rn_map, ts_map = ct.read_noise_map_from_paths(bias_paths)
ptc = ct.photon_transfer_curve(flat_groups, bias, config)
```

Polarimetry reduction:

```python
import poltools as pt

results = pt.reduce_to_stokes(frame_paths, cfg, o_positions=positions, method="lsq")

# A directory containing multiple targets/repeats/rotations must be reduced by
# provenance group; reduce_to_stokes fails closed on accidental mixing.
by_sequence = pt.reduce_pol_sequences(frame_paths, cfg, o_positions=positions)
```

Per-night reductions live in `notebooks/reductions/reduction_YYYYMMDD.ipynb`
(raw FITS are never edited; provisional products stay under `generated/`).

## Requirements

- Python 3.12+
- astropy, astroquery, numba
- alpyca (ASCOM Alpaca client)
- numpy, scipy, matplotlib
- PlaneWave PWI4 (mount control server)

## AI Disclosure

AI-assisted tools (Claude, Codex) were used during development of this repository for code architecture, implementation, and documentation.
