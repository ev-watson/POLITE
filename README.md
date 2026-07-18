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

Mount and rotator control use the PWI4 HTTP API; camera and filter wheel use ASCOM Alpaca. Night sessions are declarative Python scripts executed with logging, autoguiding, and dithering.

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
  gain.py                Photon transfer curve, full well, noise decomposition
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

scripts/                 Night session automation scripts
utils.py                 General-purpose astronomy utilities
notebooks/               Jupyter notebooks (see notebooks/README.md)
  lab_control.ipynb        Interactive lab-bench control (persistent kernel)
  observatory_control.ipynb  Interactive night control (startup->shutdown)
  polite.ipynb             Main analysis notebook
  reduction.ipynb          Image reduction pipeline notebook

FITSDATA/                Raw FITS data organized by date (YYYYMMDD)
datafiles/               Organized calibration and science frames
```

## Usage

Night sessions are defined as Python scripts in `scripts/`:

```python
from obs_utils import run_night_session, NightSessionConfig, TargetPlan, FramePlan

config = NightSessionConfig(
    targets=[
        TargetPlan(
            name="Jupiter",
            ra_j2000_hrs=..., dec_j2000_deg=...,
            frames=[FramePlan(frame_type="Light", exposure_s=12.0, count=12)]
        )
    ]
)
run_night_session(config)
```

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

Salvage first-light diagnostics (raw FITS are never edited; products remain
provisional under `generated/`):

```bash
python scripts/analyze_salvage_first_light.py FITSDATA/20260709
python scripts/reduce_salvage_drift_sequence.py FITSDATA/20260709
```

## Requirements

- Python 3.12+
- astropy, astroquery, numba
- alpyca (ASCOM Alpaca client)
- numpy, scipy, matplotlib
- PlaneWave PWI4 (mount control server)

## AI Disclosure

AI-assisted tools (Claude, Codex) were used during development of this repository for code architecture, implementation, and documentation.
