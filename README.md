# POLITE

Observatory automation and detector characterization for a PlaneWave CDK20 + QHY268M (IMX571) imaging system, controlled via ASCOM Alpaca.

## Overview

POLITE provides end-to-end observatory control — from mount startup and slewing through automated imaging sequences and calibration frame acquisition — alongside `caltools`, a detector characterization library for bias, dark, flat, noise, gain, linearity, and PRNU analysis.

The system interfaces with PlaneWave's PWI4 HTTP API for mount control and the ASCOM Alpaca REST protocol for camera and filter wheel operations. Night sessions are defined declaratively as target/frame plans and executed autonomously with logging, autoguiding, and dithering support.

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
  fits_routine.py        FITS file handling
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

caltools/                Detector characterization library (v0.1.0)
  io.py                  FITS I/O, cube loading, header parsing
  stacking.py            Master bias, dark, flat generation
  stats.py               Welford accumulator, MAD sigma, outlier masking
  noise.py               Read noise maps, DSNU, FPN, RTN detection
  dark.py                Dark current vs exposure/temperature, warm pixels
  gain.py                Photon transfer curve, full well, noise decomposition
  linearity.py           Linearity testing and error characterization
  prnu.py                Photo-response non-uniformity mapping
  plotting.py            Diagnostic plots

scripts/                 Night session automation scripts
utils.py                 General-purpose astronomy utilities
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

Detector characterization uses `caltools`:

```python
import caltools as ct
config = ct.sensor_config_from_header("frame.fit", gain=0.5)
bias = ct.master_bias(bias_paths)
rn_map, ts_map = ct.read_noise_map(bias_cube)
ptc = ct.photon_transfer_curve(flat_pairs, bias)
```

## Requirements

- Python 3.12+
- astropy, astroquery, numba
- alpyca (ASCOM Alpaca client)
- numpy, scipy, matplotlib
- PlaneWave PWI4 (mount control server)

## AI Disclosure

AI-assisted tools (Claude, Codex) were used during development of this repository for code architecture, implementation, and documentation.
