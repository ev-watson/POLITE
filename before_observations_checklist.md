# Before Starting Observations

First light 2026-07-09. Observatory Windows PC (PWI4 :8220, ASCOM Alpaca :11111). Drive the night from night_plans/20260709.yaml into FITSDATA/20260709/. Dry-run preview: 278 frames, 3 QA gates, 1.08 h open-shutter.

## Already done (software / lab)

- [x] Night plan YAML, palette bricks, QA hooks, polite naming (dry-run: 278 frames, 3 QA gates)
- [x] FITS provenance wired: EGAIN, GAIN, READMODE, OFFSET, SET-TEMP, INSTROT, HWPANG
- [x] pol_config.yaml sidecar + block_manifest.jsonl written by the night runner
- [x] HWP hardware test: 8 frames in FITSDATA/hwp_test/; no pixels pinned at 0
- [x] Reduction stack: poltools lsq (default) + double_ratio cross-check available

## Python environment (observatory PC)

Build a clean env and install every package the capture + reduction pipeline
needs. Run from the repo root. Windows PowerShell:

```powershell
py -3.13 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -U pip
pip install numpy scipy astropy photutils matplotlib pyyaml alpyca requests astroquery
pip install -e .\caltools -e .\poltools
```

macOS / Linux test box (zsh):

```zsh
python3 -m venv .venv && source .venv/bin/activate && python -m pip install -U pip \
  && pip install numpy scipy astropy photutils matplotlib pyyaml alpyca requests astroquery \
  && pip install -e ./caltools -e ./poltools
```

Sanity check the import surface:

```zsh
python -c "import numpy, scipy, astropy, photutils, matplotlib, yaml, alpaca, requests, astroquery, poltools, caltools; print('env ok')"
```

- [ ] venv created with Python 3.13 and activated
- [ ] pip install of science + device packages (numpy scipy astropy photutils matplotlib pyyaml alpyca requests astroquery) succeeds
- [ ] caltools and poltools installed editable (pip install -e)
- [ ] Import check prints "env ok" (alpaca = alpyca ASCOM client; astroquery = Horizons)
- [ ] obs_utils / alpyca_tools import in place from repo root - run all scripts from repo root, no install needed

## Drivers and device software (observatory PC)

ASCOM Platform is already installed. Confirm the rest before Alpaca bring-up.

- [ ] ASCOM Platform 6.6+ present - confirm version
- [ ] ASCOM Remote Server installed and exposing devices on Alpaca :11111 (separate from the Platform)
- [ ] QHY camera driver (QHYCCD All-In-One / SDK) + ASCOM QHY driver
- [ ] ZWO EFW filter-wheel driver + ASCOM driver
- [ ] Optec Pyxis rotator ASCOM driver (drives the HWP rotator)
- [ ] PWI4 (PlaneWave) installed; listening on :8220; pointing model file present
- [ ] USB-serial driver (FTDI / Prolific) for the Pyxis RJ12-to-USB link

## Sync code to Windows

- [ ] Latest POLITE repo on the observatory PC (first-light code may be uncommitted on the dev box - commit/push or copy over)
- [ ] astroquery import works (automated asteroid ephemerides), or JPL Horizons manual fallback ready

## ASCOM / Alpaca bring-up

- [ ] ASCOM Remote Server running; camera / EFW / rotator added; Alpaca reachable on :11111
- [ ] PWI4 GUI running (:8220); mount homed; pointing model loaded; rotator + focuser connected
- [ ] Pyxis HWP reachable as an Alpaca rotator; EFW initialized (V and R slots correct)
- [ ] user_config.py device indices match the Remote Server (camera / wheel / rotator)
- [ ] Observatory PC clock is NTP-synced (DATE-OBS and asteroid ephemerides depend on it; timing cards read the clock)
- [ ] python scripts/observatory_smoke_test.py - home, slew, HWP move, one FITS

## Detector settings (lock for the whole night)

- [ ] Gain 0 for the entire night - NOT the gain 30 used on the HWP bench test. EGAIN=1.0 e-/ADU and RON=3.5 e- are only valid at Mode 0, gain 0
- [ ] Readout Mode 0, offset 30 (matches plan camera block and 2026-03 bench)
- [ ] Cooler to -20 C; wait for stabilization (CCD-TEMP within ~1 C of SET-TEMP)
- [ ] 5 bias frames: min ADU > 0 (no pixels pinned at 0), sensible pedestal from offset 30
- [ ] Brightest standards do not saturate - keep peak below ~60% FWC (~30 kADU); short exposures already set for gamma Boo and HD 154445
- [ ] HWP backlash / settle values set from bench

## Dry run (no hardware)

- [ ] python scripts/plan_night.py night_plans/20260709.yaml
- [ ] Review: 278 frames, 3 QA gates, camera block gain=0 offset=30 cooler=-20 C, ends with "(dry-run; pass --run to execute)"
- [ ] Dry run does NOT connect, slew, move HWP, expose, or run QA on FITS

## After dry run - before roof

- [ ] FITSDATA/ exists and is writable on Windows
- [ ] Moon check for 2026-07-09: phase and altitude; keep targets >~30 deg from the Moon (block log records MOONSEP)
- [ ] Twilight flat exposure tuning ready (adjust exp between HWP angle sets; target 15-35 kADU, 30-60% FWC)
- [ ] Focus per filter (~20:40) - manual
- [ ] Do NOT change gain / readout mode / offset after bias QA passes - it invalidates masters and the CMOS error model

## Run the night

- [ ] python scripts/plan_night.py night_plans/20260709.yaml --run
- [ ] After HD 154445: check first_light_qa (reference P=3.67%, PA=88.6 deg); do NOT continue to HD 161056 if PA fails unexplained
- [ ] End of night: sequence_audit runs automatically (HWP angle-set completeness); confirm darks captured after science
