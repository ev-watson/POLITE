# Before Starting Observations

First light 2026-07-09. Observatory Windows PC (PWI4 :8220, ASCOM Alpaca :11111). Drive the night from night_plans/20260709.yaml into FITSDATA/20260709/. Dry-run preview: 270 frames, 3 QA gates, 1.01 h open-shutter.

## Already done (software / lab)

- [x] Night plan YAML, palette bricks, QA hooks, polite naming (dry-run: 270 frames, 3 QA gates)
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

```powershell
git pull
.\scripts\install_qhy_alpaca_deps.ps1
```

- [ ] Latest POLITE repo on the observatory PC (`git pull`)
- [ ] QHY Alpaca server deps installed (`scripts\install_qhy_alpaca_deps.ps1`)
- [ ] astroquery import works (automated asteroid ephemerides), or JPL Horizons manual fallback ready

## QHY268M SDK-direct camera (bypasses broken ASCOM QHY driver)

Camera on **:11112** (SDK-direct Alpaca). EFW + Pyxis stay on ASCOM Remote Server **:11111**.

Close EZCAP and all QHY apps before any step below.

```powershell
.\scripts\qhy268_bringup.ps1 -Step all
```

Or step by step:

| Step | Command |
|------|---------|
| 1 Scan | `.\scripts\scan_qhy_cameras.ps1` |
| 2 Server | `.\scripts\start_qhy_alpaca_server.ps1` (new window, leave running) |
| 3 Verify | `curl http://localhost:11112/management/apiversions` |
| 4 Camera smoke | `.\scripts\qhy268_bringup.ps1 -Step smoke-camera` |
| 5 Full smoke | `.\scripts\qhy268_bringup.ps1 -Step smoke-full` |
| 6 Night | `.\scripts\qhy268_bringup.ps1 -Step night` |

- [ ] Scan lists QHY268M (set `$env:QHYCCD_DLL` if needed)
- [ ] Camera server responds on :11112
- [ ] `qhy_alpaca_smoke_test.py` writes a FITS
- [ ] `observatory_smoke_test.py` passes (needs PWI4 + Remote Server for EFW/Pyxis)

## ASCOM / Alpaca bring-up (EFW + Pyxis on :11111)

- [ ] ASCOM Remote Server running; **EFW + Pyxis** on Alpaca :11111 (QHY camera is on :11112)
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
- [ ] Review: 270 frames, 3 QA gates, camera block gain=0 offset=30 cooler=-20 C, ends with "(dry-run; pass --run to execute)"
- [ ] Dry run does NOT connect, slew, move HWP, expose, or run QA on FITS

## After dry run - before roof

- [ ] FITSDATA/ exists and is writable on Windows
- [ ] Moon check for 2026-07-09: phase and altitude; keep targets >~30 deg from the Moon (block log records MOONSEP)
- [ ] Twilight flat exposure tuning ready (adjust exp between HWP angle sets; target 15-35 kADU, 30-60% FWC)
- [ ] Focus per filter (~20:40) - manual
- [ ] Do NOT change gain / readout mode / offset after bias QA passes - it invalidates masters and the CMOS error model

## Run the night - CORE dataset (must-get)

- [ ] python scripts/plan_night.py night_plans/20260709.yaml --run
- [ ] gamma Boo: confirm focus AND that BOTH Savart beams are visible/paired before science
- [ ] HD 154892 (unpolarized), then HD 154445 (polarized) - polV8 in V
- [ ] After HD 154445: first_light_qa reduces it (reference P=3.67%, PA=88.6 deg); reduce and confirm BEFORE rotating
- [ ] MANUAL rotator repeat: rotate PWI4 field rotator +45 deg, recenter HD 154445, run the polV8_3s repeat (POLSEQ HD154445_polV8_rot45)
- [ ] Coord-transform check: detector-frame q,u SHOULD change; sky-frame P,PA should MATCH the first run (both near ref). Mismatch = sign / WCS / beam-label / HWP-zero / rotator-convention error - note it, keep observing
- [ ] Matching darks (darks30 + darks_short) captured - core dataset is now self-contained
- [ ] End of night: sequence_audit runs automatically (HWP angle-set completeness per POLSEQ)

## Run the night - OPTIONAL (only if sky holds and time remains)

- [ ] Priority order: HD 161056, BD+32 3739, HD 204827 (+R), HD 212311, Melpomene, Juno, Hiltner 960
- [ ] Skip freely if high clouds come in (common after 01:00-02:00). A complete core dataset beats a half-finished long one
- [ ] Extra time -> repeat HD 154445 / HD 154892 or take more darks/flats rather than debugging the pipeline in the dark

## Minimum success (this is the bar for first light)

- [ ] Detector passes bias / RON sanity check
- [ ] V-band HWP flats acquired
- [ ] Both Savart beams automatically detected and stay paired through the HWP sequence
- [ ] HD 154892 reduces to low polarization
- [ ] HD 154445 shows clear 4-theta modulation
- [ ] lsq and double_ratio give consistent q, u
- [ ] Rotator +45 repeat of HD 154445 gives consistent sky-frame P, PA
- [ ] Pipeline produces q, u, P, PA and uncertainties with no manual intervention

## Stretch goals (NOT required tonight; defer to next run)

- [ ] Four polarized + four unpolarized standards; polarimetric efficiency; instrumental-polarization model
- [ ] Asteroid polarimetry (Melpomene, Juno); R-band calibration; dawn characterization + PTC ladder
- [ ] Publication-quality uncertainties

## First-light field card (tape to the console)

- [ ] Never stop collecting data because the reduction looks wrong - the sky is the scarce resource, not the pipeline
- [ ] Do NOT change gain / offset / mode / HWP-zero / focus / rotator calibration mid-night unless hardware is clearly broken
- [ ] Preserve raw FITS: never overwrite, rename, crop, or preprocess in place
- [ ] Log one-line breadcrumbs with UT: "22:43 possible wrong beam", "target drifted", "cloud"
- [ ] If one reduction fails, move on and keep collecting standards / darks / flats
- [ ] Before shutdown, verify only: all FITS exist, logs saved, calibration frames (bias/flats/darks) taken
