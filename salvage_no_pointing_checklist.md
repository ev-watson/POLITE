# Salvage Night - No Telescope Pointing

DEC drive is dead - engaging DEC auto-disconnects the mount, so there is NO slewing, NO pointing, and NO manual DEC nudge. RA can only be engaged to hold/limit RA drift, not to slew. You cannot choose a target: you observe whatever pair is already in the field. Shift the night from SKY commissioning to INSTRUMENT commissioning. Salvage goal is qualitative only - is HWP modulation present, can the beam pair be tracked frame-by-frame, does the detector-frame Stokes vector transform correctly when the whole polarimeter is rotated. NOT calibrated P or PA (that needs a working mount). Focus working means the optics, camera, and Savart splitter are almost certainly fine - that is the encouraging part.

## Document the failure first (before it changes or is forgotten)

- [ ] DEC motor: completely dead, or intermittent? Note exact symptom
- [ ] Confirm behavior: does engaging DEC still auto-disconnect the mount every time?
- [ ] Manual DEC slew: no response / disconnects / other - record what happens
- [ ] RA: does engaging RA hold the field (cancel sidereal drift) or only lock the axis? Note the residual drift you actually see
- [ ] PWI4 error messages: copy verbatim + screenshot
- [ ] Time (UTC), camera temp, and sky conditions at the time of failure

## Preserve what you already have

- [ ] Keep every focused doubled-star frame already on disk
- [ ] Do NOT overwrite, rename, crop, or reprocess existing frames in place
- [ ] Copy the existing frames to a second location NOW, before collecting more

## Camera calibration (needs no pointing - do this regardless of the mount)

- [ ] 25-50 bias frames at the final camera mode / gain / offset / temp (Mode 0, gain 0, offset 30, -20 C)
- [ ] Matched darks for EVERY exposure time used tonight - include the very short drift exposures
- [ ] V-band flats through the full optical train at HWP 0, 22.5, 45, 67.5 deg; at least 10 frames per angle; do NOT change camera settings
- [ ] Flats source: twilight sky at the current fixed pointing, or a dome / flat-panel screen if reachable (sky flats do not need tracking; drift actually averages out stars)

## Instrument commissioning (highest salvage value)

- [ ] Confirm BOTH Savart beams are present for the pair(s) currently in the field
- [ ] Measure beam separation and orientation (PA) - repeat at a few field positions if more than one pair is available, to map any variation across the detector
- [ ] Confirm HWP rotation works: run scripts/hwp_modulation_test.py (or step the HWP through the full sequence and confirm the flux ratio modulates)
- [ ] Plate scale from a known star pair, if short exposures are clean enough to centroid

## Drift-through polarimetry (bright isolated pair now in the field)

- [ ] Pick the brightest isolated pair currently on the detector - you cannot point, so use what is there. Record its approximate position and declination
- [ ] Know your drift rate: sky moves ~15 arcsec x cos(dec) per second = up to ~67 px/s near the equator at 0.224 arcsec/px. If RA holds sidereal, drift is far less
- [ ] Set exposures short enough to keep images compact (aim < ~2 px trail). Near the equator with no RA hold that is tens of ms; lengthen if RA is holding the field
- [ ] Run a complete polV8 sequence as fast as possible while the pair stays in a clean region
- [ ] Repeat the same polV8 once or twice while the pair is still clean - redundancy for later centroid tracking and photometry
- [ ] Rotate the WHOLE polarimeter +45 deg (field rotator) and repeat the same sequence on the same pair, provided both beams stay on the detector
- [ ] Keep the whole-instrument-rotation dataset and the HWP-only dataset SEPARATE - they test different coordinate transforms
- [ ] Do NOT attempt live analysis - just collect and log

## Log every block (paper or file)

- [ ] UTC; exposure time; filter; HWP angle; whole-polarimeter rotator angle
- [ ] Camera temperature; gain; readout mode; offset
- [ ] Approximate source position on the detector
- [ ] Whether the source drifted near an edge, bad pixels, or another stellar pair

## Do NOT

- [ ] Do NOT force long tracked exposures - a dead DEC gives elongated stars and frustration
- [ ] Do NOT change camera gain / mode / offset / HWP-zero mid-session (invalidates the calibration)
- [ ] Do NOT attempt to point, slew, or engage DEC (auto-disconnects the mount)

## Before shutdown - verify

- [ ] All FITS files exist and filenames are unique
- [ ] Every HWP angle is present in each sequence
- [ ] Rotator angle is recorded for every block
- [ ] Data AND logs AND the failure documentation are copied to a second location

## Salvage dataset payoff (what this buys you)

- [ ] One or more drifting polV8 sequences at the original polarimeter angle
- [ ] The same sequence after a +45 deg whole-polarimeter rotation
- [ ] Biases, matched darks, angle-matched V flats
- [ ] Beam-geometry and HWP-modulation measurements
- [ ] Complete metadata and the documented DEC failure
- [ ] Enables later tests: modulation present? beam pair trackable frame-by-frame? detector-frame Stokes transforms correctly under instrument rotation?
