# poltools science reference

Peer-reviewed sources only (project rule). Flag anything else.

## Optical model

Chain: sky ? telescope ? field rotator ? HWP ? filter ? ?-BBO Savart ? detector.

Dual-beam polarimeter: each source ? ordinary + extraordinary images. HWP steps modulate O/E ratio with **4?**, encoding linear Stokes Q,U.

Modulation curve: `R(?) = (F_e ? F_o)/(F_e + F_o) = q cos4? + u sin4?`

## Reduction methods

| Method | Module | Reference | When |
|--------|--------|-----------|------|
| `lsq` | modulation.lsq_modulation | Magalhães et al. 1984; Ramírez et al. 2017 (SOLVEPOL) | Flat-fielded; ?4 HWP angles; returns cov, ?² |
| `double_ratio` | modulation.double_ratio | Tinbergen 1996; Masiero et al. 2007 | Bad/missing flats; needs {0°,22.5°,45°,67.5°} |
| `double_difference` | modulation.double_difference | cross-check | First-order flat cancellation |

## HWP modulation schemes (how the half-wave plate is stepped)

Source B (peer-reviewed). Standard practice for dual-beam imaging polarimeters:

- **Step-and-integrate, discrete angles.** Rotate the HWP to fixed positions in
  **22.5 deg increments** and expose at each. A 22.5 deg HWP rotation rotates the
  polarization plane by 45 deg, swapping the q<->u encoding and the o<->e beams
  ("beam swapping"), so a differential reduction cancels flat-field, transmission,
  and gain differences between the two beams (Masiero et al. 2007; Berdyugin,
  Piirola & Poutanen 2019).
- **Minimum set = 4 angles {0, 22.5, 45, 67.5} deg.** q,u from beam ratios
  `Q = F_e/F_o`: `q = (Q_0 - Q_45)/Q_m`, `u = (Q_22.5 - Q_67.5)/Q_m` with
  `Q_m = Q_0 + Q_22.5 + Q_45 + Q_67.5` (Berdyugin, Piirola & Poutanen 2019).
  Using only **2 angles is strongly discouraged** — it fails to cancel first-order
  instrumental effects (Patat & Taubenberger 2011, CAFOS).
- **Redundant sets (8 or 16 angles), 0..157.5 or 0..337.5 deg.** Extra positions
  cancel first-order instrumental polarization (non-ideal analyzer/transmission)
  and support a Fourier decomposition: an ideal dual-beam polarimeter has all power
  in the **n=4** harmonic; a non-zero **n=2** term flags HWP pleochroism, and a
  non-zero **a0** term flags a non-ideal (unequal) beam split that already cancels
  with >= 4 angles (Fendt et al. 1996; Patat & Taubenberger 2011; Gonzalez-Gaitan
  et al. 2020, FORS2). FORS2 uses 8 (0..157.5); CAFOS uses 16 (0..337.5).
- **Continuous-rotation HWP.** Spin the HWP continuously and demodulate the 4x
  signal (camera readout hardware-synced to the plate encoder); good for fast/
  variable sources and systematics control (MOPTOP: Shrestha et al. 2020).

**POLITE mapping:** the HWP is carried in a **2-inch Optec Pyxis rotator** and
stepped via **ASCOM Alpaca** on a Mac — the same hardware/software pattern as
LE2Pol (Wiersema et al. 2023), which used a 2-inch Optec Pyxis clear-aperture
rotator to rotate the wave plate, Alpaca drivers, and an INDIGO-served camera on
an Apple-Silicon MacBook. `HWPANG` is the Pyxis/HWP angle; `INSTROT` remains the
telescope field-rotator (PWI4) angle.

**Bench dry-run (no sky):** step the HWP over a full turn against (a) an
unpolarized source -> flat modulation, and (b) a polarized source (e.g. LED +
polaroid film) -> clean 4-theta modulation; recover p, theta and check the
Fourier content (LE2Pol lab test, Wiersema et al. 2023). Implemented in
`scripts/hwp_modulation_test.py`.

### HWP-scheme references
- Fendt, Beck, Lesch & Neininger 1996, A&A 308, 713 (Fourier modulation form)
- Patat & Romaniello 2006, PASP 118, 146 (FORS1 instrumental polarization)
- Patat & Taubenberger 2011, A&A 529, A57 (CAFOS; >=4 angles; n=2 pleochroism)
- Berdyugin, Piirola & Poutanen 2019 (optical polarimetry methods review)
- Gonzalez-Gaitan et al. 2020, A&A 634, A70 (FORS2 imaging polarimetry)
- Shrestha, Steele et al. 2020, MNRAS 494, 4676 (MOPTOP continuous-rotation HWP)
- Wiersema, Starling, Campagnolo, Thanki & McErlean 2023, RASTI 2, 106 (LE2Pol)

## Uncertainties

| Quantity | Function | Reference |
|----------|----------|-----------|
| ?_P (residual) | errors.residual_sigma_p | Ramírez et al. 2017 / SOLVEPOL |
| debias `mas` (default) | errors.debias_mas | Plaszczynski et al. 2014, eq. 20 |
| debias `wk` | errors.debias_wardle_kronberg | Wardle & Kronberg 1974 |
| ?_? high-SNR | errors.sigma_theta_highsnr | Serkowski relation (28.65° × ?_P/P) |
| ?_? low-SNR | errors.sigma_theta_nkc | Naghizadeh-Khouei & Clarke 1993 |
| PA switch threshold | stokes._SNR_GAUSSIAN_PA = 6 | Gaussian above; NKC below |

## Calibration (order matters)

1. Subtract instrumental (q?,u?) — unpolarized standards
2. Divide by modulation efficiency — polarized standards
3. Rotate by 2?? — PA zero-point

Ref: Masiero et al. 2007, PASP 119, 1126; DUSTPol commissioning notes in calibration.py.

## Forward model

Mueller matrices: Masiero et al. 2007; DUSTPol design. Retardance ? < 180° reduces modulation amplitude.

## Photometry noise

CCD model: shot + read + sky variance. Median-combine variance inflation ?/2 (Kenney & Keeping). Stockmans et al. eq. 16 cited in photometry.py.

## Citation block (README)

Masiero et al. 2007; Plaszczynski et al. 2014, MNRAS 439, 4048; Ramírez et al. 2017, MNRAS 472, 2793.
