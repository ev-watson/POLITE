# Lab Trial Checklist — First Bench Test (no telescope)

**Goal:** On the lab **Mac (Apple Silicon)**, verify we can (1) send rotator commands to the
**Optec Pyxis 2"**, (2) swap filters on the **ZWO EFW**, (3) take a single exposure
(dark/flat/bias — content doesn't matter) with the **QHY268M**, and (4) step the Pyxis
through a **half-wave-plate (HWP) modulation sequence** and confirm a sensible modulation
curve (~4θ harmonic).

> **What the Pyxis is for.** In POLITE's dual-beam polarimeter the **Optec Pyxis 2"
> (2-inch clear aperture) carries the rotating half-wave plate** — it is the polarization
> *modulator*, not the telescope field de-rotator (that is the PlaneWave/PWI4 mount rotator,
> `INSTROT`). This is exactly the configuration of **LE2Pol** (Wiersema et al. 2023, RASTI
> 2, 106), a near-identical small-telescope dual-beam Savart polarimeter that rotates its
> HWP with a 2-inch Optec Pyxis via **ASCOM Alpaca on an Apple-Silicon MacBook** and reads
> the camera through **INDIGO** — i.e. the exact stack in this checklist. So the "rotator
> test" (goal 1) *is* the HWP-drive test, and goal 4 exercises the full modulation loop.
> (If in your build the Pyxis instead de-rotates the field and a different stage carries the
> HWP, tell me and I'll re-map the scripts.)

**Chosen route (lab Mac): INDIGO for camera + EFW, native serial for the HWP rotator.**
INDIGO runs natively on macOS; its `indigo_agent_alpaca` bridge exposes the **QHY268M**
and **ZWO EFW** as standard ASCOM Alpaca devices, and POLITE's existing `alpyca` code
drives them unchanged. The **Optec Pyxis 2" Gen3** HWP rotator, however, does **not** go
through INDIGO: its Gen3 framed protocol is incompatible with INDIGO's legacy
`indigo_rotator_optec` handshake (ESTABLISHED — verified against the driver source; it
opens the port then times out). POLITE therefore drives the Pyxis directly over its FTDI
serial link (`obs_utils/pyxis_gen3.py`). On the **observatory** Windows box the Pyxis is
instead an Alpaca rotator behind Optec's Universal ASCOM driver — that Alpaca path stays
valid there (use `--rotator N`), so the dome config is a superset of this one.

```
Pyxis 2" Gen3 (HWP)   --FTDI serial->  obs_utils/pyxis_gen3.py (native) ---------------------------> POLITE
ZWO EFW               --USB-------->  INDIGO (indigo_wheel_asi)       --+--> indigo_agent_alpaca --HTTP--> alpyca (POLITE)
QHY268M               --USB-------->  INDIGO (indigo_ccd_qhy2)        --/        (:11111)

Optical order:  ... field rotator (PWI4) -> [HWP in Pyxis] -> EFW filter -> alpha-BBO Savart -> QHY268M
```

> Why not the vendor drivers? Optec/ZWO/QHY ASCOM drivers are **Windows-only**. INDIGO is
> the macOS-native equivalent and gives us one Alpaca endpoint for all three.

---

## 0. Key facts / gotchas (read first)

- **Apple Silicon:** Use a **current** INDIGO build. Older INDIGO Server docs tag QHY/ZWO
  drivers "Intel only," but recent framework releases (>= 2.0.232 / INDIGO A1 v5) ship
  **native arm64** SDKs for `indigo_ccd_qhy2`, `indigo_wheel_asi`, and `indigo_ccd_asi`.
  If you land on an old Intel-only build it will still run under Rosetta 2, but prefer native.
- **QHY on macOS is officially "experimental."** Expect possible flakiness; run the QHY
  driver isolated (`indigo_server -i ...`) so a driver crash doesn't take down the server.
- **Python package name trap:** the ASCOM client is **`alpyca`** (PyPI) which imports as
  `alpaca`. The Mac currently has **`alpaca-py`** (a *stock-trading* library) installed,
  which squats on the same `alpaca` import name. **Use a fresh virtualenv** for this trial.
- **Pyxis 2" is RS-232 only** (no native USB) — you need a **USB-RS-232 adapter**.

---

## 1. Hardware to bring / connect

- [ ] Optec Pyxis 2" rotator + its **12 V DC power supply**
- [ ] **USB-to-RS-232 adapter** (FTDI chipset recommended) + the DB9 serial cable for the Pyxis
- [ ] ZWO EFW + USB cable (USB-HID, **no driver needed**)
- [ ] QHY268M + USB 3.0 cable + **12 V DC power supply** (needed for the TEC/cooler)
- [ ] Powered USB hub (optional but recommended for the QHY)

**Physical bring-up order (matters for the Pyxis):**
1. [ ] Plug in Pyxis power **first** and wait — it auto-homes on power-up and will **ignore
   serial** until homing completes (LED stops moving).
2. [ ] Connect the USB-RS-232 adapter to the Mac, then the Pyxis serial cable.
3. [ ] Connect EFW and QHY268M USB; power the QHY.

---

## 2. Software — needed vs already installed

| Component | Status on this Mac | Action|
|---|---|---|
| Homebrew (Apple Silicon) | ? `/opt/homebrew` | yes | -|
| `pyserial` | yes: 3.5 | - (only needed for the direct-serial fallback)|
| `astropy` | yes: 7.2.0 | -|
| `alpyca` (ASCOM Alpaca client) | no: not installed | `pip install alpyca` **in a fresh venv**|
| `alpaca-py` (trading lib) | warn: installed in base Python | Keep out of the trial venv (name clash)|
| INDIGO server (macOS) | no: not installed | Install (see below)|
| QHYCCD arm64 SDK | - | Install if INDIGO's bundled QHY SDK isn't sufficient|
| USB-RS-232 driver | check | Install FTDI/Prolific VCP driver if adapter isn't recognized|
### 2a. Create the trial Python environment
- [ ] `python3 -m venv ~/.venvs/polite-lab && source ~/.venvs/polite-lab/bin/activate`
- [ ] `pip install alpyca pyserial astropy numpy`
- [ ] Sanity: `python -c "from alpaca.rotator import Rotator; from alpaca.camera import Camera; from alpaca.filterwheel import FilterWheel; print('alpyca OK')"`
  - If this imports the trading library instead, you're in the wrong environment.

### 2b. Install INDIGO on the Mac
Pick one:
- [ ] **INDIGO A1** (App Store, Apple-Silicon native, easiest — bundles the server), **or**
- [ ] **INDIGO Server for macOS** app from the INDIGO downloads page, **or**
- [ ] Build/`brew` the CLI `indigo_server` if you prefer terminal-only.

### 2c. Enable the INDIGO drivers we need
In the INDIGO app: **Preferences > INDIGO Drivers**, enable (scroll to find):
- [ ] **Optec Pyxis Rotator** (`indigo_rotator_optec`)
- [ ] **ZWO ASI wheel** (`indigo_wheel_asi`)
- [ ] **QHY CCD** (`indigo_ccd_qhy2`) — run isolated if the option exists
- [ ] **ASCOM Alpaca Agent** (`indigo_agent_alpaca`)

CLI equivalent:
```zsh
indigo_server -i indigo_ccd_qhy2 indigo_wheel_asi indigo_rotator_optec indigo_agent_alpaca
```

---

## 3. Connect devices in INDIGO

Open the INDIGO control panel (built-in web UI at `http://localhost:7624` or the app's panel):
- [ ] **Pyxis:** do **NOT** connect it in INDIGO — the legacy `indigo_rotator_optec`
      driver cannot speak the Gen3 protocol and will hold the COM port. Leave the FTDI
      port free for the native driver and verify the rotator separately (§5 /
      `scripts/pyxis_gen3_test.py`), using the macOS `/dev/cu.*` node (not `/dev/tty.*`).
- [ ] **EFW:** Connect; confirm slot count and (if configured) filter names.
- [ ] **QHY268M:** Connect; confirm it reads sensor size and temperature.
- [ ] **Alpaca agent:** confirm each device appears in `AGENT_ALPACA_DEVICES` with an
      assigned Alpaca **device number** (note them — Rotator #, FilterWheel #, Camera #).

Verify the Alpaca endpoint is live (default port **11111**):
- [ ] `curl "http://localhost:11111/management/v1/configureddevices"` lists Camera and
      FilterWheel (no Rotator on the lab Mac — the Gen3 HWP is on native serial, not INDIGO).

---

## 4. Run the tests (via POLITE `alpyca` code)

Use the helper script (added for this trial): `scripts/lab_trial_indigo.py`.
It runs all three checks in one pass — EFW + camera over the Alpaca endpoint, the
Pyxis Gen3 over native serial (`--pyxis-serial`) — saving one FITS. The rotator
check is **read-only by default** (ping + status); homing/moving are opt-in.

- [ ] `python scripts/lab_trial_indigo.py --host localhost:11111 \`
      `--pyxis-serial --filterwheel 0 --camera 0 \`
      `--filter-slot 1 --exposure 1.0 --dark --out ./lab_trial.fits`
- [ ] To also exercise HWP motion (homes first, then moves to 90°), add:
      `--pyxis-home --pyxis-move --rotate-to 90`
- [ ] (Observatory Windows host only) drive the Pyxis as an Alpaca rotator with
      `--rotator N` instead of `--pyxis-serial`.

What it does, mapped to the three goals:

1. **Rotator commands** — connect Alpaca `Rotator`, read `Position`, `MoveAbsolute(90)`,
   poll `IsMoving` until settled, read `Position` again.
2. **Swap filters** — connect Alpaca `FilterWheel`, read `Names`/current `Position`,
   move to the requested slot, confirm it lands.
3. **Take a photo** — connect Alpaca `Camera`, set the project operating point
   (**Mode 5, gain 56, offset 20**), take one dark, download, and write FITS with
   `alpyca_tools.fits_writer` (reuses the observatory header code).

**Pass criteria**
- [ ] Rotator reaches ~90° (within its step tolerance) and reports a stable `Position`.
- [ ] Filter wheel reports the target slot after the move.
- [ ] A valid FITS lands on disk and opens in a viewer / `astropy.io.fits`.

---

## 4b. HWP modulation test (goal 4)

**How HWPs are modulated (peer-reviewed practice, Source B).** Dual-beam imaging
polarimeters step the HWP in **22.5° increments** and expose at each position. A 22.5°
HWP rotation rotates the polarization plane by 45°, which swaps the `q/u` encoding and the
ordinary/extraordinary beams — "**beam swapping**" — so a differential reduction cancels
flat-field, transmission, and gain differences between the two beams.

- **Minimum sequence = 4 angles `{0, 22.5, 45, 67.5}°`.** Using only 2 angles is *strongly
  discouraged* (fails to cancel first-order instrumental effects — Patat & Taubenberger
  2011). The script refuses `< 4` angles.
- **Redundant sequences (8 angles: 0°–157.5°, 16 angles: 0°–337.5°)** further cancel instrumental
  polarization and let you Fourier-check the modulation: an **ideal dual-beam signal sits
  entirely in the n=4 (4?) harmonic**; non-zero `a0` (unequal split) or `n=2` (HWP
  pleochroism) flag non-ideal optics (Fendt et al. 1996; González-Gaitán et al. 2020, FORS2).
- **Continuous rotation** (spin + demodulate the 4θ signal) is the alternative used by fast
  time-domain instruments (MOPTOP, Shrestha et al. 2020) — not needed for this bench test.

References: Masiero et al. 2007 (DBIP); Berdyugin, Piirola & Poutanen 2019 (review);
Wiersema et al. 2023 (LE2Pol — same Mac+Alpaca+Pyxis+Savart stack).

### 4b-i. Dry run in simulation first (no hardware)
Validates the modulation logic end-to-end using `poltools` (run under the project env):
- [ ] `/Users/blu3/miniforge3/envs/POLITE/bin/python scripts/hwp_modulation_test.py --angles 4`
- [ ] `scripts/hwp_modulation_test.py --angles 16`  (adds the Fourier n=2/n=4 check)

Expect: the unpolarized source recovers `p ~ 0`; the polarized source recovers the injected
`p`/`?`; at 16 angles the signal power concentrates in **n=4** with `a0`, `n=2 ~ 0`.

### 4b-ii. Bench dress-rehearsal on the real kit
Mirrors the LE2Pol lab test: put a small light source downstream (optional **LED**, and an
**LED + polaroid film** for a known-polarized source), then step the HWP (Pyxis) through the
sequence, taking one QHY frame per angle. Each FITS is written with the `HWPANG` keyword.
- [ ] `python scripts/hwp_modulation_test.py --mode hardware --host localhost:11111 \`
      `--rotator 0 --camera 0 --filterwheel 0 --filter-slot 1 \`
      `--angles 8 --exposure 1.0 --object-name HWPTEST --out-dir ./FITSDATA/hwp_test`

> **Backend note.** `hwp_modulation_test.py --mode hardware` uses the **Alpaca**
> rotator (`--rotator N`) — the observatory Windows path. On the **lab Mac** the Gen3
> HWP is on native serial, so `--rotator 0` (INDIGO Alpaca) will not drive it; use
> `scripts/pyxis_gen3_test.py --home --move <deg>` for single moves, or the combined
> `lab_trial_indigo.py --pyxis-serial` above. A serial-backed multi-angle capture in
> `hwp_modulation_test` is a pending follow-up.

**Pass criteria**
- [ ] Pyxis reaches each commanded angle (0, 22.5°, …) and reports a stable position
      (~1.4 s per 22.5° step is normal for a Pyxis 2").
- [ ] One FITS per angle is written, each carrying the correct `HWPANG`.
- [ ] (If a polarized source is used) reducing the frames later shows a clean 4? modulation;
      an unpolarized source shows a flat curve.

---

## 5. Native-serial rotator (Gen3) + camera/wheel fallback

**The Pyxis 2" Gen3 rotator is driven over native serial, not INDIGO** — verify it with
`scripts/pyxis_gen3_test.py` (read-only ping+status by default; add `--home --move 90` to
exercise), using the macOS `/dev/cu.*` node. The Gen3 controller speaks a *framed* protocol
(`<R1iiGETSTA>` -> `!ii ... END`); the **legacy** `CCLINK`/`CPA###` commands listed below do
**NOT** work on it — that legacy handshake is exactly what INDIGO's `indigo_rotator_optec`
attempts, and why it times out on Gen3. They are kept only as a reference for pre-Gen3 Pyxis
units. For the QHY/EFW, if INDIGO's path misbehaves the vendor arm64 SDKs are a last resort:
- [ ] `screen /dev/tty.usbserial-XXXX 19200` (or a pyserial one-liner)
- [ ] Send `CCLINK` -> expect `!` (confirms serial loop / homing done)
- [ ] `CGETPA` -> returns current angle `nnn`
- [ ] `CPA090` -> rotates to 90° (`!` per step, `F` when complete)

For camera/wheel, the vendor **arm64 SDKs** (QHYCCD SDK, ZWO EFW SDK) can be driven from
Python via `ctypes` as a last resort. Flag this to the team before going down that path —
it bypasses the Alpaca layer and needs extra glue code.

---

## 6. Teardown
- [ ] Warm the QHY TEC back toward ambient before powering off (avoid thermal shock).
- [ ] Disconnect devices in INDIGO, quit the server.
- [ ] Record: which INDIGO version/build, driver versions, serial port, and any errors,
      so the observatory (Windows) config can be cross-checked.

---

### Reference — Pyxis 2" serial protocol (19200 8N1)
| Command | Reply | Purpose|
|---|---|---|
| `CCLINK` | `!` | Confirm serial loop / homing complete (required before other cmds)|
| `CHOMES` | `!` per step, `F` done | Home to PA 0|
| `CGETPA` | `nnn` | Read current position angle (000–359)|
| `CPA###` | `!` per step, `F` done | Move to angle (zero-padded, e.g. `CPA010`)|
| `CDn` | none | Set default rotation direction (0/1)|
| `CSLEEP` / `CWAKUP` | — / `!` | Power stepper down / up|