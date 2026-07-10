# POLITE scripts

Observatory-control and bench scripts. The automation "brain" is the scripted
POLITE stack (`alpyca` + PWI4); INDIGO hosts the server + `indigo_agent_alpaca`
bridge and provides its Control Panel / simulators / FITS Preview for bring-up
and quick-look — it does **not** replace these scripts.

Device split:
- **INDIGO → Alpaca** (`:11111`): QHY268M camera, ZWO EFW filter wheel.
- **Direct serial**: Optec Pyxis 2" **Gen3** HWP rotator via
  `obs_utils/pyxis_gen3.py`. Its Gen3 controller protocol is incompatible with
  INDIGO's legacy `indigo_rotator_optec`, so POLITE drives it natively over the
  FTDI link (bypassing INDIGO/Alpaca for the rotator only). Bench-test with
  `pyxis_gen3_test.py` (below) before wiring into night plans.
- **PWI4** (`:8220`): PlaneWave mount + field rotator (`INSTROT`).

## Observatory Windows — QHY268M SDK-direct bring-up

When the QHY ASCOM driver fails but EZCAP works, the camera uses a separate
SDK-direct Alpaca server on **:11112** (`third_party/alpaca-qhyccd-camera`).
EFW + Pyxis stay on ASCOM Remote Server **:11111**.

```powershell
git pull
.\scripts\install_qhy_alpaca_deps.ps1    # once
.\scripts\qhy268_bringup.ps1 -Step all
```

| Script | Step |
|--------|------|
| `scan_qhy_cameras.ps1` | SDK scan (close EZCAP first) |
| `start_qhy_alpaca_server.ps1` | Camera Alpaca server :11112 |
| `qhy_alpaca_smoke_test.py` | Camera-only FITS capture |
| `observatory_smoke_test.py` | Mount + HWP + EFW + camera |
| `qhy268_bringup.ps1` | Guided wrapper for all steps |

See `before_observations_checklist.md` for the full first-light checklist.

## Lab bring-up / testing pipeline

Exercises the HWP rotator + EFW + camera in one run — EFW + camera through the
INDIGO→Alpaca path, the Pyxis **Gen3** HWP over native serial (its Gen3 protocol
can't go through INDIGO). See the full procedure in
[`docs/lab_trial_checklist.md`](../docs/lab_trial_checklist.md).

1. **`lab_trial_indigo.py`** — full bench smoke test of all three devices.
   Each check is isolated so a partial lab setup still runs. The rotator check
   is **read-only by default** (ping + status); homing/moving are opt-in.

   ```zsh
   # Lab Mac: Pyxis Gen3 over serial (read-only), EFW + camera over INDIGO/Alpaca.
   .../python scripts/lab_trial_indigo.py --host localhost:11111 \
       --pyxis-serial --filterwheel 0 --camera 0 \
       --filter-slot 1 --exposure 1.0 --dark --out ./lab_trial.fits

   # ...add HWP motion (homes, then moves to 90°):
   .../python scripts/lab_trial_indigo.py --host localhost:11111 \
       --pyxis-serial --pyxis-home --pyxis-move --rotate-to 90 \
       --filterwheel 0 --camera 0 --filter-slot 1 --exposure 1.0 --dark
   ```

   On the **observatory** Windows stack the Pyxis is an Alpaca rotator behind
   Optec's Universal ASCOM driver — use `--rotator N` there instead of
   `--pyxis-serial`.

2. **`hwp_modulation_test.py`** — half-wave-plate modulation test.
   - `--mode sim` (default, **no hardware**): validates the modulation logic
     end-to-end with `poltools` (unpolarized → p≈0; polarized → recovers p/θ;
     ≥8 angles adds the Fourier n=2/n=4 check). Refuses `<4` HWP angles.

     ```zsh
     .../python scripts/hwp_modulation_test.py --angles 4
     .../python scripts/hwp_modulation_test.py --angles 16
     ```
   - `--mode hardware`: bench dress-rehearsal — steps the Pyxis through the
     angle sequence and captures one QHY frame per angle (FITS carries
     `HWPANG`).

     ```zsh
     .../python scripts/hwp_modulation_test.py --mode hardware --host localhost:11111 \
         --rotator 0 --camera 0 --filterwheel 0 --filter-slot 1 \
         --angles 8 --exposure 1.0 --object-name HWPTEST --out-dir ./FITSDATA/hwp_test
     ```

## Observatory smoke test (Windows, production)

**`observatory_smoke_test.py`** — the fastest end-to-end "is the whole chain
alive?" check, run **on the observatory Windows PC**. Homes the mount, slews to a
random field near zenith, rotates the HWP, picks an arbitrary filter, and writes
one short light frame — minimal time, minimal checks (no plate-solve / focus /
guiding). Mount + field rotator are PWI4 (`:8220`); camera + EFW + HWP-Pyxis are
Alpaca devices on the Windows ASCOM Remote Server (`:11111`) — both `localhost`
since the script runs on that PC. The HWP here is the **Alpaca** rotator behind
Optec's Universal ASCOM driver (not the lab serial path).

```zsh
# HWP Pyxis is Alpaca Rotator #0 on the Remote Server (override with --rotator-index):
.../python scripts/observatory_smoke_test.py --rotator-index 0
# already-homed re-run, longer exposure, explicit output:
.../python scripts/observatory_smoke_test.py --skip-home --exposure 3 --out D:/tmp/smoke.fits
```

## Native Pyxis Gen3 rotator (direct serial)

The HWP-carrying Optec Pyxis 2" **Gen3** rotator is driven over its FTDI serial
link by `obs_utils/pyxis_gen3.py` (`PyxisGen3` / `connect_pyxis_gen3`), *not*
INDIGO — the Gen3 framed protocol (`<R1 ii CCCCCC …>` → `!ii … END`) is
incompatible with INDIGO's legacy `indigo_rotator_optec` handshake. Serial port
+ baud live in `obs_utils.user_config.PYXIS_CONFIG`.

**`pyxis_gen3_test.py`** — bench verification of the driver. Run it once the
Optec cable is plugged in (`ls /dev/cu.usbserial*`) and the rotator has 12 V
power, to confirm the protocol/baud before wiring the HWP into night plans.
The default run is **read-only** (ping + status); movement is opt-in.

```zsh
# read-only: connect, GETDNN ping, GETSTA status (probes baud automatically)
.../python scripts/pyxis_gen3_test.py --port /dev/cu.usbserial-OP7XD6WD
# home, then move to 90° and report the settled PA
.../python scripts/pyxis_gen3_test.py --home --move 90
# add -v to see raw TX/RX frames; --no-autodetect to lock the baud
```

The driver exposes an ASCOM-`Rotator`-compatible shim (`MoveAbsolute` /
`IsMoving` / `Position` / `Connected`), so once hardware-verified it drops into
`imaging.select_hwp_angle` / `pol_seq` bricks in place of the Alpaca rotator.
Offline protocol logic is covered by `tests/` (fake-serial round-trips).

## Building a night: brick-based plans (recommended)

Instead of hand-typing a Python session script, describe a night as a **palette
of reusable bricks** laid under targets — see [`../night_plans/`](../night_plans/)
and `obs_utils/night_plan.py`.

- [`night_plans/palette.yaml`](../night_plans/palette.yaml) — shared, committed:
  reusable bricks (`polV8`, `L60x10`, `BVR45`, `darks60`, …) + a target catalog.
- `night_plans/<date>.yaml` — per night: `uses: palette.yaml`, then a `plan:`
  that lays bricks under targets, with inline overrides (`{polV8: {exp: 45}}`).

Brick types: `stack`, `pol_seq` (HWP-angle sequence on the Pyxis → `HWPANG`
frames), `filter_loop`, `cal`.

```zsh
# Preview the expanded frame timeline + exposure total (no hardware):
.../python scripts/plan_night.py night_plans/example.yaml
# Execute (needs INDIGO/Alpaca + PWI4 up):
.../python scripts/plan_night.py night_plans/example.yaml --run
```

## Session + simulation

- **`new_night_session.py`** — generate a blank *Python* night-session script
  from the `obs_utils.night_session` template (uses the current QHY/EFW/Pyxis
  Alpaca config via `obs_utils.user_config`). Lower-level alternative to the
  brick plans above; prefer the YAML plans for routine nights.
- **`polarimetry_showcase.py`** — end-to-end `poltools` **simulation** showcase
  (no hardware); writes figures/tables under `docs/polarimetry/`.

## Archived

Superseded / dated one-offs live in [`archive/`](archive/).
