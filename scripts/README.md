# POLITE scripts

Operator entry points. All Python scripts add the repo root to `sys.path`, so
run them from the repo root with the POLITE interpreter:

    PY=/Users/blu3/miniforge3/envs/POLITE/bin/python

The device stack: QHY268M camera + ZWO EFW filter wheel + Optec Pyxis HWP
rotator are exposed over ASCOM Alpaca (INDIGO `indigo_agent_alpaca` bridge or
the POLITE QHY Alpaca server); the PlaneWave mount + field rotator stay on
PWI4. **DEC drive is down** — the pointing path below is wired but unexercised.

## Server

| Script | What it does |
|---|---|
| `launch_qhy_server.ps1` | Launch the POLITE QHY Alpaca camera server (obs PC / Windows). Shim; the implementation is `qhy_alpaca/scripts/start_qhy_alpaca_server.ps1`. |

## Night execution

| Script | What it does |
|---|---|
| `plan_night.py <plan.yaml>` | **Preview** a brick plan — expands the frame timeline + exposure total. Touches no hardware. |
| `execute_night.py <plan.yaml> --run` | **Run** a brick plan (mount + camera + EFW + HWP). Settings banner with hardware read-back, fail-closed gain/offset/readout + EFW-name checks, conditional mount and HWP gates, cooler-stabilization gate, per-invocation output subdir. `--setpoint T`, `--yes`, `--skip-*-check`. |

Brick plans live in `night_plans/` (bricks defined once in `palette.yaml`).

**HWP.** `--hwp auto` (the default) connects the Pyxis rotator only when the
loaded plan actually steps it, so a dark ladder never drags the rotator into the
run and a polarimetry plan can never capture at an unknown angle: `--hwp off` on
a plan with HWP bricks aborts before touching disk. Preflight moves to the
plan's first angle and verifies arrival within `--hwp-tol` (default 0.25°, well
above the open-loop Gen3's ~0.012° step quantization). Override with
`--no-hwp-preflight` (no motion) or `--skip-hwp-check` (no gate).

**Mount.** `--mount auto` (the default) connects PWI4 only when the plan names a
sky position — `ra`/`dec` or `alt`/`az` on a target. `night_safety.verify_mount`
then connects, energizes **both axes with a deadline**, homes, and logs the state
it read back; a plan with coordinates run as `--mount off` aborts unless
`--unpointed` is also given. Cal-only plans never touch the mount.
`--no-mount-home` skips homing, `--park-on-finish` parks after the data is safe,
`--skip-mount-check` downgrades the gate to a warning.

POLITE's DEC drive (axis 1) does not currently engage, so this path has **not**
been run against a working mount — the first pointed night is commissioning.
Nothing is special-cased to the fault: the gate simply refuses to slew an axis
that will not energize, and `obs_utils.mount.enable_motors`' unbounded `while
True` poll is bypassed so a dead drive aborts in under a minute instead of
hanging until dawn.

**Detector settings.** The default operating point is **Mode 5, gain 56, offset
20** (QHY's lowest-read-noise QHY268M mode at the lowest gain reaching that floor,
chosen to preserve dynamic range — a first-order polarimetric constraint; **Mode 3,
gain 0** is the high-full-well alternative). It comes from
`SessionCaptureContext`, so a plan without a `camera:` block runs on it after a
WARN naming the values; an unrecognized key inside `camera:` also WARNs and is
ignored, which is what catches `offest: 20`. The settings banner then read-backs
gain/offset/readout from hardware and the check is fail-closed. Conversion gain and
read noise are reduction results and are not accepted here.

The shared safety gates live in `obs_utils/night_safety.py`, so the runner and
the control notebooks judge a night by the same code.

## QA gates

Each prints a JSON `QAResult` and exits non-zero on FAIL. Also dispatched
in-pipeline during a run via `obs_utils/qa_gates.py`.

| Script | Gate |
|---|---|
| `bias_qa.py <paths>` | Bias histogram + read-noise (`--ron-target`, `--ron-tol`). |
| `flat_quality_gate.py <paths>` | Flat quality: `lsq` vs `double_ratio` q,u discrepancy. |
| `first_light_qa.py <paths>` | Polarimetric first-light gate against a standard. |

## Analysis utilities

None. Analysis belongs in `caltools` / `poltools` and is driven from a notebook,
not from a one-off script — the single-point Janesick gain check that used to
live here is now `caltools.conversion_gain_from_flat_pair`, called from §8 of
`notebooks/templates/cal_night.ipynb`.

## Site checklists

Removed 2026-07-28. `generate_site_checklists.py` and the four PDFs it emitted
hard-coded the retired July-9 plan and printed **unsafe** instructions — mount
homing on a dead DEC drive, and a `plan_night --run` entry point that no longer
exists. A checklist that contradicts the runner is worse than none. The
authoritative pre-flight is now the runner itself: `execute_night.py` without
`--run` prints the full execution order, the device set it will connect, and
every gate it will apply.
