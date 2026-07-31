# Control-notebook palette

Copy `night.ipynb` into `notebooks/observation_notebooks/YYYYMMDD_<purpose>.ipynb`.
Then pull only the needed cells from the three palettes; do not edit a palette in
place.

| Notebook | Purpose |
|---|---|
| `night.ipynb` | Starter, tonight's provenance card, non-moving connection/status, and shutdown. |
| `devices.ipynb` | Connect, inspect, and explicitly operate camera, EFW, HWP, PWI4 field rotator, focuser, and mount. |
| `capture.ipynb` | Supervised probe, focus, centering, calibration, modulation, plan-preview, and ledger cells. |
| `inspect.ipynb` | Read-only frame, group, trend, image, live-watch, and QA views. |

The first two code cells are byte-identical in all four notebooks. Run them
first. Palette cells are deliberately short; a cell that commands hardware
starts with `# MOTION`. `night.ipynb` never commands mechanical motion.

`live` is read-only. `obs` commands the assembly. The HWP is the Alpaca/serial
Pyxis (`connect_hwp`, `hwp_rotator`); PWI4's instrument de-rotator is always the
field rotator (`connect_field_rotator`, `field_rotator_*`). Do not run palette
capture cells while `execute_night.py --run` owns the camera or mount.

Preview a plan in a terminal with:

```zsh
python scripts/execute_night.py night_plans/<tonight>.yaml
```

Add `--run` only after reviewing that preview and its gates. The plan runner is
the normal path for recorded calibration and science sets; palette captures are
supervised diagnostics. The default detector setting is Mode 5, gain 56, offset
20 unless a reviewed plan explicitly deviates. Notebook outputs are intentionally
empty; reduction, rather than a live display, is the scientific product.
