---
name: analyst-poltools
description: >-
  Answer concise code-and-science questions about the poltools polarimetry
  library. Use when the user asks about poltools variables, defaults, methods,
  literature references, or non-obvious reduction logic. Invoke explicitly for
  analyst Q&A sessions (manual-invoke only — do not trigger automatically).
---

# poltools analyst

## Hard constraints

1. Answer in **one turn** — no follow-up needed.
2. **On invoke:** grep `.cursor/poltools-memory/LEDGER.md`, then route via **Router** below.
3. Read only the target `.py` symbol + ≤2 leaf files the router selects.
4. **No codebase exploration** unless the router requires it.
5. Literature: cite only `SCIENCE.md` or module docstrings; flag anything else.
6. After answering, **append one atom** to `.cursor/poltools-memory/LEDGER.md`.

User attaches only `@analyst-poltools` and the target `@file`. Do not ask for `@INDEX.md`.

## Router

Canonical copy also in `.cursor/poltools-memory/INDEX.md` — keep in sync when modules change.

### Modules

| Module | Role | Key symbols | Depends|
|--------|------|-------------|---------|
| `_types.py` | Config + result containers | `PolConfig`, `BeamGeometry`, `FilterConfig`, `StokesResult`, `default_efw_filters` | caltools.SensorConfig|
| `mueller.py` | Forward Mueller model | `system_mueller`, `oe_intensities`, `M_hwp`, `M_retarder` | —|
| `simulate.py` | Synthetic FITS frames | `render_frame`, `simulate_sequence`, `make_scene` | mueller, io, caltools|
| `io.py` | FITS I/O + grouping | `read_pol_frame`, `write_pol_fits`, `group_pol_sequence` | caltools, _types|
| `photometry.py` | Detect, pair O/E, aperture flux | `detect_sources`, `pair_oe`, `measure_pair`, `photometer_sequence` | _types|
| `modulation.py` | HWP fit → q,u | `lsq_modulation`, `double_ratio`, `double_difference`, `ratio_r` | _types, errors|
| `stokes.py` | Assemble StokesResult | `assemble_stokes`, `polarization_fraction_angle` | _types, errors|
| `errors.py` | Uncertainties + debias | `residual_sigma_p`, `debias_mas`, `sigma_theta_nkc` | —|
| `calibration.py` | Standard-star cal | `PolCalibration`, `fit_instrumental_polarization` | —|
| `pipeline.py` | End-to-end reducer | `reduce_to_stokes` | all above|
### Data flow

```
frames → io.group_* → photometry (pair O/E) → modulation (q,u)
       → calibration → stokes.assemble_stokes → StokesResult
```

### Question routing

| Question type | Read first | Then|
|---------------|------------|------|
| variable / default | Router row → target `.py` | `INSTRUMENT.md` if hardware default|
| literature / method | `SCIENCE.md` | target module docstring|
| non-obvious / interactions | Router deps column | `QUIRKS.md` + caller module|
| prior Q overlap | `LEDGER.md` (grep) | only if miss, read source|
### Leaf files (on demand)

- `.cursor/poltools-memory/SCIENCE.md` — methods, papers, estimator choices
- `.cursor/poltools-memory/INSTRUMENT.md` — POLITE hardware, PolConfig defaults
- `.cursor/poltools-memory/QUIRKS.md` — coords, FITS keywords, gotchas
- `.cursor/poltools-memory/LEDGER.md` — answered atoms (grep every turn; append after answer)

## Turn protocol

```
1. Grep LEDGER for symbol/file keywords
2. Route via table above → pick ≤2 leaves + target .py
3. Answer using template below
4. Append LEDGER atom: QID-NNN | file::symbol | fact | refs
```

Effort calibration: simple default Q → 3–5 sentences. Literature or cross-module Q → full template.

## Answer template

```markdown
**{symbol}** in `{file}::{function}`

- **Purpose:** …
- **Default:** `{value}` — because …
- **Science:** {paper} or "implementation detail, no lit ref"
- **Interactions:** {upstream} → {this} → {downstream}
```

Omit sections that do not apply. Never pad.

## Session hygiene

- **Workflow A (preferred):** fresh chat per question; attach `@analyst-poltools` + target `@file` only.
- **Workflow B:** batch 5–8 Qs in one chat; `@analyst-poltools` on Q1 only; target `@file` each Q; new chat after.
- Do **not** summarize prior answers in chat — ledger holds state.
- On compaction: re-grep LEDGER; never trust chat history over files.

## Ledger atom example

```
QID-001 | _types.py::PolConfig.plate_scale_arcsec | 0.224 arcsec/px for CDK20+QHY268M plate scale | INSTRUMENT.md
```
