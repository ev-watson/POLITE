# Report 2 — Other consolidation candidates

**Date:** 2026-07-30 · **Status:** analysis, no code changed
**Companion to:** `01_package_merge_feasibility.md`
**Baseline:** `pytest` → 244 passed, 0 failed

Report 1 answered the question that was asked. This one answers the follow-up: where
else does the repo keep one thing in two places? Same standard as Report 1 — every
split needs a reason — and the same willingness to say "leave it alone," which §11
does for six candidates I looked at and rejected.

Findings are ordered by how much they matter to the pending relicense commit, not by
line count. None of this is committed yet, so anything not fixed now gets frozen into
the single AGPL commit that becomes POLITE's entire public history
(`RELICENSE-AGPLv3.md` Step 4).

| # | Finding | Effort | Relicense leverage |
|---|---|---|---|
| C0 | **Finish removing `run_night_session` — ~256 orphaned lines now, ~355 once `observatory_control.ipynb` is retired, plus two `CLEANUP.md` notes that describe them as live** | M | **High** |
| C1 | `caltools`/`poltools` still ship standalone-distribution metadata | S | **High — deletes three relicense steps** |
| C2 | `CLAUDE.md` and `AGENTS.md` are one document maintained twice, already drifted | S | Medium |
| C3 | 14 MB of stale notebook output about to be frozen into the baseline commit | S–M | **High — permanent** |
| C4 | The FITS keyword contract is written down twice, in two packages | M | Medium |
| C5 | Three CLI wrappers over the same four QA functions | S | Low |
| C6 | `live.py`'s documented read-only invariant is no longer true | S | Low (correctness) |
| C7 | Four output directories for one kind of thing; two are already empty | S | Low |
| C8 | Site geodesy lives in the block-logging module | XS | Low |

---

## C0 — Finish removing `run_night_session`

`run_night_session` was phased out when `plan_night.py` + `execute_night.py` became
the way a night runs. The function is still in the tree, and so is everything that
only it used. This section says what is left over and what has to be fixed at the same
time, because two notes in `CLEANUP.md` describe leftovers as if they were live.

### C0a — What is left over

`scripts/execute_night.py` does not call `run_night_session`. It imports five helpers
out of `night_session` and runs the night itself:

```86:94:scripts/execute_night.py
from obs_utils.night_session import (
    _apply_session_camera,
    _flush_pol_config_sidecar,
    _run_cal_frames,
    _run_frames,
    _slew_to_target,
    plan_estimated_duration_s,
    plan_total_frames,
)
```

Confirmed by search: `run_night_session` (`night_session.py:776-952`, ~176 lines) has
**no caller** in `obs_utils`, `scripts`, `tests`, or any notebook. The only two
references left are its export entry (`obs_utils/__init__.py:67`) and a docstring in
`night_plan.py:6` that still calls it "the run engine."

Three things are used by that function and nothing else. All three matter more than
ordinary leftover code, because the project's own notes describe them as live.

**1. The read-back settings gate exists twice, byte-identical.** `CLEANUP.md` §3 says
the shared gates were moved into `night_safety.py` and are "imported by the runner and
by the control notebooks, **so the two cannot judge a night differently**." That move
did not finish — `night_session` kept its own copy:

```435:444:obs_utils/night_session.py
    mismatches = []
    for label, commanded, actual in (
        ("gain", ctx.gain_setting, _read_camera(camera, "Gain")),
        ("offset", ctx.offset_setting, _read_camera(camera, "Offset")),
        ("readout_mode", ctx.readout_mode, _read_camera(camera, "ReadoutMode")),
    ):
        if actual is not None and int(actual) != int(commanded):
            mismatches.append(f"{label}: commanded {commanded}, camera reports {actual}")
```

`night_safety.py:468-475` is the same loop with the same message string. Only the
failure branch differs: `night_safety` raises `SystemExit` and offers a
`--skip-settings-check` override, `night_session` raises a plain `RuntimeError`. Their
private readers, `_read_camera` (`night_session.py:425-429`) and `_read`
(`night_safety.py:75-80`), are the same five-line try/except.

**2. The cooler-policy split has only one live half.** `CLEANUP.md` §3 and
`night_safety.py:28-34` both describe two cooler policies kept apart on purpose:
`cooler_gate` prompts on timeout, `night_session.wait_for_cooler` raises. The reason
given is good — a cal ladder at −13 °C is still usable data, a science frame at an
unknown temperature is not.

But `wait_for_cooler` is called in only two places: `night_session.py:794`, inside
`run_night_session`, and `tests/test_cooler_policy.py:50`. Once `run_night_session`
goes, the strict policy has no caller at all. The note reads as though both policies
are in use today. Only one is.

**3. ~~The unbounded mount enable~~ — FIXED IN TREE 2026-07-30, during this review.**
When I surveyed, `mount.enable_motors` looped `while True:` with no timeout and
`startup.py` called it, so the all-night hang on the dead DEC drive was still reachable.
That has since been fixed independently: `obs_utils/waits.py` is new, and
`mount.connect_mount`, `enable_motors`, `home_mount`, and `wait_for_slew` now all take
deadlines (`mount.py:19-22`, `CONNECT_TIMEOUT_S=30`, `ENABLE_TIMEOUT_S=60`,
`HOME_TIMEOUT_S=300`, `SLEW_TIMEOUT_S=300`).

**This finding is closed. Nothing to do.** `mount.enable_motors` is no longer a
deletion candidate on hang grounds — see the table below.

### C0b — What to delete, and what to keep

> **Line numbers below were re-verified 2026-07-30 14:35** against a tree that changed
> mid-review (`mount.py`, `startup.py`, and the new `waits.py` were rewritten while
> this report was being written). Re-check before deleting anything.

**Delete — verified no caller other than `run_night_session`:**

| Item | Where | Lines | Callers found |
|---|---|---:|---|
| `run_night_session` | `night_session.py:776` | 176 | none |
| `_verify_session_hardware` | `night_session.py:432` | 27 | `night_session.py:791` only |
| `_read_camera` | `night_session.py:425` | 5 | `night_session.py:437-439` only |
| `wait_for_cooler` | `night_session.py:467` | 48 | `night_session.py:794`, `tests/test_cooler_policy.py:50` |
| `obs_utils/__init__.py:67` export | — | 1 | — |

**≈ 256 lines.** The duplicated settings gate goes with them.

**Blocked on retiring a notebook, not on this deletion:**

| Item | Where | Lines | Blocker |
|---|---|---:|---|
| `startup_observatory` | `startup.py:137` | ~90 | **`notebooks/observatory_control.ipynb:128,137` calls it** — tracked, not yet retired |
| `StartupState` | `startup.py:47` | 8 | Returned by the above |
| `obs_utils/__init__.py:44-45` exports | — | 2 | — |

`CLEANUP.md` §4 already plans to retire `observatory_control.ipynb` once the
`notebooks/templates/` family has flown a real night. Do that first, then this becomes
a clean deletion. Until then, deleting `startup_observatory` breaks a tracked notebook.

**Do NOT delete — these are live, and two changed during this review:**

| Item | Where | Why it stays |
|---|---|---|
| `startup.connect_field_rotator` | `startup.py:67` | Was the private `_connect_rotator`. Now public **on purpose** — `startup.py:56-57` says so: "Both are public so a notebook can bring up one device without running the whole of `startup_observatory()`." |
| `startup.connect_focuser` | `startup.py:102` | Same. |
| `mount.enable_motors` | `mount.py:40` | Now bounded (`timeout_s=ENABLE_TIMEOUT_S`). The hang is fixed; there is no longer a reason to remove it. |
| `mount.home_mount` | `mount.py:65` | Used by `night_safety.py:323,389` inside `verify_mount`. Also now bounded. |
| `StartupConfig` | `startup.py:31` | Not part of the old engine — it is the config object `execute_night` reads: `config.startup.alpaca` (`execute_night.py:273,275,279`), `.pwi4` (`:297`), `.timing` (`:338`), `.slew_limits` (`:416`). `night_plan.py:296,364` builds one into every plan. The **name** is now misleading, since nothing starts up from it; renaming is cosmetic and separate. |

**Fix in the same commit, or the docs will describe code that no longer exists:**

- `night_plan.py:6` — stop calling `run_night_session` "the run engine."
- `CLEANUP.md` §3 — the cooler-policy note. Keep the reasoning, drop the claim that
  two callers use two policies.
- `CLEANUP.md` §3 — the "gates promoted so the two cannot diverge" note, which is only
  true once the duplicate in `night_session` is gone.

**One thing to decide while doing this.** `execute_night` currently reaches into five
underscore-prefixed helpers (`_run_frames`, `_run_cal_frames`, `_slew_to_target`,
`_apply_session_camera`, `_flush_pol_config_sidecar`). The underscore means "private to
this module," and they are now the run engine, called from outside it. Renaming them
without the underscore is a five-line change and makes the import honest. Not urgent,
but this is the natural moment.

### C0c — `pointing.py` and `platesolve.py`: 207 lines, no callers

`obs_utils/pointing.py` (137) builds a PWI4 pointing model; `obs_utils/platesolve.py`
(70) runs `ps3cli` in a subprocess. Nothing imports `pointing.py` — searching for
`obs_utils.pointing`, `from .pointing`, `build_pointing_model`, and `create_point_list`
finds only its own definitions. `platesolve` is imported at `pointing.py:11` and
nowhere else. Neither is in `obs_utils/__init__.py`'s export table, and neither has a
test.

This is different from C0a. `run_night_session` was replaced; these two were never
wired up in the first place, and a pointing model is exactly what the observatory needs
once the DEC drive is fixed. `CLEANUP.md` §3 records the decision to ship the pointing
path now rather than wait for the repair, so deleting these would undo that decision by
accident.

**Recommendation: keep them, but say so in the code.** One line in each module
docstring — staged for the first pointed night, not yet called by the runner. That
stops the next person reading 207 uncalled lines as an oversight and deleting them. If
they are actually abandoned, delete them; the thing to avoid is leaving them silent.

Sequencing note: `pointing.py:60` is one of the `CaptureRequest` call sites Report 1
§A.8 step 4 would migrate. If these modules are staying, that migration includes them.

---

## C1 — `caltools/` and `poltools/` still advertise themselves as separate packages

**This is the highest-value item in the report, because it shortens the relicense
runbook rather than adding to it.**

The owner vendored both packages on 2026-07-28 (`CLEANUP.md` §5 A.2): they stopped
being independent distributions and became POLITE source. The nested `.git/` dirs
were moved out and the gitlink trap was disarmed. But **the distribution metadata was
left behind**, and all of it is tracked:

| File | Tracked | Says |
|---|---|---|
| `caltools/pyproject.toml` | yes | `name = "caltools"`, `version = "0.1.0"`, `license = { text = "MIT" }`, its own `[build-system]` and `[tool.setuptools]` |
| `poltools/pyproject.toml` | yes | same, plus `dependencies = [..., "caltools>=0.1.0", ...]` |
| `caltools/LICENSE` | yes | MIT, in a tree about to become AGPL |
| `caltools/.gitignore`, `poltools/.gitignore` | yes | seven lines each, all already in the root `.gitignore` |

Three of those are now false statements:

- **`license = { text = "MIT" }` × 2.** POLITE is relicensing to AGPL-3.0-only.
- **`poltools` declares a PyPI dependency on `caltools>=0.1.0`** — a package that is
  no longer published and is now a sibling directory in the same repo. Nothing can
  resolve that.
- **Both declare a build backend**, while the root `pyproject.toml:3-4` states there
  *is* no build step: "modules are imported in place from the repository root."
  Nothing in the repo builds either package; `pytest` reaches them via
  `pythonpath = ["."]`.

**Why this matters more than tidiness.** `RELICENSE-AGPLv3.md` spends three steps
maintaining this metadata *as if the packages still shipped separately*:

- Step 1 (`:220-221`) — copy the AGPL text to `caltools/LICENSE` and `poltools/LICENSE`
- Step 2 (`:231`) — copy `NOTICE` into both, "with their own descriptions"
- Step 3 (`:235-259`) — edit `license =` in both `pyproject.toml`, add an OSI
  classifier to both `classifiers` lists, and bump `setuptools>=61` → `>=77` in both
  because the bare-string `license = "AGPL-3.0-only"` form is **PEP 639** and
  setuptools only accepts it from 77.0

That last one is the tell. The project is about to research and apply a PEP 639
compatibility fix to two build backends that never run, so that two packages that are
never built declare the right license in metadata nobody reads.

**Recommendation.** Delete `caltools/pyproject.toml`, `poltools/pyproject.toml`,
`caltools/LICENSE`, `caltools/.gitignore`, `poltools/.gitignore`. One root `LICENSE`,
one root `NOTICE`, one root `pyproject.toml`, one root `.gitignore` — which is what a
vendored-source layout should look like, and what the owner's A.2 decision already
implied. Steps 1–3 of the runbook then collapse to "put the AGPL text at the root."

**Cost:** 5 files, ~90 lines. **Risk:** none to the test suite — verify with `pytest`
(nothing imports the pyprojects) and by confirming `caltools` and `poltools` still
import, which they do via `pythonpath = ["."]`.

**Then update `RELICENSE-AGPLv3.md` Steps 1–3 to match**, or the next session will
dutifully recreate the files this deletes. That doc is now the single largest carrier
of the pre-vendoring assumption; §A.2's closure note updated the *reasoning* but not
the *procedure*.

**Note:** the version strings (`0.1.0` × 2) are the only thing lost. If versioning
these matters, it belongs in the root `pyproject.toml`, once.

---

## C2 — `CLAUDE.md` and `AGENTS.md` are one document maintained twice

Both files describe the same project to the same audience, with the **same nine
section headings in the same order**: Environment, Shell Scripts, Project Structure,
Conventions, Detector operating point (QHY268M), Data Reduction & Analysis Methods,
Code Style, Package Usage, Current status.

They are not copies — they are paraphrases, 157 and 134 lines, sharing only 25
identical lines. That is the worst of both arrangements: a reader must read both to
be sure, and an editor updating one has no signal that the other exists.

**And they have already drifted.** This is measured, not predicted:

| | `CLAUDE.md` | `AGENTS.md` | Actual |
|---|---|---|---|
| Status section date | `[2026-07-29]` | `[2026-07-28]` | — |
| Test count | "**240 passed**" (`:137`) | "**230 passed**" (`:123`) | **244 passed** (verified this session) |

Neither file is right. A third of the drift happened inside two days, on the one
number a reader is most likely to act on.

`CLAUDE.md` has also **regrown** two stacked `⚠️ Auto-saved before context compact`
stubs (`:151`, `:155`, timestamped 23:16 and 23:19 the same evening, three minutes
apart, both saying the same thing). `CLEANUP.md` §5 P2 records pruning exactly these
on 2026-07-28. They came back within 24 hours, which says the pruning is not a fix.

**Recommendation.** Keep one file as the source. `AGENTS.md` is the cross-tool
convention and the more portable name; `CLAUDE.md` is the longer and more current
text. Merge the content into `AGENTS.md` and reduce `CLAUDE.md` to a pointer, or
symlink it. Both are untracked by design (`.gitignore:40-41`), so this is
zero-risk and invisible to the relicense.

While there: the compact-stub accumulation is worth a one-line rule in the surviving
file — *replace the marker, don't stack it* — since the mechanism writing them clearly
appends.

---

## C3 — 14 MB of stale notebook output is about to become permanent

Tracked notebooks total **15 MB**, of which roughly **14.4 MB is stored output** —
base64 PNGs embedded in the `.ipynb` JSON:

| Notebook | Size | Tracked |
|---|---|---|
| `reductions/reduction_20260302.ipynb` | 5.2 M | yes |
| `reductions/reduction_20260709.ipynb` | 3.6 M | yes |
| `reductions/reduction_20260717.ipynb` | 2.6 M | yes |
| `polarimetry_simulation.ipynb` | 2.4 M | yes |
| `polite.ipynb` | 1.0 M | yes |

**Two of those are already known to be wrong**, and both are already written up:

- `reduction_20260717.ipynb` — 25 of its saved outputs are **copied from the July-9
  reduction**: they report `gain 0` where the night ran at gain 30, and carry `0709_`
  figure names. `CLEANUP.md` §5 P3 records this as **ESTABLISHED** by inspection, with
  the warning that "the saved output is not July-17 science and must not be cited as
  such."
- `polarimetry_simulation.ipynb` — its sources are correct, but its stored outputs
  show the **retired FITS cards** and the **old 60 px beam split** that Q7/Q8 removed
  on 2026-07-29 (`next-session-prompts.md:57-64`).

The relicense makes this urgent in a way it is not today. `RELICENSE-AGPLv3.md`
Step 4 replaces history with **a single orphan commit**. Everything tracked at that
moment becomes POLITE's entire published history, permanently, with no earlier
revision to point at. Right now that includes ~5 MB of figures that assert the wrong
gain and the wrong beam geometry, in the two files a reader is most likely to open to
see what POLITE produces.

**Recommendation.** Before the baseline commit, strip stored output from the tracked
notebooks — `jupyter nbconvert --clear-output --inplace`, or `nbstripout` as a git
filter if it should stay stripped. Sources are unaffected; the notebooks stay
runnable, and `tests/test_notebook_templates.py` already asserts the templates ship
unexecuted, so the convention exists. Reviewed figures belong in a dated `generated/`
subtree (the disposition `CLEANUP.md` §5 P3 already proposes) or in `docs/`, where
`docs/polarimetry/figures/` holds five tracked PNGs today.

Two smaller notebook items, both owner calls:

- **`workspace.ipynb` (24 K) is tracked but absent from `README.md`'s notebook
  block** (`README.md:75-83`), which lists every other notebook in the tree. Its only
  mention anywhere is `docs/polarimetry/02_architecture_analysis.md:16`, which pairs
  it with a `reduction.ipynb` that no longer exists — so that reference is stale too.
  `polite.ipynb` (1.0 M) *is* listed, as "Main analysis notebook" (`README.md:82`);
  the question there is only whether 1 MB of stored output belongs in the baseline
  commit, which is the same question as above.
- **`lab_control.ipynb` and `observatory_control.ipynb` are superseded** by
  `templates/commissioning.ipynb` + `cal_night.ipynb` (`CLEANUP.md` §4). The stated
  condition for retiring them is "once the family has been used on a real night." A
  night is running today, so that condition is about to be met — worth checking
  before the commit rather than after.

---

## C4 — The FITS keyword contract is written down twice, in two packages

`alpyca_tools/fits_writer.py` (acquisition) and `poltools/io.py` (reduction and the
simulator) each hold a hand-maintained table of the polarimetry FITS keywords, and
each documents itself as mirroring the other:

- `fits_writer.py:35` — "Mirrors ``poltools.io.POL_KEYWORDS`` so simulated and real
  POLITE frames carry identical cards."
- `poltools/io.py:34` — "See :class:`alpyca_tools.fits_writer.PolarimetryCards` for
  the reasoning."

`PolarimetryCards` carries six fields (HWPANG, WPUNCERT, INSTROT, POLBEAM, POLSEQ,
POLSEQN); `POL_KEYWORDS` carries those six plus PIXSCALE, READMODE, SET-TEMP. Nothing
derives one from the other. The only thing holding them together is
`tests/test_fits_header_contract.py`.

That the pair *stayed* consistent through the Q7 header-provenance change — seven
keywords retired across both files on 2026-07-29 — is a credit to the process, not
evidence that the arrangement is safe. It is one edit away from a simulator writing a
card the reducer does not read.

**The underlying problem:** `fits_writer.py` is in the wrong place. It is not an Alpaca
camera helper — it is the POLITE FITS schema, and it already has three users on both
sides of the acquisition/reduction line: the live camera path, the simulator, and
`poltools.io`. That is why a reduction library ends up importing an acquisition package
at `poltools/io.py:19`.

**Recommendation.** Sequence this *after* Report 1's merge, then make the moved
`obs_utils/fits_writer.py` the single owner of the keyword table — including
`POL_KEYWORDS` — and have `poltools/io.py` import it instead of restating it. The
`_set_card` / comment strings become data in one place, and
`test_fits_header_contract.py` can drop the halves that exist only to compare two
tables.

If the layering inversion is judged worse than the duplication, the alternative is a
small dependency-free `politefits/` (or `obs_utils/fits_schema.py` imported by both).
That is a bigger call than this report should make — flagging it as **OPEN**.

---

## C5 — Three CLI wrappers over the same four QA functions

`obs_utils/qa_lib.py` exposes `run_bias_qa`, `run_first_light_qa`,
`run_flat_quality_gate`, `run_sequence_audit`. There are three separate surfaces over
them:

1. **`obs_utils/qa_gates.py`** — YAML-driven in-pipeline dispatch with a `_HANDLERS`
   table (`qa_gates.py:28-33`), abort policy, and level-aware logging. **Genuine
   work; keep.**
2. **`obs_utils/live.py:519-544`** — four three-line notebook one-liners plus
   `_as_paths` coercion. **Cheap; keep.**
3. **`scripts/{bias_qa,first_light_qa,flat_quality_gate}.py`** — 96 lines across three
   files, of which ~60 are `argparse` declarations restating each function's keyword
   arguments. All three are the same shape: repo-root `sys.path` insert, build parser,
   call one function, `print(result.to_json())`, exit on `result.passed`.

A fourth surface already existed and was already retired: `scripts/sequence_audit.py`
was deleted in the 2026-07-26 trim precisely because the logic lives in `qa_lib` and
the manual invocation belongs in a notebook (`CLEANUP.md` §2). The same reasoning
applies to the three that remain — the trim just did not finish the row.

**Recommendation.** One `scripts/qa.py <gate> <paths> [--opt ...]` dispatching through
`qa_gates._HANDLERS`, so the CLI and the runner cannot disagree about what a gate is
by construction. 3 files → 1, ~96 lines → ~45, and the flag defaults stop being
restated (note `bias_qa.py:19-20` hardcodes `--ron-target 3.5` in the CLI, a number
that exists nowhere else).

**Risk:** low, but it is an operator-facing interface change — `scripts/README.md` and
any checklist naming the three commands must move in the same edit.

---

## C6 — `live.py`'s read-only invariant is no longer true

`CLEANUP.md` §4 states the design guarantee plainly:

> Read-only by construction: it only reads FITS off disk, so a stale cell re-run
> costs a redrawn figure, never a frame.

That is now false. `obs_utils/live.py:69-96` exposes six pass-throughs into
`obs_utils/interactive.py`, all of which command hardware:

```69:96:obs_utils/live.py
def connect_camera(**kwargs):
    """Connect the QHY268M only. See :func:`obs_utils.interactive.connect_camera`."""
    return _interactive().connect_camera(**kwargs)
...
def shutdown() -> None:
    """Release camera / wheel / rotator. Does not touch the mount."""
    _interactive().shutdown()
```

All six are in `live.__all__`. A stale re-run of a cell calling `live.shutdown()`
disconnects the camera mid-sequence — exactly the failure mode the invariant was
written to exclude.

**Recommendation: restore the boundary, do not merge it away.** Delete the six
pass-throughs. The three control notebooks already import `interactive` directly and
under its own name (`from obs_utils import interactive as obs`, in the byte-identical
shared preamble of all three templates), so the pass-throughs save nobody a keystroke
— they only make `live` ambiguous.

This is the one place in either report where I am recommending *against* consolidation
on a strict reading of the criteria. Deleting 30 lines and one convenience is worth it,
because that guarantee is the reason `live.py` exists — it is what makes it safe to
re-run a cell at 3 a.m. mid-sequence. If the pass-throughs are genuinely wanted, put
them behind `live.control.*` so the name warns you, and correct `CLEANUP.md` §4 to
match.

**Then add a test.** `tests/test_live.py` already exists, and
`tests/test_optional_imports.py` shows how (assert a module is absent from
`sys.modules`). A rule this important should not live only in a paragraph of a planning
document.

---

## C7 — Four output directories for one kind of thing, two already empty

| Directory | Files | Size | Tracked | `.gitignore` |
|---|---|---|---|---|
| `tmpimg/` | 13 | 16 M | 0 | ignored |
| `final-imgs/` | 5 | 107 M | 0 | ignored |
| `paperfigs/` | **0** | **0** | 0 | ignored |
| `generated/` | **0** | **0** | 0 | ignored |
| `docs/polarimetry/figures/` | 5 | — | **5** | tracked |

Four ignored directories holding generated images, plus a fifth tracked one that holds
the same kind of thing for a different reason. Three of the four are named for their
*intent* (`tmp`, `final`, `paper`) rather than their *provenance*, which is why it is
not possible to tell from the tree which figures came from which reduction.

**Two of them are empty, and that changes a pending owner action.**
`next-session-prompts.md:70-73` and `CLAUDE.md` both still list
`rm -rf generated/salvage_first_light_20260709` (~331 MB) as an outstanding owner
decision. `generated/` now contains **zero files and zero bytes** — it has already
been done. `paperfigs/`, which both documents say is "deliberately kept," is likewise
empty. The top prompt is stale on both counts and will send the next session looking
for files that are gone.

**Recommendation.** Two parts, in order:
1. **Correct `next-session-prompts.md` and `CLAUDE.md`** — remove the `generated/`
   deletion from the outstanding list. This is documentation accuracy, and it is the
   part that matters, since a stale prompt costs the next session real time.
2. **Then consider** one ignored `outputs/` with dated subdirectories, replacing four
   entries in `.gitignore` with one. `final-imgs/` (107 MB) is the only one with real
   content; `tmpimg/0709_*` is explicitly kept. Low priority and purely cosmetic —
   but note it is now a *four-line* change, not a data migration, because two of the
   directories are empty.

---

## C8 — Site geodesy lives in the block-logging module

```18:20:obs_utils/block_log.py
JULIAN_LAT_DEG = 33.0701
JULIAN_LON_DEG = -116.6451
JULIAN_ELEV_M = 1294.0
```

Single-sourced — no duplication, so this is a naming finding, not a correctness one.
But the observatory's position is imported *from a module whose docstring reads
"Structured per-block observing log (JSONL manifest)"* by `session_context.py:10` and
`night_session.py:17`, and it ends up in `SITELAT`/`SITELONG`/`OBSGEO-*` on every
frame. `obs_utils/config.py` is where a reader looks for it.

**Recommendation.** Move the three constants to `config.py`; re-export from
`block_log.py` for one release or fix the two import sites directly (there are only
two). **Cost:** ~5 lines. **Risk:** none — `tests/test_fits_header_contract.py:101-103`
and `tests/test_obs_math.py:37` both pin the values.

---

## C9 — Script locations, for completeness

There are three `scripts/` directories: `scripts/`, `alpyca_tools/scripts/`, and
`qhy_alpaca/scripts/`. Report 1 §A.4 folds the `alpyca_tools` pair into `scripts/`.
`qhy_alpaca/scripts/` should stay — it ships with the server, runs on the Windows
observatory PC, and is the implementation the root shim resolves to.

Two is the right number, and it is reached as a side effect of Report 1. Nothing extra
to do here.

---

## C10 — Smaller verified items

Each of these is confirmed by direct inspection. None is worth a section on its own;
together they are the kind of thing that should not be frozen into a baseline commit.

**One list, five source locations — the EFW carousel.** The physical filter order is
written out at `obs_utils/config.py:60-70` (dataclass default),
`obs_utils/user_config.py:42-48` (the live instance, restating the default verbatim),
`obs_utils/night_safety.py:57` (`INSTALLED_EFW_NAMES`), `poltools/_types.py:163-172`
(`default_efw_filters`), and `obs_utils/night_session.py:103` (slug map). The first
three each carry their own comment insisting the order must match the carousel. That
is three warnings guarding three copies of one fact that hardware bring-up verified
once, on 2026-07-10.

**A default that bring-up already disproved.** `PyxisSerialConfig.baud` defaults to
`19200` (`config.py:32`) with a docstring saying "Baud is not stated by the manual —
19200 is the most likely value"; the live instance sets `115200`
(`user_config.py:28`). Not a defect — `autodetect_baud=True` covers it — but the
default and its justification are now known-wrong, and the value that works is in the
file a reader is less likely to open.

**`_slugify` is duplicated verbatim**, `obs_utils/logging.py:30-34` and
`obs_utils/night_session.py:159-163`. Same five lines; the fallback string differs
(`"session"` vs `"target"`), which is the entire distinction.

**`AnalysisResult` and `StokesResult` are the same dataclass.**
`caltools/_types.py:62-81` and `poltools/_types.py:343-363` both carry `name`,
`scalar_summary`, `maps`, `metadata` with identical types and defaults, differing only
in `__repr__` — and the poltools docstring already says "compatible with
`caltools.AnalysisResult`". `obs_utils.qa_lib.QAResult` is a third result container
with a different field set. Given `poltools` already imports `caltools`
(`poltools/io.py:18`), one could subclass the other. Low urgency, but it is the kind of
near-miss that makes a shared plotting or reporting helper impossible to write.

**`poltools/plotting.py` forces the Agg backend at import time**
(`plotting.py:15-16`, `matplotlib.use("Agg")`). `caltools/plotting.py` does not. Because
`poltools/__init__.py` does not import `.plotting`, this only fires on an explicit
`import poltools.plotting` — at which point a notebook silently loses interactive
figures for the rest of the kernel. A library should not choose the backend for its
caller; the three control-notebook templates would be the ones affected.

**`poltools/README.md` has mojibake in its first paragraph** — line 3 reads `?-BBO
Savart analyzer` where `caltools/README.md` and `poltools/__init__.py:12-13` use a
proper `α`. It is the second sentence of a file that is about to be published. The same
class of defect was already fixed once in `scripts/launch_qhy_server.ps1`
(`RELICENSE-AGPLv3.md` A.3), so a quick encoding sweep before the baseline commit is
cheap.

**`plan_night.py` prints less than `execute_night.py`'s dry run does.** Both are in
active use, so this is not a retirement candidate — it is a mismatch worth knowing
about. `plan_night.py:38-48` is `print(describe(config))` plus a pointer to the runner.
`execute_night.py:524-526` calls `_describe`, which starts with the same
`print(describe(config))` at `:160` and then adds cooler setpoint, output directory,
duration estimate, HWP angles, and the mount plan (`:161-199`). So `plan_night` gives a
strictly smaller preview than `execute_night` without `--run`. If the difference is
deliberate — a fast plan check that needs no camera context — a line in
`scripts/README.md` saying which to use when would settle it. If it is not deliberate,
`plan_night` should call `_describe` too.

**`scripts/__pycache__/` holds 19 `.pyc`, 14 with no source file** — including
`run_calibration_night`, `generate_site_checklists`, `analyze_salvage_first_light`,
`reduce_salvage_drift_sequence`, and `sequence_audit`, all deleted in the 2026-07-26
and 07-28 trims, plus `night_session_20260302` from the retired dated-script era and
two `cpython-314` files for a Python the project does not target. Ignored by git, so
invisible to the commit — but they are why a stale import can still resolve locally.
`find . -name __pycache__ -prune -exec rm -rf {} +` before the final test run makes the
244-passing result mean what it says.

**`reports/` and `docs/` hold the same kind of thing with opposite tracking.**
`reports/` has three substantive prose analyses — `20260709_salvage_first_light_analysis.md`
(26 K), `20260710_code_reduction_audit.md` (13 K), `eb_draft_science_roadmap.md` (27 K)
— all untracked, because `.gitignore:57` ignores the directory wholesale. The
`.gitignore:32-35` comment states the intended rule: `docs/` is for "operator- and
science-facing documents," root is scratch. These three are neither scratch nor
operator docs; they are the same genre as `docs/polarimetry/02_architecture_analysis.md`,
which *is* tracked. Decide per file — a science roadmap and a first-light analysis
plausibly belong in `docs/`; a code audit superseded by these two reports plausibly
does not.

---

## §11 — Checked and rejected

Listing these because "we looked and the boundary is real" is a finding, and because
each is a plausible-looking merge that would make the project worse.

| Candidate | Why it stays split |
|---|---|
| `night_display.py` (246) vs `live.py` (583) | `night_display` renders a `rich` progress bar **during** a run on a TTY and no-ops without one; `live` reads finished FITS **off disk** after the fact. No shared code, opposite lifetimes. |
| `startup.py` (196) vs `interactive.py` (600) | `startup_observatory` is all-or-nothing scripted bring-up that **homes the mount**; `interactive` is an idempotent notebook singleton that tops up one subsystem at a time and deliberately never touches the mount (`interactive.py:19-21`). Merging would put a mount-homing side effect one keystroke from a notebook cell. `startup.py` also now exposes `connect_field_rotator` and `connect_focuser` publicly so a notebook can bring up one device without homing anything (`startup.py:56-57`) — which is this split working as intended. |
| `qa_gates.py` (101) vs `qa_lib.py` (412) | Policy vs measurement. `qa_gates` owns abort behaviour, per-handler path resolution, and the rule that a gate must never kill a capture night (`qa_gates.py:81`). `qa_lib` computes. Correct split. |
| `caltools/` vs `poltools/` | Different domains (detector characterization vs Stokes reduction) with a real one-way dependency: `poltools/io.py:18` imports `caltools`. Merging would hide that direction. Only their *packaging metadata* should go — see C1. |
| `obs_math.py` (113) vs `pointing.py` (137) / `platesolve.py` (70) / `horizons.py` (97) | `obs_math` is deliberately free of every `obs_utils` import so writers and readers can use it without the device layer (`obs_math.py:3-5`) — a stated constraint, and the reason `poltools` can reach it. The other three are a PWI4 pointing-model builder, a `ps3cli` subprocess wrapper, and a JPL Horizons client. Four unrelated things that happen to involve the sky. |
| `obs_utils/pol_config.py` (86) | Looks like a shim, is an adapter: it translates `SessionCaptureContext` (acquisition) to `SessionDetectorConfig` (reduction) and is the only place the nominal-vs-measured beam-geometry flag is set on the capture side (`pol_config.py:20-34`). That logic is Q8's fail-closed guarantee. Keep. |

---

## Suggested order

Ordered so the relicense-blocking work lands first and nothing depends on Report 1.

**Before the baseline commit** — these change what gets frozen into it:
1. **C0** — delete the `run_night_session` chain (≈256 lines) and fix the two
   `CLEANUP.md` notes and the `night_plan.py:6` docstring in the same commit. Largest
   single reduction here, and it removes the duplicated settings gate. `startup_observatory`
   waits on retiring `observatory_control.ipynb`.
2. **C1** — delete the standalone package metadata, then update `RELICENSE-AGPLv3.md`
   Steps 1–3 so the next session does not recreate it.
3. **C3** — strip stored notebook output; decide on `workspace.ipynb`.
4. **C7 part 1** — correct the stale `generated/` action in `next-session-prompts.md`
   and `CLAUDE.md`.
5. **C2** — collapse `CLAUDE.md` into `AGENTS.md`. Untracked, so it costs nothing, but
   it is what the next session reads first.
6. **C10, the cheap half** — clear `__pycache__` before the final test run, fix the
   `poltools/README.md` mojibake, decide `reports/` per file.

**Independent, any time:**
7. **C0c** — one docstring line each in `pointing.py` and `platesolve.py`.
8. **C6** — restore `live.py`'s read-only guarantee and add a test for it.
9. **C8** — move the site constants out of `block_log.py`.
10. **C5** — one `scripts/qa.py`.

**After Report 1's merge:**
11. **C4** — single-source the FITS keyword table.

Every step above is verified the same way: `pytest -q` stays at 244 passed, plus
`tests/test_optional_imports.py` for anything touching import structure. Nothing in
this report changes acquisition numerics, header *values*, or `FITSDATA/`.

One caveat on that baseline, from C0: `tests/test_cooler_policy.py` exercises
`wait_for_cooler`, which nothing else calls, so part of the 244 is testing code that
is about to be deleted. Expect the count to drop when C0 lands — that is the correct
outcome, not a regression.
