# Report 1 — Feasibility of merging `alpyca_tools`, `qhy_alpaca` and `obs_utils`

**Date:** 2026-07-30 · **Status:** analysis, no code changed
**Baseline verified this session:** `pytest` → **244 passed, 0 failed** (14.2 s)
**Context:** repo consolidation during the AGPLv3 relicense
(`RELICENSE-AGPLv3.md`, `CLEANUP.md` §5).

---

## Verdict

| Merge | Verdict | One-line reason |
|---|---|---|
| `alpyca_tools` → `obs_utils` | **FEASIBLE — recommended** | Same license class, same process, same machine, already imported by six `obs_utils` modules; the boundary buys nothing and costs a duplicated dataclass, a dead `__init__`, and a compatibility shim. |
| `qhy_alpaca` → `obs_utils` | **NOT FEASIBLE — and should not be attempted** | It is third-party MIT code under another author's copyright, it is a separate FastAPI server process on a different machine, and the relicense plan uses its *directory path* as the legal boundary. |

The premise in the request — "technically `alpyca_tools` and `qhy_alpaca` are all
utilities for the observatory" — is right about `alpyca_tools` and misleading about
`qhy_alpaca`. `qhy_alpaca` is not a utility POLITE calls; it is a **peer process
POLITE talks to over HTTP**, and it is not POLITE's code to fold in.

---

## How I read the criteria

The stated criteria were: does it make sense, is it more streamlined, simpler, less
spaghetti/clutter, fewer lines. Taken literally, "fewer lines" argues for merging
everything, because gluing two directories together always deletes some `__init__` and
import boilerplate. So I used the intent instead: **a reader should find one thing in
one place, and every split in the tree should have a reason.**

That is why the two answers come out differently:

- **`alpyca_tools` vs `obs_utils` is one thing split in two.** Both are POLITE's own
  code, both go under AGPL, both are plain Python running in the same process on the
  same Mac. There is nothing on the other side of that line.
- **`qhy_alpaca` is genuinely separate**, in three ways at once: someone else owns the
  copyright, it runs as its own process, and it runs on a different computer. Merging
  it would make the tree shorter and the project harder to understand.

I also weighed one thing nobody asked for, because we are doing this during a
relicense: **does the merge make the licensing simpler or messier?** Merging
`alpyca_tools` makes it simpler — one fewer directory to list in Step 6. Merging
`qhy_alpaca` makes it much messier, for the reasons in Part B.

---

## Part A — `alpyca_tools` → `obs_utils`: feasible, recommended

### A.1 The boundary is not real

`alpyca_tools` describes itself as "internal Alpaca client helpers"
(`alpyca_tools/README.md:3`). Everything about how it is used says it is already a
part of `obs_utils`:

- **Six `obs_utils` modules import it directly**: `alpaca.py:10`, `imaging.py:9-10`,
  `night_session.py:12-13`, `autoguide.py:19`, `pointing.py:8`,
  `interactive.py:221`, plus `fits_routine.py`.
- **Nothing imports it as a package.** There is not one `from alpyca_tools import X`
  anywhere in the repo — every consumer reaches past the package into a submodule.
  The 43-line lazy-export table in `alpyca_tools/__init__.py` is therefore **dead
  surface**: 13 entries, zero users.
- **The relicense already treats the two identically.** `RELICENSE-AGPLv3.md:468`
  lists the AGPL header "minimum coverage" as `obs_utils/*.py`, `alpyca_tools/*.py`,
  `scripts/*.py`, `caltools/*.py`, `poltools/*.py` — one class, five paths. Merging
  removes a path from that list without changing its meaning.
- **No module-name collisions.** `alpyca_tools` contributes `camera_device`,
  `camera_ops`, `fits_writer`, `discovery`, `schema`, `telemetry`; none of the 29
  `obs_utils` module names clash.

### A.2 The cost of keeping it: one dataclass defined twice

This is the single strongest piece of evidence, and it is not a style complaint.

`obs_utils.imaging.CaptureRequest` (`imaging.py:17-31`) and
`alpyca_tools.camera_ops.ExposureSettings` (`camera_ops.py:33-47`) are the **same
thirteen fields, in the same order, with the same defaults, byte for byte**:

```17:31:obs_utils/imaging.py
@dataclass
class CaptureRequest:
    exposure_s: float
    is_light: bool = True
    binx: int = 1
    biny: int = 1
    startx: int = 0
    starty: int = 0
    numx: Optional[int] = None
    numy: Optional[int] = None
    gain: Optional[int] = None
    offset: Optional[int] = None
    readout_mode: Optional[int] = None
    sub_exposure_duration: Optional[float] = None
    fast_readout: Optional[bool] = None
```

Because they are separate types, `imaging.py` converts one into the other by hand —
**twice**, once in `capture_image_array` (`imaging.py:56-70`) and once in
`capture_fits_file` (`imaging.py:89-103`). Thirty lines whose entire job is to
rename a type.

And the duplication has already leaked upward. `obs_utils/night_session.py` imports
**both** and constructs **both**, for the same camera in the same run:
`ExposureSettings` at line 387 to apply the session's gain/offset/mode, and
`CaptureRequest` at line 584 to take the frame. A reader has to discover that these
are the same thing.

`obs_utils/pointing.py:60` and `obs_utils/interactive.py:203` build `CaptureRequest`;
`alpyca_tools/scripts/snap_fits.py:53` and `obs_utils/fits_routine.py:79` build
`ExposureSettings` — for the same operation. Two vocabularies for one concept, split
along the package line, is exactly the clutter the criteria target.

### A.3 A compatibility shim with nothing left to be compatible with

`obs_utils/fits_routine.py` (111 lines) opens by saying so:

```1:7:obs_utils/fits_routine.py
"""Compatibility wrapper for the POLITE FITS acquisition writer.

New acquisition code should import from :mod:`alpyca_tools.fits_writer`.
This module remains for older scripts that use ``CaptureConfig`` and the
``obs_utils`` capture entry point; it delegates all header construction and
FITS serialization to the authoritative writer and contains no schema copy.
"""
```

The "older scripts" it exists for **no longer exist** — they were deleted in the
2026-07-26 script trim (`CLEANUP.md` §2). Its only remaining consumer in the whole
repo is `tests/test_fits_header_contract.py:12`, a test asserting that the shim
agrees with the thing it wraps. It is 111 lines of code plus a test, guarding a
compatibility promise made to nobody.

### A.4 Target layout

```
obs_utils/
  camera.py        ← alpyca_tools/camera_ops.py + camera_device.py + schema.py
  fits_writer.py   ← alpyca_tools/fits_writer.py  (moved verbatim)
  alpaca.py        ← obs_utils/alpaca.py + obs_utils/imaging.py, CaptureRequest dropped
  discovery.py     ← alpyca_tools/discovery.py    (moved verbatim; see A.6)
scripts/
  camera_diag.py   ← alpyca_tools/scripts/camera_diag.py
  snap_fits.py     ← alpyca_tools/scripts/snap_fits.py   (see A.6)

deleted: alpyca_tools/__init__.py, alpyca_tools/telemetry.py,
         alpyca_tools/README.md, obs_utils/fits_routine.py, obs_utils/imaging.py
```

Two judgement calls inside that layout:

- **`schema.py` folds into `camera.py`.** It is a 7-line `CameraState(IntEnum)` with
  exactly one consumer, `camera_ops.py:108`. A 12-line file for one enum used by its
  own neighbour is a file that exists because of the package split.
- **`telemetry.py` is deleted, not moved.** Its `Telemetry` class has **zero
  references** repo-wide, and its `setup_logging` (`telemetry.py:43-47`) is a
  five-line `basicConfig` that is shadowed in practice by the real
  `obs_utils.logging.setup_logging` (`logging.py:71-106`), which does session log
  paths, UTC dating, and handler de-duplication. Two functions with one name, one of
  them a strictly worse version.

### A.5 Line accounting

| | Files | Lines |
|---|---|---|
| `alpyca_tools/**.py` today | 9 | 954 |
| `obs_utils/{alpaca,imaging,fits_routine}.py` today | 3 | 443 |
| **Total in scope** | **12** | **1,397** |

| Change | Δ lines |
|---|---|
| Delete `alpyca_tools/__init__.py`; add ~14 export entries to `obs_utils/__init__.py` | −29 |
| Fold `schema.py` into `camera.py` | −4 |
| Delete `telemetry.py` (unused; duplicate `setup_logging`) | −47 |
| Delete `fits_routine.py` | −111 |
| Collapse `CaptureRequest` into `ExposureSettings`; merge `imaging.py` into `alpaca.py` | ≈ −85 |
| **Net** | **≈ −276 (−20 %)** |

Files: **12 → 7**. Directories: one fewer top-level package. If the owner also
retires the two dead-code items in A.6, the total is **≈ −421 (−30 %)** and **12 → 5
files**.

The structural win is larger than the line count. Today a single science frame is
described by two identical dataclasses across a package boundary and written through
a module that announces itself as a shim. After the merge there is one exposure
type, one FITS writer, one place a reader looks.

### A.6 Two dead-code calls for the owner (not mine to make)

- **`alpyca_tools/discovery.py` (54 lines)** — a UDP Alpaca-discovery client, **zero
  references** repo-wide. It is not a duplicate of `qhy_alpaca/src/discovery.py`
  (that is the *responder* side). Legitimately useful for bring-up on a new network;
  keep or delete on judgement, but it should not be assumed live.
- **`alpyca_tools/scripts/snap_fits.py` (91 lines)** — 44 of its 91 lines are
  `argparse` declarations that restate `ExposureSettings` and `FitsHeaderConfig`
  field-by-field. A one-brick night plan run through `execute_night.py` does the same
  job with the safety gates attached, which is the stated convention
  (`CLAUDE.md` → Conventions, "One runner"). `camera_diag.py` (50 lines) has no
  equivalent — it is a genuine no-plan bring-up probe and should be kept.

### A.7 Constraints the merge must respect

Three things must not break:

1. **`tests/test_optional_imports.py` must keep passing.** It checks that importing
   the FITS writer does not drag in the camera driver
   (`test_optional_imports.py:8-12`), which is what lets a reduction machine import it
   with no observatory dependencies installed. The deferred imports at
   `fits_writer.py:10-13` and `:339` are what make that true; they must move verbatim,
   and the new `obs_utils/__init__.py` entries must stay lazy.
2. **`poltools/io.py:19` imports `alpyca_tools.fits_writer`.** That is a reduction
   library importing an acquisition package. After the merge it becomes `poltools`
   importing `obs_utils`, which looks worse even though it is harmless
   (`obs_utils/__init__.py` is lazy, so no hardware import follows). The real problem
   is that `fits_writer.py` is misfiled: it is not an Alpaca helper, it is the POLITE
   FITS schema, used by acquisition, the simulator, and reduction alike. Report 2 §C4
   deals with it.
3. **Only two files outside `obs_utils` and `tests` need editing.** Exactly two
   consumers name `alpyca_tools`: `poltools/io.py:19` and one cell in
   `notebooks/observation_notebooks/20260730_observation.ipynb:220`. The three
   `notebooks/templates/*` control notebooks never touch it, so the byte-identical
   shared preamble guarded by `tests/test_notebook_templates.py` is unaffected.

   One of the `CaptureRequest` call sites that step 4 below would migrate,
   `obs_utils/pointing.py:60`, is in a module nothing currently calls — see Report 2
   §C0c. It is being kept for the first pointed night, so migrate it like the rest;
   just do not treat it as evidence that the code path is exercised.

### A.8 Suggested sequence

Each step leaves the suite green, so it can stop at any point.

1. `git rm obs_utils/fits_routine.py` and drop the shim half of
   `tests/test_fits_header_contract.py` (keep the `DetectorCards` contract assertions,
   retarget them at `fits_writer`). Independent of everything else; do it first.
2. `git mv alpyca_tools/fits_writer.py obs_utils/fits_writer.py`, fix 8 import sites.
3. `git mv` `camera_ops.py` → `obs_utils/camera.py`; fold in `camera_device.py` and
   `schema.py`.
4. Delete `CaptureRequest`; make `imaging.py`'s six functions take `ExposureSettings`
   directly and move them into `alpaca.py`. This is the only step that changes
   behaviour-adjacent code — `night_session.py:584` and `pointing.py:60` change type
   at the call site.
5. Delete `alpyca_tools/telemetry.py` and `alpyca_tools/__init__.py`; move the two
   scripts to `scripts/`; remove the now-empty package.
6. Update `README.md:38-45` (the Project Structure block), `CLAUDE.md` /
   `AGENTS.md` → Project Structure, and `RELICENSE-AGPLv3.md:468` coverage list.

**Verification at each step:** `pytest -q` stays at 244 passed, and
`tests/test_optional_imports.py` in particular. Nothing here touches acquisition
numerics, header values, or `FITSDATA/`.

---

## Part B — `qhy_alpaca` → `obs_utils`: not feasible

Four independent blockers. Any one of them is sufficient; the licensing one is
disqualifying on its own.

### B.1 It is not POLITE's code — and the relicense depends on that

`qhy_alpaca/` is derived from
[`ryanswindle/alpaca-qhyccd-camera`](https://github.com/ryanswindle/alpaca-qhyccd-camera),
MIT, © 2026 Ryan Swindle (`qhy_alpaca/UPSTREAM.md:3`, `qhy_alpaca/LICENSE`). The
relicense runbook already settled this and the reasoning is worth quoting, because it
was written in answer to almost the same question:

> **Modification does not transfer copyright.** Ryan Swindle remains the copyright
> holder of the upstream portions of `qhy_alpaca/` no matter how much you change; a
> derivative work carries the original author's rights forward.
> — `RELICENSE-AGPLv3.md:164-168`

The plan then builds the whole attribution mechanism on **the directory path**:

- `RELICENSE-AGPLv3.md:224` — "Leave `qhy_alpaca/LICENSE` **untouched**."
- `RELICENSE-AGPLv3.md:468-470` — AGPL headers cover `obs_utils/*.py`,
  `alpyca_tools/*.py`, …; "**Do not** add these headers to files under `qhy_alpaca/`
  that you did not substantially rewrite."
- `RELICENSE-AGPLv3.md:427-434` — `THIRD-PARTY-NOTICES.md` names `qhy_alpaca/` as the
  scope of the MIT grant and points at `qhy_alpaca/LICENSE` for full text.
- `RELICENSE-AGPLv3.md:388` — the final verification step is to confirm in a browser
  that `qhy_alpaca/LICENSE` still shows the MIT text.

Merging the tree into `obs_utils/` destroys the only thing that says which files those
notices apply to. You would end up with ~3,200 lines of another author's MIT code mixed
in with POLITE's AGPL code, a blanket rule saying to put AGPL headers on all of it, and
an attribution file pointing at a directory that no longer exists. That is a licensing
violation inside a copyleft project — which `RELICENSE-AGPLv3.md:169-171` already calls
the worst possible position to be in. **This alone settles it.**

### B.2 It is a separate process on a separate machine

Nothing in POLITE imports `qhy_alpaca`. Every reference is out-of-process:

- `obs_utils/alpaca_servers.py:159-163` launches `qhy_alpaca/src/main.py` with
  `subprocess.Popen`, its own `cwd`, and a synthesized `PYTHONPATH`
  (`alpaca_servers.py:126-136`).
- `scripts/launch_qhy_server.ps1:16` is a PowerShell shim resolving to
  `qhy_alpaca/scripts/start_qhy_alpaca_server.ps1`.
- Everything else is a comment (`obs_utils/alpaca.py:48`).

The two halves talk **HTTP on port 11112**, and `alpaca_servers.py:225-228` states the
launcher "is intentionally restricted to a Windows local-host layout." POLITE's
control code runs on macOS/Apple Silicon; `qhy_alpaca` runs on the Windows observatory
PC, next to the camera, because it `ctypes`-loads the vendor QHY SDK DLL
(`qhy_alpaca/src/libqhyccd.py:12-15`). A Python import boundary cannot express
"different machine, different OS, different process"; a directory does.

Folding a server into the client that calls it also removes the property that makes
the current design debuggable: the server can be started, killed, and conformance-
tested (`qhy_alpaca/tests/test_conformu.py`) without POLITE running at all.

### B.3 Its dependency set is disjoint from POLITE's

`qhy_alpaca/requirements.txt` pulls **fastapi, fastapi[standard], uvicorn, pydantic,
pydantic-settings, loguru**. Grepping the repo, every one of those appears **only**
under `qhy_alpaca/src/` — POLITE proper uses none of them. It also ships its own
`Dockerfile` and `.gitignore`, i.e. it is packaged as a deployable, not a library.

Merging would put a web framework and an ASGI server into the import path of a
reduction package that today needs numpy, astropy, and alpyca. Reduction runs on
machines where the QHY server is neither installed nor wanted.

### B.4 It is not a Python package, and three module names collide

`qhy_alpaca/src/` has **no `__init__.py`**, and its 12 modules import each other with
**flat absolute imports** — 21 of them across 8 files:

```19:21:qhy_alpaca/src/camera_device.py
from config import DeviceConfig
from libqhyccd import QHY_CONTROL, QHY_GPS, QHY_SUCCESS, QHY_ERROR, load_qhyccd_library
from log import get_logger
```

That is *why* `alpaca_servers.py:129-135` has to put `qhy_alpaca/src` on
`PYTHONPATH`. Merging means rewriting all 21 to relative imports — i.e. editing the
MIT-derived third-party files that B.1 says to leave alone, which would also make
future comparison against upstream impossible.

And three names collide head-on with the merged target:

| `qhy_alpaca/src/` | collides with | note |
|---|---|---|
| `config.py` (89) | `obs_utils/config.py` (99) | server device profile vs. client dataclass configs |
| `camera_device.py` (1,150) | `alpyca_tools/camera_device.py` (9) | QHY SDK driver vs. a 3-line Alpaca HTTP subclass |
| `discovery.py` (37) | `alpyca_tools/discovery.py` (54) | discovery **responder** vs. discovery **client** |

The `camera_device` pair is the telling one. The same name means "the thing holding
the QHY SDK handle" on one side and "an HTTP client stub" on the other. Merging forces
a rename of one of them, and whichever way it goes the name gets worse. Two names that
cannot both be right in one namespace is the definition of a boundary that is doing
work.

### B.5 What to do with `qhy_alpaca` instead

The directory should stay exactly where it is. Three small, real cleanups are
available and none of them require a merge:

1. **Name it for what it is.** Rename `qhy_alpaca/` → `vendor/qhy_alpaca/` or
   `third_party/qhy_alpaca/` so the tree itself says "different copyright lives here."
   This *strengthens* `RELICENSE-AGPLv3.md` Step 2 rather than weakening it. Cost: one
   `git mv`, plus `alpaca_servers.py:237` and the `.ps1` shim. **Owner call** — it was
   deliberately moved *out* of `third_party/` in an earlier session
   (`qhy_alpaca/UPSTREAM.md:5`) on the reasoning that POLITE now maintains it, so
   reversing that is a decision, not hygiene.
2. **`qhy_alpaca/tests/{test.py,test_conformu.py}` are untracked** (confirmed via
   `git ls-files`), so a `git pull` on the observatory PC does not deliver the
   conformance test for the server it is running. Either track them or delete them.
3. **Retire the `.ps1` shim.** `scripts/launch_qhy_server.ps1` exists only to resolve
   a path into `qhy_alpaca/scripts/`. Amendment A.3 already fixed the *name* collision
   (`RELICENSE-AGPLv3.md:722-733`); the remaining question is whether a 17-line
   path-resolver earns a file, or whether `scripts/README.md` should just name the
   real launcher. Low priority.

---

## Summary

**Merge `alpyca_tools` into `obs_utils`.** It is one body of code split under two
names, and the split costs us a 13-field dataclass defined twice, a 43-line export
table nothing uses, a 111-line compatibility module for callers deleted four days ago,
and a logging helper that duplicates a better one. About **276 lines and 5 files** go
away — more if you also retire the two dead-code items in §A.6 — and
`RELICENSE-AGPLv3.md:468` has one fewer directory to list.

**Leave `qhy_alpaca` where it is.** It would delete lines, so it passes the criteria on
a literal reading, but it fails on intent. That directory boundary is currently doing
four jobs at once: it separates someone else's copyright from ours, one operating
system from another, one process from another, and one dependency set from another.
Removing it means editing the exact files the relicense plan tells us not to touch. It
is not clutter; it is where the deployment actually splits.
