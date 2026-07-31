"""The control-notebook template family stays runnable and stays consistent.

Notebooks are not executed by CI, so nothing else catches a template that was
hand-edited into a syntax error or one whose preamble drifted away from its
siblings -- and a control notebook only gets discovered broken at the telescope.
"""

from __future__ import annotations

import ast
import json
import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
TEMPLATE_DIR = ROOT / "notebooks" / "templates"
TEMPLATES = sorted(TEMPLATE_DIR.glob("*.ipynb"))

# The family the streamline plan calls for; a missing one means an operator
# opens the directory and finds no starting point for that kind of night.
EXPECTED = {"night.ipynb", "devices.ipynb", "capture.ipynb", "inspect.ipynb"}

# The starter is intentionally motion-free. Palette motion is allowed, but it
# must be obvious at the top of the cell before a user runs it at the telescope.
MOTION_SYMBOLS = (
    "slew_radec_j2000", "slew_altaz", "mount_park", "home_mount",
    "mount_offset", "spiral_offset", "s.focus(", "focus_relative(",
    "field_rotator_goto", "field_rotator_offset", "s.hwp(", "s.filter(",
    "home_hwp", "s.expose(", "s.warm_up(", "mount_tracking_",
    "mount_spiral_offset", "mount_stop", "mount_park",
)


def _cells(path: Path, kind: str = "code"):
    nb = json.loads(path.read_text())
    return ["".join(c["source"]) for c in nb["cells"] if c["cell_type"] == kind]


def _strip_magics(source: str) -> str:
    """IPython line magics and shell escapes are not Python; blank them out."""
    return "\n".join(
        "pass" if line.lstrip().startswith(("%", "!")) else line
        for line in source.splitlines()
    )


def test_template_family_is_present():
    assert {p.name for p in TEMPLATES} == EXPECTED


@pytest.mark.parametrize("path", TEMPLATES, ids=lambda p: p.name)
def test_template_is_valid_nbformat_4(path):
    nb = json.loads(path.read_text())
    assert nb["nbformat"] == 4
    for cell in nb["cells"]:
        assert cell["cell_type"] in ("code", "markdown")
        assert isinstance(cell["source"], list)


@pytest.mark.parametrize("path", TEMPLATES, ids=lambda p: p.name)
def test_every_code_cell_parses(path):
    for i, source in enumerate(_cells(path)):
        try:
            ast.parse(_strip_magics(source))
        except SyntaxError as exc:  # pragma: no cover - failure detail
            pytest.fail(f"{path.name} code cell {i}: {exc}")


def test_shared_preamble_is_identical_across_templates():
    """The 'one agreed-upon import set' only holds if it is literally one set."""
    preambles = {p.name: _cells(p)[:2] for p in TEMPLATES}
    first = next(iter(preambles.values()))
    for name, preamble in preambles.items():
        assert preamble == first, f"{name} preamble drifted from the family"


def test_preamble_bootstraps_the_repo_root_and_imports_live():
    boot, imports = _cells(TEMPLATES[0])[:2]
    assert "sys.path.insert" in boot, "bootstrap must make obs_utils importable"
    assert "from obs_utils import live" in imports
    assert "SESSION_DIR" in imports, "templates key off a single session directory"


@pytest.mark.parametrize("path", TEMPLATES, ids=lambda p: p.name)
def test_templates_ship_unexecuted(path):
    """Committed outputs are stale the moment they are written, and a template
    carrying last night's numbers invites reading them as tonight's."""
    nb = json.loads(path.read_text())
    for i, cell in enumerate(nb["cells"]):
        if cell["cell_type"] != "code":
            continue
        assert cell.get("outputs") == [], f"{path.name} cell {i} has saved output"
        assert cell.get("execution_count") is None


def test_night_starter_is_motion_free_and_palette_motion_is_marked():
    for path in TEMPLATES:
        for source in _cells(path):
            moves = any(symbol in source for symbol in MOTION_SYMBOLS)
            if path.name == "night.ipynb":
                assert not moves, f"{path.name} must not command motion: {source}"
            elif moves:
                assert source.startswith("# MOTION"), (
                    f"{path.name} motion cell must start with # MOTION: {source}"
                )


@pytest.mark.parametrize("path", TEMPLATES, ids=lambda p: p.name)
def test_palette_cells_stay_short(path):
    """Palette cells must remain easy to read and copy at the telescope."""
    for i, source in enumerate(_cells(path)):
        assert len(source.splitlines()) <= 8, f"{path.name} code cell {i} is too long"


def test_live_symbols_used_by_templates_exist():
    """A renamed helper must fail here, not in a cell at 3 a.m."""
    from obs_utils import live

    used = set()
    for path in TEMPLATES:
        for source in _cells(path):
            for name in dir(live):
                if name.startswith("_"):
                    continue
                if f"live.{name}" in source:
                    used.add(name)
    assert {"watch", "frame_report", "session_table", "qa_print"} <= used
    for name in used:
        assert hasattr(live, name)


def test_interactive_symbols_used_by_templates_exist():
    """The notebooks may only name public module helpers that actually exist."""
    from obs_utils import interactive

    used = set()
    for path in TEMPLATES:
        for source in _cells(path):
            used.update(re.findall(r"\\bobs\\.([A-Za-z_]\\w*)", source))
    for name in used:
        assert hasattr(interactive, name), f"{path.name} calls missing obs.{name}"
