#!/usr/bin/env python3
"""Generate printable on-site observatory checklists (B&W, letter size).

Editable Markdown sources in the repo root; run this script to rebuild PDFs:

    python scripts/generate_site_checklists.py
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

from fpdf import FPDF
from fpdf.enums import XPos, YPos

ROOT = Path(__file__).resolve().parents[1]

FONT = "Times"
BOX = 0.15
BOX_GAP = 0.11
ROW = 0.19
ITEM_PAD = 0.05
SECTION_GAP = 0.1
CHECK_COL = 0.28
INK = (0, 0, 0)
RULE = (140, 140, 140)
PANEL = (235, 235, 235)
BANNER = (220, 220, 220)


MARGIN = 0.32
INNER_GAP = 0.10
FOOTER_BAND = 0.30


def _frame_geometry(pdf: ChecklistPDF) -> dict[str, float]:
    w, h = pdf.w, pdf.h
    outer_l = MARGIN
    outer_t = MARGIN
    outer_r = w - MARGIN
    outer_b = h - MARGIN
    inner_l = outer_l + INNER_GAP
    inner_t = outer_t + INNER_GAP
    inner_r = outer_r - INNER_GAP
    inner_b = outer_b - FOOTER_BAND
    return {
        "outer_l": outer_l,
        "outer_t": outer_t,
        "outer_r": outer_r,
        "outer_b": outer_b,
        "inner_l": inner_l,
        "inner_t": inner_t,
        "inner_r": inner_r,
        "inner_b": inner_b,
    }


class ChecklistPDF(FPDF):
  def footer(self) -> None:
    g = _frame_geometry(self)
    rule_y = g["inner_b"]
    self.set_draw_color(*RULE)
    self.set_line_width(0.003)
    self.line(g["inner_l"], rule_y + 0.028, g["inner_r"], rule_y + 0.028)
    text_y = rule_y + (g["outer_b"] - rule_y - 0.1) / 2
    self.set_y(text_y)
    self.set_font(FONT, "", 7.5)
    self.set_text_color(80, 80, 80)
    self.cell(
      0,
      0.1,
      f"POLITE Observatory  |  Site Operations  |  Page {self.page_no()}",
      align="C",
    )


def _page_frame(pdf: ChecklistPDF) -> None:
  g = _frame_geometry(pdf)
  outer_w = g["outer_r"] - g["outer_l"]
  outer_h = g["outer_b"] - g["outer_t"]
  inner_w = g["inner_r"] - g["inner_l"]
  inner_h = g["inner_b"] - g["inner_t"]

  pdf.set_draw_color(*INK)
  pdf.set_line_width(0.02)
  pdf.rect(g["outer_l"], g["outer_t"], outer_w, outer_h)
  pdf.set_line_width(0.006)
  pdf.rect(g["inner_l"], g["inner_t"], inner_w, inner_h)

  tick = 0.08
  for ox in (g["inner_l"], g["inner_r"]):
    oy = g["inner_t"]
    pdf.set_line_width(0.012)
    pdf.line(ox - tick, oy, ox + tick, oy)
    pdf.line(ox, oy - tick, ox, oy + tick)


def _ornament_rule(pdf: ChecklistPDF, y: float, half_width: float = 2.35) -> None:
  cx = pdf.w / 2
  pdf.set_draw_color(*INK)
  pdf.set_line_width(0.018)
  pdf.line(cx - half_width, y, cx + half_width, y)
  pdf.set_draw_color(*RULE)
  pdf.set_line_width(0.004)
  pdf.line(cx - half_width - 0.15, y + 0.035, cx + half_width + 0.15, y + 0.035)
  pdf.line(cx - half_width - 0.15, y - 0.02, cx + half_width + 0.15, y - 0.02)


def _official_header(pdf: ChecklistPDF, title: str) -> None:
  x0 = pdf.l_margin
  width = pdf.w - pdf.l_margin - pdf.r_margin
  y0 = pdf.t_margin - 0.02
  banner_h = 0.54

  pdf.set_fill_color(*BANNER)
  pdf.set_draw_color(*INK)
  pdf.set_line_width(0.01)
  pdf.rect(x0, y0, width, banner_h, style="DF")

  pdf.set_xy(x0 + 0.12, y0 + 0.1)
  pdf.set_font(FONT, "B", 8.5)
  pdf.set_text_color(60, 60, 60)
  pdf.cell(width - 0.24, 0.12, "POLITE OBSERVATORY", align="C")

  pdf.set_xy(x0 + 0.12, y0 + 0.28)
  pdf.set_font(FONT, "B", 14)
  pdf.set_text_color(*INK)
  pdf.cell(width - 0.24, 0.2, title, align="C")

  pdf.set_y(y0 + banner_h + 0.04)
  _ornament_rule(pdf, pdf.get_y())
  pdf.ln(0.16)


def _meta_row(pdf: ChecklistPDF) -> None:
  x0 = pdf.l_margin
  width = pdf.w - pdf.l_margin - pdf.r_margin
  y0 = pdf.get_y()
  pdf.set_fill_color(*PANEL)
  pdf.set_draw_color(*INK)
  pdf.set_line_width(0.006)
  pdf.rect(x0, y0, width, 0.3, style="DF")

  gap = 0.3
  field_w = (width - 0.24 - gap) / 2
  for i, label in enumerate(("Date", "Observer")):
    x = x0 + 0.12 + i * (field_w + gap)
    pdf.set_xy(x, y0 + 0.08)
    pdf.set_font(FONT, "B", 9)
    pdf.set_text_color(*INK)
    pdf.cell(0.55, 0.14, f"{label}:")
    pdf.set_font(FONT, "", 9)
    line_y = y0 + 0.22
    pdf.set_line_width(0.005)
    pdf.line(x + 0.58, line_y, x + field_w, line_y)

  pdf.set_y(y0 + 0.38)


def _section_block(pdf: ChecklistPDF, title: str, x: float, width: float) -> None:
  pdf.ln(SECTION_GAP)
  # Keep the header with at least its first row so it is never orphaned at the
  # foot of a page (0.24 header band + ~0.30 for one item row).
  _ensure_space(pdf, 0.24 + 0.30)
  y = pdf.get_y()
  pdf.set_fill_color(*INK)
  pdf.set_draw_color(*INK)
  pdf.set_line_width(0.006)
  pdf.rect(x, y, width, 0.24, style="F")
  pdf.set_xy(x + 0.1, y + 0.055)
  pdf.set_font(FONT, "B", 10)
  pdf.set_text_color(255, 255, 255)
  pdf.cell(width - 0.2, 0.14, title.upper(), align="L")
  pdf.set_y(y + 0.26)


def _ensure_space(pdf: ChecklistPDF, needed_h: float) -> None:
  """Start a fresh framed page if ``needed_h`` will not fit before the footer.

  Only active when auto page break is disabled (the single-column Markdown
  checklists paginate manually because rows are drawn with fixed-coordinate
  borders that fpdf's automatic break would desync from the text ``multi_cell``).
  """
  if pdf.auto_page_break:
    return
  g = _frame_geometry(pdf)
  if pdf.get_y() + needed_h > g["inner_b"] - 0.08:
    pdf.add_page()
    _page_frame(pdf)
    pdf.set_y(g["inner_t"] + 0.18)


def _text_line_count(pdf: ChecklistPDF, text: str, width: float) -> int:
    pdf.set_font(FONT, "", 10.5)
    words = text.split()
    if not words:
        return 1
    lines = 1
    current = words[0]
    for word in words[1:]:
        trial = f"{current} {word}"
        if pdf.get_string_width(trial) <= width:
            current = trial
        else:
            lines += 1
            current = word
    return lines


def _item_row(
  pdf: ChecklistPDF,
  text: str,
  x: float,
  width: float,
  *,
  last: bool = False,
  checked: bool = False,
) -> None:
    text_w = width - CHECK_COL - BOX_GAP - 0.08
    lines = _text_line_count(pdf, text, text_w)
    row_h = max(0.28, 0.1 + lines * ROW)
    _ensure_space(pdf, row_h)
    y0 = pdf.get_y()
    pdf.set_draw_color(*RULE)
    pdf.set_line_width(0.004)
    pdf.line(x, y0, x + width, y0)
    if last:
        pdf.line(x, y0 + row_h, x + width, y0 + row_h)
    pdf.line(x, y0, x, y0 + row_h)
    pdf.line(x + CHECK_COL, y0, x + CHECK_COL, y0 + row_h)
    pdf.line(x + width, y0, x + width, y0 + row_h)

    check_x = x + (CHECK_COL - BOX) / 2
    check_y = y0 + (row_h - BOX) / 2
    pdf.set_draw_color(*INK)
    pdf.set_line_width(0.01)
    pdf.rect(check_x, check_y, BOX, BOX)
    if checked:
        pdf.set_line_width(0.012)
        pdf.line(check_x + 0.03, check_y + 0.04, check_x + BOX - 0.03, check_y + BOX - 0.04)
        pdf.line(check_x + 0.03, check_y + BOX - 0.04, check_x + BOX - 0.03, check_y + 0.04)

    pdf.set_xy(x + CHECK_COL + BOX_GAP, y0 + 0.045)
    pdf.set_font(FONT, "", 10.5)
    pdf.set_text_color(*INK)
    pdf.multi_cell(text_w, ROW, text, align="L")
    pdf.set_y(y0 + row_h + ITEM_PAD)


def _render_section_checked(
  pdf: ChecklistPDF,
  title: str,
  items: Sequence[tuple[str, bool]],
  x: float,
  width: float,
) -> None:
  _section_block(pdf, title, x, width)
  for i, (text, checked) in enumerate(items):
    _item_row(pdf, text, x, width, last=(i == len(items) - 1), checked=checked)


@dataclass
class _MdItem:
  text: str
  checked: bool = False


@dataclass
class _MdSection:
  title: str
  items: list[_MdItem] = field(default_factory=list)


def _strip_md_inline(text: str) -> str:
  text = re.sub(r"\*\*([^*]+)\*\*", r"\1", text)
  text = text.replace("`", "")
  return text.strip()


def parse_checklist_markdown(path: Path) -> tuple[str, str, list[_MdSection]]:
  """Parse a simple checklist Markdown file (# title, ## sections, - [ ] items)."""
  title = path.stem.replace("_", " ").title()
  subtitle_parts: list[str] = []
  sections: list[_MdSection] = []
  current: _MdSection | None = None

  for raw in path.read_text(encoding="utf-8").splitlines():
    line = raw.rstrip()
    if not line:
      continue
    if line.startswith("# "):
      title = _strip_md_inline(line[2:])
      continue
    if line.startswith("## "):
      current = _MdSection(title=_strip_md_inline(line[3:]))
      sections.append(current)
      continue
    if current is None:
      subtitle_parts.append(_strip_md_inline(line))
      continue
    if line.startswith("- [x] "):
      current.items.append(_MdItem(_strip_md_inline(line[6:]), checked=True))
    elif line.startswith("- [ ] "):
      current.items.append(_MdItem(_strip_md_inline(line[6:]), checked=False))
    elif line.startswith("- "):
      current.items.append(_MdItem(_strip_md_inline(line[2:]), checked=False))

  return title, " ".join(subtitle_parts), sections


def checklist_pdf_from_markdown(md_path: Path) -> Path:
  """Render a Markdown checklist to a sibling PDF in the same directory."""
  title, subtitle, sections = parse_checklist_markdown(md_path)
  pdf = ChecklistPDF("P", "in", "Letter")
  # Manual pagination: rows have fixed-coordinate borders, so an automatic break
  # inside a text multi_cell would desync borders from text. _ensure_space adds
  # framed continuation pages proactively instead.
  pdf.set_auto_page_break(False)
  pdf.set_margins(0.72, 0.62, 0.72)
  pdf.add_page()
  _page_frame(pdf)

  _official_header(pdf, title)
  if subtitle:
    x = pdf.l_margin
    width = pdf.w - pdf.l_margin - pdf.r_margin
    pdf.set_x(x)
    pdf.set_font(FONT, "I", 9)
    pdf.set_text_color(60, 60, 60)
    pdf.multi_cell(width, 0.14, subtitle, align="L")
    pdf.ln(0.08)

  x = pdf.l_margin
  width = pdf.w - pdf.l_margin - pdf.r_margin
  for section in sections:
    pairs = [(item.text, item.checked) for item in section.items]
    _render_section_checked(pdf, section.title, pairs, x, width)

  out = md_path.with_suffix(".pdf")
  pdf.output(str(out))
  return out


def _render_section(
  pdf: ChecklistPDF,
  title: str,
  items: Sequence[str],
  x: float,
  width: float,
) -> None:
  _section_block(pdf, title, x, width)
  for i, text in enumerate(items):
    _item_row(pdf, text, x, width, last=(i == len(items) - 1))


def _render_column(
  pdf: ChecklistPDF,
  title: str,
  items: Sequence[str],
  x: float,
  width: float,
  y_start: float,
) -> float:
  pdf.set_xy(x, y_start)
  _section_block(pdf, title, x, width)
  for i, text in enumerate(items):
    _item_row(pdf, text, x, width, last=(i == len(items) - 1))
  return pdf.get_y()


def _column_divider(pdf: ChecklistPDF, x: float, y0: float, y1: float) -> None:
  pdf.set_draw_color(*INK)
  pdf.set_line_width(0.008)
  pdf.line(x, y0, x, y1)
  pdf.set_draw_color(*RULE)
  pdf.set_line_width(0.003)
  pdf.line(x + 0.03, y0, x + 0.03, y1)
  pdf.line(x - 0.03, y0, x - 0.03, y1)


def wire_cable_checklist() -> Path:
  pdf = ChecklistPDF("P", "in", "Letter")
  pdf.set_auto_page_break(auto=True, margin=MARGIN + FOOTER_BAND + 0.12)
  pdf.set_margins(0.72, 0.62, 0.72)
  pdf.add_page()
  _page_frame(pdf)

  _official_header(pdf, "Wire / Cable Checklist")

  x = pdf.l_margin
  width = pdf.w - pdf.l_margin - pdf.r_margin
  devices = [
    ("Detector", ["USB B to USB A cable", "12 V DC power supply"]),
    ("EFW", ["USB B to USB A cable"]),
    ("HWPR", ["12 V DC power supply", "RJ12 serial to USB A cable"]),
    ("PWI Rotator", ["RJ45 serial to USB A cable"]),
    ("PWI Focuser", ["RJ45 serial to USB A cable"]),
        ("Shared Cabling", [
            "3x USB A to USB A extender",
            "USB A hub",
        ]),
  ]
  for name, cables in devices:
    _render_section(pdf, name, cables, x, width)

  out = ROOT / "wire_cable_checklist.pdf"
  pdf.output(str(out))
  return out


def startup_shutdown_checklist() -> Path:
  pdf = ChecklistPDF("P", "in", "Letter")
  pdf.set_auto_page_break(auto=True, margin=MARGIN + FOOTER_BAND + 0.12)
  pdf.set_margins(0.72, 0.62, 0.72)
  pdf.add_page()
  _page_frame(pdf)

  _official_header(pdf, "Startup / Shutdown Routine")
  _meta_row(pdf)

  content_w = pdf.w - pdf.l_margin - pdf.r_margin
  gutter = 0.22
  col_w = (content_w - gutter) / 2
  right_x = pdf.l_margin + col_w + gutter
  y0 = pdf.get_y()

  startup = (
    "Plug in all cables and 12 V supplies",
    "Floor surge protector ON",
    "Mount power ON",
    "Observatory PC ON",
    "Open PWI4 GUI",
    "Open Interactive Control notebook",
    "Run connect / startup cells",
    "Home mount; connect rotator and focuser; confirm pointing model loaded",
    "Set camera cooler to target temperature",
    "Begin observations",
  )
  shutdown = (
    "Stop exposures and notebook cells",
    "Run shutdown cell",
    "Turn off cooler",
    "Park mount; disconnect mount",
    "Close PWI4",
    "Mount power OFF",
    "Observatory PC OFF",
    "Floor surge protector OFF",
    "Unplug USB and 12 V cables",
    "Cover equipment with tarp",
    "Close roof",
    "Lock shed",
  )

  y_left = _render_column(pdf, "Startup", startup, pdf.l_margin, col_w, y0)
  y_right = _render_column(pdf, "Shutdown", shutdown, right_x, col_w, y0)
  _column_divider(pdf, pdf.l_margin + col_w + gutter / 2, y0, max(y_left, y_right))

  out = ROOT / "startup_shutdown_checklist.pdf"
  pdf.output(str(out))
  return out


def before_observations_checklist() -> Path:
  return checklist_pdf_from_markdown(ROOT / "before_observations_checklist.md")


def salvage_no_pointing_checklist() -> Path:
  return checklist_pdf_from_markdown(ROOT / "salvage_no_pointing_checklist.md")


def main() -> None:
  for path in (
    wire_cable_checklist(),
    startup_shutdown_checklist(),
    before_observations_checklist(),
    salvage_no_pointing_checklist(),
  ):
    print(path)


if __name__ == "__main__":
  main()
