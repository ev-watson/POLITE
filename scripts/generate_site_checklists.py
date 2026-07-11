#!/usr/bin/env python3
"""Generate printable on-site observatory checklists (B&W, letter size).

Editable Markdown sources in the repo root; run this script to rebuild PDFs:

    python scripts/generate_site_checklists.py
"""

from __future__ import annotations

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


def checklist_pdf_from_sections(
  title: str,
  subtitle: str,
  sections: Sequence[tuple[str, Sequence[tuple[str, bool]]]],
  out: Path,
) -> Path:
  """Render a checklist PDF from inline (section-title, [(item, checked)]) data.

  Self-contained: checklist content is defined in code (the builder functions
  below), so no external Markdown source files are read.
  """
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
  for section_title, items in sections:
    _render_section_checked(pdf, section_title, list(items), x, width)

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
  title = "Before Starting Observations"
  subtitle = (
    "First light 2026-07-09. Observatory Windows PC (PWI4 :8220, ASCOM Alpaca "
    ":11111). Drive the night from night_plans/20260709.yaml into "
    "FITSDATA/20260709/. Dry-run preview: 270 frames, 3 QA gates, 1.01 h "
    "open-shutter."
  )
  sections = [
    ("Already done (software / lab)", [
      ("Night plan YAML, palette bricks, QA hooks, polite naming (dry-run: 270 frames, 3 QA gates)", True),
      ("FITS provenance wired: EGAIN, GAIN, READMODE, OFFSET, SET-TEMP, INSTROT, HWPANG", True),
      ("pol_config.yaml sidecar + block_manifest.jsonl written by the night runner", True),
      ("HWP hardware test: 8 frames in FITSDATA/hwp_test/; no pixels pinned at 0", True),
      ("Reduction stack: poltools lsq (default) + double_ratio cross-check available", True),
    ]),
    ("Python environment (observatory PC)", [
      ("venv created with Python 3.13 and activated", False),
      ("pip install of science + device packages (numpy scipy astropy photutils matplotlib pyyaml alpyca requests astroquery) succeeds", False),
      ("caltools and poltools installed editable (pip install -e)", False),
      ("Import check prints \"env ok\" (alpaca = alpyca ASCOM client; astroquery = Horizons)", False),
      ("obs_utils / alpyca_tools import in place from repo root - run all scripts from repo root, no install needed", False),
    ]),
    ("Drivers and device software (observatory PC)", [
      ("ASCOM Platform 6.6+ present - confirm version", False),
      ("ASCOM Remote Server installed and exposing devices on Alpaca :11111 (separate from the Platform)", False),
      ("QHY camera driver (QHYCCD All-In-One / SDK) + ASCOM QHY driver", False),
      ("ZWO EFW filter-wheel driver + ASCOM driver", False),
      ("Optec Pyxis rotator ASCOM driver (drives the HWP rotator)", False),
      ("PWI4 (PlaneWave) installed; listening on :8220; pointing model file present", False),
      ("USB-serial driver (FTDI / Prolific) for the Pyxis RJ12-to-USB link", False),
    ]),
    ("Sync code to Windows", [
      ("Latest POLITE repo on the observatory PC (git pull)", False),
      ("QHY Alpaca server deps installed (scripts\\install_qhy_alpaca_deps.ps1)", False),
      ("astroquery import works (automated asteroid ephemerides), or JPL Horizons manual fallback ready", False),
    ]),
    ("QHY268M SDK-direct camera (bypasses broken ASCOM QHY driver)", [
      ("Scan lists QHY268M (set $env:QHYCCD_DLL if needed)", False),
      ("Camera server responds on :11112", False),
      ("qhy_alpaca_smoke_test.py writes a FITS", False),
      ("observatory_smoke_test.py passes (needs PWI4 + Remote Server for EFW/Pyxis)", False),
    ]),
    ("ASCOM / Alpaca bring-up (EFW + Pyxis on :11111)", [
      ("ASCOM Remote Server running; EFW + Pyxis on Alpaca :11111 (QHY camera is on :11112)", False),
      ("PWI4 GUI running (:8220); mount homed; pointing model loaded; rotator + focuser connected", False),
      ("Pyxis HWP reachable as an Alpaca rotator; EFW initialized (V and R slots correct)", False),
      ("user_config.py device indices match the Remote Server (camera / wheel / rotator)", False),
      ("Observatory PC clock is NTP-synced (DATE-OBS and asteroid ephemerides depend on it; timing cards read the clock)", False),
      ("python scripts/observatory_smoke_test.py - home, slew, HWP move, one FITS", False),
    ]),
    ("Detector settings (lock for the whole night)", [
      ("Gain 0 for the entire night - NOT the gain 30 used on the HWP bench test. EGAIN=1.0 e-/ADU and RON=3.5 e- are only valid at Mode 0, gain 0", False),
      ("Readout Mode 0, offset 30 (matches plan camera block and 2026-03 bench)", False),
      ("Cooler to -20 C; wait for stabilization (CCD-TEMP within ~1 C of SET-TEMP)", False),
      ("5 bias frames: min ADU > 0 (no pixels pinned at 0), sensible pedestal from offset 30", False),
      ("Brightest standards do not saturate - keep peak below ~60% FWC (~30 kADU); short exposures already set for gamma Boo and HD 154445", False),
      ("HWP backlash / settle values set from bench", False),
    ]),
    ("Dry run (no hardware)", [
      ("python scripts/plan_night.py night_plans/20260709.yaml", False),
      ("Review: 270 frames, 3 QA gates, camera block gain=0 offset=30 cooler=-20 C, ends with \"(dry-run; pass --run to execute)\"", False),
      ("Dry run does NOT connect, slew, move HWP, expose, or run QA on FITS", False),
    ]),
    ("After dry run - before roof", [
      ("FITSDATA/ exists and is writable on Windows", False),
      ("Moon check for 2026-07-09: phase and altitude; keep targets >~30 deg from the Moon (block log records MOONSEP)", False),
      ("Twilight flat exposure tuning ready (adjust exp between HWP angle sets; target 15-35 kADU, 30-60% FWC)", False),
      ("Focus per filter (~20:40) - manual", False),
      ("Do NOT change gain / readout mode / offset after bias QA passes - it invalidates masters and the CMOS error model", False),
    ]),
    ("Run the night - CORE dataset (must-get)", [
      ("python scripts/plan_night.py night_plans/20260709.yaml --run", False),
      ("gamma Boo: confirm focus AND that BOTH Savart beams are visible/paired before science", False),
      ("HD 154892 (unpolarized), then HD 154445 (polarized) - polV8 in V", False),
      ("After HD 154445: first_light_qa reduces it (reference P=3.67%, PA=88.6 deg); reduce and confirm BEFORE rotating", False),
      ("MANUAL rotator repeat: rotate PWI4 field rotator +45 deg, recenter HD 154445, run the polV8_3s repeat (POLSEQ HD154445_polV8_rot45)", False),
      ("Coord-transform check: detector-frame q,u SHOULD change; sky-frame P,PA should MATCH the first run (both near ref). Mismatch = sign / WCS / beam-label / HWP-zero / rotator-convention error - note it, keep observing", False),
      ("Matching darks (darks30 + darks_short) captured - core dataset is now self-contained", False),
      ("End of night: sequence_audit runs automatically (HWP angle-set completeness per POLSEQ)", False),
    ]),
    ("Run the night - OPTIONAL (only if sky holds and time remains)", [
      ("Priority order: HD 161056, BD+32 3739, HD 204827 (+R), HD 212311, Melpomene, Juno, Hiltner 960", False),
      ("Skip freely if high clouds come in (common after 01:00-02:00). A complete core dataset beats a half-finished long one", False),
      ("Extra time -> repeat HD 154445 / HD 154892 or take more darks/flats rather than debugging the pipeline in the dark", False),
    ]),
    ("Minimum success (this is the bar for first light)", [
      ("Detector passes bias / RON sanity check", False),
      ("V-band HWP flats acquired", False),
      ("Both Savart beams automatically detected and stay paired through the HWP sequence", False),
      ("HD 154892 reduces to low polarization", False),
      ("HD 154445 shows clear 4-theta modulation", False),
      ("lsq and double_ratio give consistent q, u", False),
      ("Rotator +45 repeat of HD 154445 gives consistent sky-frame P, PA", False),
      ("Pipeline produces q, u, P, PA and uncertainties with no manual intervention", False),
    ]),
    ("Stretch goals (NOT required tonight; defer to next run)", [
      ("Four polarized + four unpolarized standards; polarimetric efficiency; instrumental-polarization model", False),
      ("Asteroid polarimetry (Melpomene, Juno); R-band calibration; dawn characterization + PTC ladder", False),
      ("Publication-quality uncertainties", False),
    ]),
    ("First-light field card (tape to the console)", [
      ("Never stop collecting data because the reduction looks wrong - the sky is the scarce resource, not the pipeline", False),
      ("Do NOT change gain / offset / mode / HWP-zero / focus / rotator calibration mid-night unless hardware is clearly broken", False),
      ("Preserve raw FITS: never overwrite, rename, crop, or preprocess in place", False),
      ("Log one-line breadcrumbs with UT: \"22:43 possible wrong beam\", \"target drifted\", \"cloud\"", False),
      ("If one reduction fails, move on and keep collecting standards / darks / flats", False),
      ("Before shutdown, verify only: all FITS exist, logs saved, calibration frames (bias/flats/darks) taken", False),
    ]),
  ]
  return checklist_pdf_from_sections(title, subtitle, sections, ROOT / "before_observations_checklist.pdf")


def salvage_no_pointing_checklist() -> Path:
  title = "Salvage Night - No Telescope Pointing"
  subtitle = (
    "DEC drive is dead - engaging DEC auto-disconnects the mount, so there is "
    "NO slewing, NO pointing, and NO manual DEC nudge. RA can only be engaged "
    "to hold/limit RA drift, not to slew. You cannot choose a target: you "
    "observe whatever pair is already in the field. Shift the night from SKY "
    "commissioning to INSTRUMENT commissioning. Salvage goal is qualitative "
    "only - is HWP modulation present, can the beam pair be tracked "
    "frame-by-frame, does the detector-frame Stokes vector transform correctly "
    "when the whole polarimeter is rotated. NOT calibrated P or PA (that needs "
    "a working mount). Focus working means the optics, camera, and Savart "
    "splitter are almost certainly fine - that is the encouraging part."
  )
  sections = [
    ("Document the failure first (before it changes or is forgotten)", [
      ("DEC motor: completely dead, or intermittent? Note exact symptom", False),
      ("Confirm behavior: does engaging DEC still auto-disconnect the mount every time?", False),
      ("Manual DEC slew: no response / disconnects / other - record what happens", False),
      ("RA: does engaging RA hold the field (cancel sidereal drift) or only lock the axis? Note the residual drift you actually see", False),
      ("PWI4 error messages: copy verbatim + screenshot", False),
      ("Time (UTC), camera temp, and sky conditions at the time of failure", False),
    ]),
    ("Preserve what you already have", [
      ("Keep every focused doubled-star frame already on disk", False),
      ("Do NOT overwrite, rename, crop, or reprocess existing frames in place", False),
      ("Copy the existing frames to a second location NOW, before collecting more", False),
    ]),
    ("Camera calibration (needs no pointing - do this regardless of the mount)", [
      ("25-50 bias frames at the final camera mode / gain / offset / temp (Mode 0, gain 0, offset 30, -20 C)", False),
      ("Matched darks for EVERY exposure time used tonight - include the very short drift exposures", False),
      ("V-band flats through the full optical train at HWP 0, 22.5, 45, 67.5 deg; at least 10 frames per angle; do NOT change camera settings", False),
      ("Flats source: twilight sky at the current fixed pointing, or a dome / flat-panel screen if reachable (sky flats do not need tracking; drift actually averages out stars)", False),
    ]),
    ("Instrument commissioning (highest salvage value)", [
      ("Confirm BOTH Savart beams are present for the pair(s) currently in the field", False),
      ("Measure beam separation and orientation (PA) - repeat at a few field positions if more than one pair is available, to map any variation across the detector", False),
      ("Confirm HWP rotation works: run scripts/hwp_modulation_test.py (or step the HWP through the full sequence and confirm the flux ratio modulates)", False),
      ("Plate scale from a known star pair, if short exposures are clean enough to centroid", False),
    ]),
    ("Drift-through polarimetry (bright isolated pair now in the field)", [
      ("Pick the brightest isolated pair currently on the detector - you cannot point, so use what is there. Record its approximate position and declination", False),
      ("Know your drift rate: sky moves ~15 arcsec x cos(dec) per second = up to ~67 px/s near the equator at 0.224 arcsec/px. If RA holds sidereal, drift is far less", False),
      ("Set exposures short enough to keep images compact (aim < ~2 px trail). Near the equator with no RA hold that is tens of ms; lengthen if RA is holding the field", False),
      ("Run a complete polV8 sequence as fast as possible while the pair stays in a clean region", False),
      ("Repeat the same polV8 once or twice while the pair is still clean - redundancy for later centroid tracking and photometry", False),
      ("Rotate the WHOLE polarimeter +45 deg (field rotator) and repeat the same sequence on the same pair, provided both beams stay on the detector", False),
      ("Keep the whole-instrument-rotation dataset and the HWP-only dataset SEPARATE - they test different coordinate transforms", False),
      ("Do NOT attempt live analysis - just collect and log", False),
    ]),
    ("Log every block (paper or file)", [
      ("UTC; exposure time; filter; HWP angle; whole-polarimeter rotator angle", False),
      ("Camera temperature; gain; readout mode; offset", False),
      ("Approximate source position on the detector", False),
      ("Whether the source drifted near an edge, bad pixels, or another stellar pair", False),
    ]),
    ("Do NOT", [
      ("Do NOT force long tracked exposures - a dead DEC gives elongated stars and frustration", False),
      ("Do NOT change camera gain / mode / offset / HWP-zero mid-session (invalidates the calibration)", False),
      ("Do NOT attempt to point, slew, or engage DEC (auto-disconnects the mount)", False),
    ]),
    ("Before shutdown - verify", [
      ("All FITS files exist and filenames are unique", False),
      ("Every HWP angle is present in each sequence", False),
      ("Rotator angle is recorded for every block", False),
      ("Data AND logs AND the failure documentation are copied to a second location", False),
    ]),
    ("Salvage dataset payoff (what this buys you)", [
      ("One or more drifting polV8 sequences at the original polarimeter angle", False),
      ("The same sequence after a +45 deg whole-polarimeter rotation", False),
      ("Biases, matched darks, angle-matched V flats", False),
      ("Beam-geometry and HWP-modulation measurements", False),
      ("Complete metadata and the documented DEC failure", False),
      ("Enables later tests: modulation present? beam pair trackable frame-by-frame? detector-frame Stokes transforms correctly under instrument rotation?", False),
    ]),
  ]
  return checklist_pdf_from_sections(title, subtitle, sections, ROOT / "salvage_no_pointing_checklist.pdf")


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
