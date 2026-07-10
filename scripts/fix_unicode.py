#!/usr/bin/env python3
"""Normalize text files: UTF-8 encoding, no emojis, repair common mojibake."""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

SKIP_PARTS = {".git", ".pytest_cache", "__pycache__", "FITSDATA", "node_modules"}
SKIP_SUFFIXES = {
    ".pdf", ".fits", ".fit", ".png", ".jpg", ".jpeg", ".gif", ".webp",
    ".zip", ".pyc", ".exe", ".dll", ".ico", ".bmp", ".snk", ".gz", ".Z",
    ".bz2", ".doc", ".log",
}
TEXT_SUFFIXES = {
    ".md", ".py", ".ipynb", ".html", ".yaml", ".yml", ".txt", ".toml",
    ".json", ".sh", ".zsh", ".css", ".js", ".ts",
}

EMOJI_RE = re.compile(
    "["
    "\U0001F300-\U0001FAFF"  # pictographs
    "\U00002600-\U000027BF"  # misc symbols (warning, checkmarks, stars)
    "\U0001F600-\U0001F64F"  # emoticons
    "\U0001F680-\U0001F6FF"  # transport/map
    "\U0000FE0F"             # variation selector
    "]+",
    flags=re.UNICODE,
)

EMOJI_REPLACEMENTS = [
    ("\u26a0\ufe0f", "WARNING:"),  # warning sign + VS-16
    ("\u26a0", "WARNING:"),         # warning sign
    ("\u2714", "(checked)"),        # heavy check mark
    ("\u2713", ""),                 # check mark
    ("\u2605", "*"),                # black star
    ("\u2610", "[ ]"),              # ballot box
    ("\u2794", "->"),               # heavy arrow
]


def should_process(path: Path) -> bool:
    if not path.is_file():
        return False
    if any(part in SKIP_PARTS for part in path.parts):
        return False
    if path.suffix.lower() in SKIP_SUFFIXES:
        return False
    if path.suffix.lower() not in TEXT_SUFFIXES:
        return False
    if path.name == "fix_unicode.py":
        return False
    return True


def read_text(path: Path) -> tuple[str, str]:
    raw = path.read_bytes()
    try:
        return raw.decode("utf-8"), "utf-8"
    except UnicodeDecodeError:
        return raw.decode("cp1252"), "cp1252"


def strip_emojis(text: str) -> str:
    for old, new in EMOJI_REPLACEMENTS:
        text = text.replace(old, new)
    text = EMOJI_RE.sub("", text)
    # Remove stray space before table cell ends left by deleted checkmarks.
    text = re.sub(r" +(\|)\s*$", r"\1", text, flags=re.MULTILINE)
    return text


def fix_lab_trial_checklist(text: str) -> str:
    em = "\u2014"  # em dash
    deg = "\u00b0"
    th = "\u03b8"  # theta
    subs = [
        ("Checklist \ufffd First", f"Checklist {em} First"),
        ("dark/flat/bias \ufffd content", f"dark/flat/bias {em} content"),
        ("curve (\ufffd4b)", f"curve (~4{th} harmonic)"),
        ("plate** \ufffd it is", f"plate** {em} it is"),
        ("**INDIGO** \ufffd i.e.", f"**INDIGO** {em} i.e."),
        ("native USB) \ufffd you need", f"native USB) {em} you need"),
        ("wait \ufffd it auto-homes", f"wait {em} it auto-homes"),
        ("Software \ufffd needed", f"Software {em} needed"),
        ("`/opt/homebrew` | \ufffd |", "`/opt/homebrew` | yes | - |"),
        ("| ? 3.5 | \ufffd (only", "| yes: 3.5 | - (only"),
        ("| ? 7.2.0 | \ufffd |", "| yes: 7.2.0 | - |"),
        ("| ? not installed |", "| no: not installed |"),
        ("| ?? installed in base", "| warn: installed in base"),
        ("| ? not installed | Install", "| no: not installed | Install"),
        ("| ? verify |", "| check |"),
        ("| ? |", "| - |"),
        ("(? 2.0.232", "(>= 2.0.232"),
        ("USB?RS-232", "USB-RS-232"),
        ("Preferences ? INDIGO", "Preferences > INDIGO"),
        ("easiest \ufffd bundles", f"easiest {em} bundles"),
        ("indigo_ccd_qhy2`) \ufffd run", f"indigo_ccd_qhy2`) {em} run"),
        ("note them \ufffd Rotator", f"note them {em} Rotator"),
        ("commands** \ufffd connect", f"commands** {em} connect"),
        ("filters** \ufffd connect", f"filters** {em} connect"),
        ("photo** \ufffd connect", f"photo** {em} connect"),
        (f"~90\ufffd", f"~90{deg}"),
        (f"22.5\ufffd increments", f"22.5{deg} increments"),
        (f"A 22.5\ufffd", f"A 22.5{deg}"),
        (f"by 45\ufffd,", f"by 45{deg},"),
        ("q?u", "q/u"),
        ("beams \ufffd \"**beam", f"beams {em} \"**beam"),
        ("swapping**\" \ufffd so", f"swapping**\" {em} so"),
        (f"67.5}}\ufffd`.", f"67.5}}{deg}`."),
        ("effects \ufffd Patat", f"effects {em} Patat"),
        (
            f"(8 ? 0\ufffd157.5\ufffd, 16 ? 0\ufffd337.5\ufffd)",
            f"(8 angles: 0{deg}\u2013157.5{deg}, 16 angles: 0{deg}\u2013337.5{deg})",
        ),
        ("Gonz\ufffdlez-Gait\ufffdn", "Gonz\u00e1lez-Gait\u00e1n"),
        (f"the 4\ufffd signal", f"the 4{th} signal"),
        (f"2020) \ufffd not needed", f"2020) {em} not needed"),
        (f"LE2Pol \ufffd same", f"LE2Pol {em} same"),
        ("[ ] `\ufffd scripts", "[ ] `scripts"),
        ("p ? 0", "p ~ 0"),
        (f"`p`/`\ufffd`", f"`p`/`{th}`"),
        ("n=2 ? 0", "n=2 ~ 0"),
        (f"(0, 22.5, \ufffd)", f"(0, 22.5{deg}, \u2026)"),
        (f"?1.4 s per 22.5\ufffd", f"~1.4 s per 22.5{deg}"),
        (f"to 90\ufffd", f"to 90{deg}"),
        (f"path \ufffd\n", f"path {em}\n"),
        (f"Reference \ufffd Pyxis", f"Reference {em} Pyxis"),
        (f"(000\ufffd359)", "(000\u2013359)"),
        (f"`CWAKUP` | \ufffd /", f"`CWAKUP` | {em} /"),
        (f"4\ufffd modulation", f"4{th} modulation"),
    ]
    for old, new in subs:
        text = text.replace(old, new)
    if "\ufffd" in text:
        text = text.replace("\ufffd", em)
    return text


def process_file(path: Path, dry_run: bool = False) -> bool:
    original, encoding = read_text(path)
    text = original

    if path.name == "lab_trial_checklist.md":
        text = fix_lab_trial_checklist(text)

    text = strip_emojis(text)

    if encoding == "cp1252":
        pass  # will be saved as UTF-8

    if text == original and encoding == "utf-8":
        return False

    if not dry_run:
        path.write_text(text, encoding="utf-8", newline="\n")
    return True


def main() -> int:
    dry_run = "--dry-run" in sys.argv
    changed: list[str] = []
    for path in sorted(ROOT.rglob("*")):
        if not should_process(path):
            continue
        rel = path.relative_to(ROOT)
        try:
            if process_file(path, dry_run=dry_run):
                changed.append(str(rel))
        except Exception as exc:
            print(f"ERROR {rel}: {exc}", file=sys.stderr)

    for name in changed:
        print(name)
    print(f"\n{'Would change' if dry_run else 'Changed'} {len(changed)} files")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
