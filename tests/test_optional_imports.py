"""Offline analysis imports must not load observatory hardware drivers."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _assert_clean_import(code: str) -> None:
    """Run the import assertion before any other test can load a driver."""
    result = subprocess.run(
        [sys.executable, "-c", code], cwd=ROOT, text=True, capture_output=True,
    )
    assert result.returncode == 0, result.stderr


def test_alpyca_fits_writer_does_not_import_camera_driver():
    _assert_clean_import("""
from alpyca_tools.fits_writer import FitsHeaderConfig
assert FitsHeaderConfig is not None
assert 'alpyca_tools.camera_device' not in __import__('sys').modules
""")


def test_obs_utils_qa_does_not_import_imaging_stack():
    _assert_clean_import("""
from obs_utils.qa_lib import QAResult
assert QAResult is not None
assert 'obs_utils.imaging' not in __import__('sys').modules
""")
