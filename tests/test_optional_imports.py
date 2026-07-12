"""Offline analysis imports must not load observatory hardware drivers."""

from __future__ import annotations

import sys


def test_alpyca_fits_writer_does_not_import_camera_driver():
    from alpyca_tools.fits_writer import FitsHeaderConfig

    assert FitsHeaderConfig is not None
    assert "alpyca_tools.camera_device" not in sys.modules


def test_obs_utils_qa_does_not_import_imaging_stack():
    from obs_utils.qa_lib import QAResult

    assert QAResult is not None
    assert "obs_utils.imaging" not in sys.modules
