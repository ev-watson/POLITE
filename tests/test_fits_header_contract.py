"""Header contract tests for capture paths."""
from __future__ import annotations

import numpy as np
from astropy.io import fits

from alpyca_tools.fits_writer import DetectorCards, FitsHeaderConfig, build_header
from obs_utils.fits_routine import CaptureConfig, DetectorCards as RoutineDetectorCards, _build_header


class _FakeCamera:
    LastExposureDuration = 1.0
    LastExposureStartTime = "2026-07-09T02:00:00"
    BinX = BinY = 1
    Gain = 0
    Offset = 30
    CCDTemperature = -9.8
    SensorName = "QHY268M"


def test_fits_writer_detector_cards():
    cfg = FitsHeaderConfig(
        imagetyp="BIAS",
        detector=DetectorCards(
            gain_setting=0,
            egain_e_per_adu=1.0,
            readout_mode=0,
            readout_mode_name="Mode 0",
            offset_setting=30,
            cooler_setpoint_c=-10.0,
            pixel_size_um=3.76,
            ron_e=3.5,
        ),
    )
    hdr = build_header(_FakeCamera(), cfg, np.dtype(np.uint16), (100, 100))
    assert hdr["GAIN"] == 0
    assert hdr["EGAIN"] == 1.0
    assert hdr["READMODE"] == 0
    assert hdr["SET-TEMP"] == -10.0
    assert hdr["XPIXSZ"] == 3.76


def test_fits_routine_detector_cards_match():
    det = RoutineDetectorCards(
        gain_setting=0,
        egain_e_per_adu=1.0,
        readout_mode=0,
        readout_mode_name="Mode 0",
        offset_setting=30,
        cooler_setpoint_c=-10.0,
        pixel_size_um=3.76,
        ron_e=3.5,
    )
    cfg = CaptureConfig(imagetyp="BIAS", detector=det)
    hdr = _build_header(_FakeCamera(), cfg, np.dtype(np.uint16), (100, 100))
    for key in ("GAIN", "EGAIN", "READMODE", "SET-TEMP", "XPIXSZ", "YPIXSZ"):
        assert key in hdr
