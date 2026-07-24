"""Header contract tests for capture paths."""
from __future__ import annotations

import numpy as np

from alpyca_tools.fits_writer import (
    DetectorCards,
    FitsHeaderConfig,
    PolarimetryCards,
    build_header,
)
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
            readout_mode=0,
            readout_mode_name="Mode 0",
            offset_setting=30,
            cooler_setpoint_c=-10.0,
            pixel_size_um=3.76,
        ),
    )
    hdr = build_header(_FakeCamera(), cfg, np.dtype(np.uint16), (100, 100))
    assert hdr["GAIN"] == 0
    assert hdr["READMODE"] == 0
    assert hdr["SET-TEMP"] == -10.0
    assert hdr["XPIXSZ"] == 3.76
    # Actual detector temperature uses the generic (non-CCD) keyword.
    assert hdr["DET-TEMP"] == -9.8
    assert "CCD-TEMP" not in hdr
    # Measured-only detector constants must never be stamped into the raw header.
    for absent in ("EGAIN", "RON", "PEDESTAL"):
        assert absent not in hdr


def test_fits_routine_detector_cards_match():
    assert RoutineDetectorCards is DetectorCards
    det = RoutineDetectorCards(
        gain_setting=0,
        readout_mode=0,
        readout_mode_name="Mode 0",
        offset_setting=30,
        cooler_setpoint_c=-10.0,
        pixel_size_um=3.76,
    )
    cfg = CaptureConfig(imagetyp="BIAS", detector=det)
    hdr = _build_header(_FakeCamera(), cfg, np.dtype(np.uint16), (100, 100))
    for key in ("GAIN", "READMODE", "SET-TEMP", "XPIXSZ", "YPIXSZ"):
        assert key in hdr
    assert "EGAIN" not in hdr


def test_shared_header_contains_required_grouping_and_detector_cards():
    cfg = FitsHeaderConfig(
        imagetyp="dark",
        exposure_s=0.25,
        date_obs="2026-07-10T02:00:00.000",
        instrument="sim-detector",
        detector=DetectorCards(pixel_size_um=3.76),
    )
    hdr = build_header(None, cfg, np.dtype(np.uint16), (2, 2))
    assert hdr["IMAGETYP"] == "DARK"
    assert hdr["EXPTIME"] == 0.25
    assert hdr["DATE-OBS"] == "2026-07-10T02:00:00.000"
    assert hdr["INSTRUME"] == "sim-detector"
    assert hdr["XPIXSZ"] == 3.76
    assert hdr["BZERO"] == 32768
    assert "EGAIN" not in hdr


def test_science_provenance_cards_present_and_derived():
    cfg = FitsHeaderConfig(
        imagetyp="light",
        exposure_s=384.0,
        date_obs="2025-01-10T00:00:00",
        object_name="HD_test",
        origin="POLITE",
        ra="23:12:41.30",
        dec="-05:21:02.3",
        ra_deg=346.5402,
        dec_deg=-5.3497,
        alt_deg=47.5,
        az_deg=130.2,
        airmass=1.353,
        lst="23:12:24.00",
        radesys="FK5",
        site_lat_deg=33.0701,
        site_lon_deg=-116.6451,
        site_elev_m=1294.0,
        focal_len_mm=3454.0,
        focal_ratio=6.8,
        aperture_mm=508.0,
        pixscale=0.224,
        polarimetry=PolarimetryCards(
            hwp_angle_deg=22.5, instrument_rotator_deg=12.3,
            eff_wavelength_nm=551.0, hwp_uncert_deg=0.012,
        ),
    )
    hdr = build_header(None, cfg, np.dtype(np.uint16), (4210, 6280))

    for key in (
        "MJD-OBS", "DATE-END", "DATE-AVG", "MJD-AVG",
        "SITELAT", "SITELONG", "SITEELEV", "OBSGEO-X", "OBSGEO-Y", "OBSGEO-Z",
        "FOCALLEN", "FOCRATIO", "APTDIA", "PIXSCALE",
        "OBJCTALT", "OBJCTAZ", "AIRMASS", "LST", "RADESYS", "CRVAL1", "CRVAL2",
        "WPUNCERT", "ORIGIN",
    ):
        assert key in hdr, f"missing {key}"

    # Pointing seed carries no full WCS solution (solved in reduction).
    for absent in ("CTYPE1", "CD1_1", "CDELT1"):
        assert absent not in hdr

    # Derived time cards: end = start + exptime, avg = start + exptime/2.
    assert hdr["DATE-END"] == "2025-01-10T00:06:24.000"
    assert hdr["DATE-AVG"] == "2025-01-10T00:03:12.000"
    assert abs(hdr["MJD-AVG"] - (hdr["MJD-OBS"] + 192.0 / 86400.0)) < 1e-9
    assert hdr["RADESYS"] == "FK5"
    assert hdr["CRVAL1"] == 346.5402
