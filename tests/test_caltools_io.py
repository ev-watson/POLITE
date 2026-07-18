"""Header-authoritative calibration I/O tests."""

from __future__ import annotations

import warnings

import numpy as np
import pytest
from astropy.io import fits

import caltools as ct


def _frame(path, *, imagetyp="DARK", exptime=1.0, **extra):
    header = fits.Header({"IMAGETYP": imagetyp, "EXPTIME": exptime, **extra})
    fits.PrimaryHDU(np.zeros((2, 2), dtype=np.uint16), header=header).writeto(path)
    return str(path)


def test_grouping_uses_headers_for_polite_and_arbitrary_names(tmp_path):
    polite = _frame(tmp_path / "2026-07-10_target_V_1s_001.fits", exptime=2.0)
    arbitrary = _frame(tmp_path / "not-a-camera-name.fits", exptime=2.0)
    groups = ct.group_by_type_and_exposure([polite, arbitrary])
    assert groups == {("DARK", 2.0): sorted([polite, arbitrary])}


def test_stale_filename_is_not_interpreted_as_metadata(tmp_path):
    # This resembles an old camera naming scheme but its header is authoritative.
    path = _frame(tmp_path / "00000042Dark10secs.fits", imagetyp="LIGHT", exptime=0.25)
    groups = ct.group_by_type_and_exposure([path])
    assert list(groups) == [("LIGHT", 0.25)]


@pytest.mark.parametrize("missing", ["IMAGETYP", "EXPTIME"])
def test_grouping_rejects_missing_required_header_card(tmp_path, missing):
    header = fits.Header({"IMAGETYP": "DARK", "EXPTIME": 1.0})
    del header[missing]
    path = tmp_path / "missing.fits"
    fits.PrimaryHDU(np.zeros((2, 2), dtype=np.uint16), header=header).writeto(path)
    with pytest.raises(KeyError, match=missing):
        ct.group_by_type_and_exposure([str(path)])


def test_sensor_config_requires_detector_metadata_or_explicit_fallbacks(tmp_path):
    path = _frame(
        tmp_path / "detector.fits",
        imagetyp="BIAS",
        exptime=0.0,
        EGAIN=1.2,
        INSTRUME="test-detector",
    )
    with pytest.raises(KeyError, match="XPIXSZ"):
        ct.sensor_config_from_header(path)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        cfg = ct.sensor_config_from_header(path, pixel_size_um=3.76)
    assert cfg.sensor_name == "test-detector"
    assert cfg.pixel_size_um == 3.76
    assert not any("QHY" in str(w.message) for w in caught)

