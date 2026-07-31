import numpy as np
import pytest

from obs_utils.focus import measure_fwhm, measure_hfd, star_profile


def _gaussian(*, sigma=2.5, background=100.0):
    y, x = np.mgrid[:101, :101]
    return background + 5000.0 * np.exp(-((x - 50.3) ** 2 + (y - 49.7) ** 2) / (2 * sigma ** 2))


def test_focus_metrics_recover_a_gaussian_width():
    sigma = 2.5
    expected = 2.0 * np.sqrt(2.0 * np.log(2.0)) * sigma
    image = _gaussian(sigma=sigma)
    profile = star_profile(image, center=(50.3, 49.7))

    assert profile.fwhm_px == pytest.approx(expected, abs=0.2)
    assert profile.hfd_px == pytest.approx(expected, abs=0.3)
    assert measure_fwhm(image, center=(50.3, 49.7), method="moments") == pytest.approx(expected, abs=0.05)
    assert measure_hfd(image, center=(50.3, 49.7)) == pytest.approx(expected, abs=0.3)


def test_focus_metric_rejects_unknown_fwhm_method():
    with pytest.raises(ValueError, match="method"):
        measure_fwhm(_gaussian(), method="fit")
