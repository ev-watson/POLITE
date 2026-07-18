import numpy as np
import pytest

from scripts.quick_unity_gain import estimate_conversion_gain


def test_single_point_ptc_recovers_unity_gain():
    rng = np.random.default_rng(20260717)
    difference = rng.normal(size=(100, 100))
    difference -= difference.mean()
    difference *= np.sqrt(2000.0 / np.var(difference, ddof=1))

    bias = np.full((100, 100), 50.0)
    flat_a = bias + 1000.0 + difference / 2.0
    flat_b = bias + 1000.0 - difference / 2.0

    metrics = estimate_conversion_gain(flat_a, flat_b, bias, bias)

    assert metrics["conversion_gain_e_per_adu"] == pytest.approx(1.0)
    assert metrics["flat_pair_change_fraction"] == pytest.approx(0.0)


def test_single_point_ptc_rejects_mismatched_regions():
    data = np.ones((10, 10))

    with pytest.raises(ValueError, match="same shape"):
        estimate_conversion_gain(data, data, data, np.ones((9, 9)))
