import numpy as np
import pytest

from caltools import conversion_gain_from_flat_pair


def _unity_gain_pair(shape=(100, 100)):
    """Flat pair whose difference variance equals the paired signal (gain = 1)."""
    rng = np.random.default_rng(20260717)
    difference = rng.normal(size=shape)
    difference -= difference.mean()
    difference *= np.sqrt(2000.0 / np.var(difference, ddof=1))

    bias = np.full(shape, 50.0)
    flat_a = bias + 1000.0 + difference / 2.0
    flat_b = bias + 1000.0 - difference / 2.0
    return flat_a, flat_b, bias


def test_single_point_ptc_recovers_unity_gain():
    flat_a, flat_b, bias = _unity_gain_pair()

    result = conversion_gain_from_flat_pair(flat_a, flat_b, bias, bias)

    assert result.scalar_summary["conversion_gain_e_per_adu"] == pytest.approx(1.0)
    assert result.scalar_summary["flat_pair_change_fraction"] == pytest.approx(0.0)
    assert result.metadata["flat_pair_stable"] is True


def test_single_point_ptc_rejects_mismatched_regions():
    data = np.ones((10, 10))

    with pytest.raises(ValueError, match="same shape"):
        conversion_gain_from_flat_pair(data, data, data, np.ones((9, 9)))


def test_central_fraction_crops_before_measuring():
    flat_a, flat_b, bias = _unity_gain_pair()
    # Corners the crop must exclude: unflattened vignetting would wreck the mean.
    flat_a[:10, :10] = 60000.0
    flat_b[:10, :10] = 60000.0

    result = conversion_gain_from_flat_pair(
        flat_a, flat_b, bias, bias, central_fraction=0.5
    )

    assert result.metadata["region_shape"] == (50, 50)
    assert result.scalar_summary["mean_flat_a_adu"] == pytest.approx(1050.0, abs=5.0)


def test_drifting_flat_pair_is_flagged_not_rejected():
    flat_a, flat_b, bias = _unity_gain_pair()
    flat_b = bias + (flat_b - bias) * 0.9  # sky faded 10% between the two flats

    result = conversion_gain_from_flat_pair(flat_a, flat_b, bias, bias)

    assert result.metadata["flat_pair_stable"] is False
    assert result.scalar_summary["flat_pair_change_fraction"] > 0.02
