"""Modulation reductions: round-trip + flat-field independence (research map §2)."""

import numpy as np
import pytest

import poltools as pt
from conftest import make_beamfluxes


ANGLES4 = (0.0, 22.5, 45.0, 67.5)
ANGLES8 = tuple(i * 22.5 for i in range(8))


@pytest.mark.parametrize("q,u", [(0.03, -0.02), (-0.01, 0.04), (0.0, 0.0), (0.10, 0.10)])
def test_noiseless_roundtrip_all_methods(q, u):
    bfs4 = make_beamfluxes(q, u, 1e6, ANGLES4)
    bfs8 = make_beamfluxes(q, u, 1e6, ANGLES8)
    a_ratio = pt.method_a_double_ratio(bfs4)
    a_diff = pt.method_a_double_difference(bfs4)
    b = pt.method_b_lsq(bfs8)
    for res in (a_ratio, b):
        assert res["q"] == pytest.approx(q, abs=1e-9)
        assert res["u"] == pytest.approx(u, abs=1e-9)
    # difference form is only first-order exact, but noiseless+matched beams -> exact
    assert a_diff["q"] == pytest.approx(q, abs=1e-9)
    assert a_diff["u"] == pytest.approx(u, abs=1e-9)


def test_double_ratio_flat_field_independent():
    """Method A (double-ratio) is invariant to a per-beam throughput imbalance;
    the difference form is not (only first order)."""
    q, u = 0.03, -0.02
    matched = make_beamfluxes(q, u, 1e6, ANGLES4, throughput=(1.0, 1.0))
    imbal = make_beamfluxes(q, u, 1e6, ANGLES4, throughput=(0.9, 1.1))
    ratio_m = pt.method_a_double_ratio(matched)
    ratio_i = pt.method_a_double_ratio(imbal)
    diff_i = pt.method_a_double_difference(imbal)
    # double-ratio: identical to machine precision
    assert ratio_i["q"] == pytest.approx(ratio_m["q"], abs=1e-9)
    assert ratio_i["u"] == pytest.approx(ratio_m["u"], abs=1e-9)
    # double-ratio recovers truth despite 22% imbalance
    assert ratio_i["q"] == pytest.approx(q, abs=1e-9)
    # difference form is biased by the imbalance (sanity: it deviates)
    assert abs(diff_i["q"] - q) > 1e-4


def test_method_a_requires_four_angles():
    bfs = make_beamfluxes(0.02, 0.0, 1e6, (0.0, 22.5, 45.0))
    with pytest.raises(ValueError):
        pt.method_a_double_ratio(bfs)


def test_method_b_chi2_and_cov(rng):
    bfs = make_beamfluxes(0.03, -0.02, 5e5, ANGLES8, rng=rng)
    B = pt.method_b_lsq(bfs)
    assert B["n_angles"] == 8
    assert B["dof"] == 6
    assert np.isfinite(B["chi2"])
    assert B["sigma_q"] > 0 and B["sigma_u"] > 0
