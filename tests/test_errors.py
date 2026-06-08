"""Error metrics & debiasing: MAS bias removal, NK&C PA interval, residual σ_P.

These are validated against direct Monte-Carlo, which is ground truth for the
Rice/PA statistics (research map §3; Plaszczynski 2014, Montier II, NK&C 1993).
"""

import numpy as np
import pytest

import poltools as pt
from poltools import errors


def _mc_qu(p_true, sigma, n, rng, theta_true_deg=0.0):
    """Sample measured (q,u) with q,u ~ N(true, sigma)."""
    th = np.deg2rad(theta_true_deg)
    q0, u0 = p_true * np.cos(2 * th), p_true * np.sin(2 * th)
    q = rng.normal(q0, sigma, n)
    u = rng.normal(u0, sigma, n)
    return q, u


@pytest.mark.parametrize("snr", [1.0, 2.0, 3.0])
def test_mas_reduces_rice_bias(snr, rng):
    sigma = 0.005
    p_true = snr * sigma
    q, u = _mc_qu(p_true, sigma, 40000, rng)
    p_hat = np.hypot(q, u)
    p_mas = np.array([errors.debias_mas(p, sigma) for p in p_hat])

    bias_naive = p_hat.mean() - p_true
    bias_mas = p_mas.mean() - p_true
    # MAS strictly reduces the positive Rice bias
    assert abs(bias_mas) < abs(bias_naive)
    # at SNR>=3 MAS is nearly unbiased
    if snr >= 3.0:
        assert abs(bias_mas) < 0.25 * p_true


def test_wardle_kronberg_and_naive():
    assert errors.debias_naive(0.05) == 0.05
    # WK floors at 0 when p<sigma
    assert errors.debias_wardle_kronberg(0.01, 0.05) == 0.0
    assert errors.debias_wardle_kronberg(0.05, 0.03) == pytest.approx(np.sqrt(0.05**2 - 0.03**2))


@pytest.mark.parametrize("snr", [1.5, 3.0])
def test_nkc_pa_interval_matches_mc(snr, rng):
    sigma = 0.01
    p_true = snr * sigma
    q, u = _mc_qu(p_true, sigma, 200000, rng, theta_true_deg=0.0)
    theta = 0.5 * np.rad2deg(np.arctan2(u, q))  # offsets about 0 in (-90,90]
    theta = (theta + 90) % 180 - 90
    # empirical central 68.27% half-width
    half_emp = np.percentile(np.abs(theta), 68.27)
    half_nkc = errors.sigma_theta_nkc(snr)
    assert half_nkc == pytest.approx(half_emp, rel=0.15)


def test_sigma_theta_highsnr_value():
    assert errors.sigma_theta_highsnr(0.10, 0.01) == pytest.approx(28.6479 * 0.1, rel=1e-4)
    # NK&C converges to the Gaussian asymptotic at high SNR
    assert errors.sigma_theta_nkc(30.0) == pytest.approx(28.6479 / 30.0, rel=1e-6)


def test_residual_sigma_p_zero_for_perfect_fit():
    # noiseless modulation: residual σ_P ~ 0
    angles = [i * 22.5 for i in range(8)]
    from conftest import make_beamfluxes
    bfs = make_beamfluxes(0.03, -0.02, 1e8, angles)
    B = pt.method_b_lsq(bfs)
    assert B["sigma_p_resid"] == pytest.approx(0.0, abs=1e-6)
