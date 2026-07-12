"""Focused checks for the salvage-only frame tracker and modulation fit."""

from __future__ import annotations

import numpy as np
import pytest

from scripts.reduce_salvage_drift_sequence import (
    _fit_modulation,
    _local_flux,
    _match_pair,
)


def test_match_pair_uses_full_measured_vector():
    peaks = [
        {"x": 100.0, "y": 200.0, "smooth_snr": 20.0},
        {"x": -26.0, "y": 403.0, "smooth_snr": 18.0},
        {"x": 500.0, "y": 500.0, "smooth_snr": 40.0},
    ]
    ordinary, extraordinary = _match_pair(peaks, -127.0, 203.0, 2.0)
    assert ordinary["x"] == 100.0
    assert extraordinary["y"] == 403.0


def test_local_flux_recovers_source_above_constant_background():
    yy, xx = np.mgrid[:101, :101]
    image = np.full((101, 101), 12.0)
    source = 200.0 * np.exp(-((xx - 50) ** 2 + (yy - 50) ** 2) / (2 * 3.0 ** 2))
    image += source
    flux, sigma, background = _local_flux(
        image, np.zeros_like(image, dtype=bool), 50, 50,
        r_ap=12, r_in=18, r_out=28,
    )
    assert flux == pytest.approx(source.sum(), rel=0.02)
    assert background == pytest.approx(12.0)
    assert sigma == pytest.approx(0.0, abs=1e-8)


def test_modulation_fit_recovers_fourth_harmonic_with_offset():
    q, u, a0 = 0.03, -0.02, 0.01
    rows = []
    for angle in (0.0, 22.5, 45.0, 67.5, 90.0):
        z = a0 + q * np.cos(np.deg2rad(4 * angle)) + u * np.sin(np.deg2rad(4 * angle))
        rows.append({
            "tracked": True,
            "flux_o_adu": 1.0 - z,
            "flux_e_adu": 1.0 + z,
            "beam_ratio": z,
            "hwp_angle_deg": angle,
        })
    result = _fit_modulation(rows)
    assert result["throughput_offset_a0"] == pytest.approx(a0)
    assert result["q_detector_uncalibrated"] == pytest.approx(q)
    assert result["u_detector_uncalibrated"] == pytest.approx(u)
