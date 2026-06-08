"""End-to-end pipeline: 2D injection-recovery + Monte-Carlo pull calibration."""

import numpy as np
import pytest

import poltools as pt
from poltools._types import BeamFlux
from conftest import make_beamfluxes


ANGLES4 = (0.0, 22.5, 45.0, 67.5)
ANGLES8 = tuple(i * 22.5 for i in range(8))


def test_end_to_end_2d_injection_recovery(cfg, rng, tmp_path):
    positions = [(70.0, 80.0), (180.0, 150.0)]
    truth = [(0.030, -0.020), (-0.015, 0.035)]
    stokes = [(1, q, u, 0) for (q, u) in truth]
    scene = pt.make_scene(positions, stokes, [4.0e6, 3.0e6], names=["s0", "s1"])
    paths = pt.simulate_sequence(scene, cfg, out_dir=tmp_path, exptime_s=15.0,
                                 seeing_arcsec=2.0, sky_e_per_px=80.0, rng=rng,
                                 shape=(256, 256))
    res = pt.reduce_to_stokes([str(p) for p in paths], cfg, o_positions=positions,
                              names=["s0", "s1"], method="double_ratio",
                              r_ap=7, r_in=12, r_out=20, bias_adu=1000.0)
    assert len(res) == 2
    for r, (qt, ut) in zip(res, truth):
        s = r.scalar_summary
        # recovered within 4σ (high SNR -> tight)
        assert abs(s["q"] - qt) < 4 * s["sigma_q"] + 1e-3
        assert abs(s["u"] - ut) < 4 * s["sigma_u"] + 1e-3
        assert s["snr"] > 10


def test_methods_a_and_b_agree(cfg, rng, tmp_path):
    positions = [(128.0, 128.0)]
    scene = pt.make_scene(positions, [(1, 0.04, -0.03, 0)], [6e6])
    cfg8 = cfg.with_hwp_angles(ANGLES8)
    paths = pt.simulate_sequence(scene, cfg8, out_dir=tmp_path, exptime_s=15.0,
                                 seeing_arcsec=2.0, sky_e_per_px=60.0, rng=rng,
                                 shape=(256, 256))
    rA = pt.reduce_to_stokes([str(p) for p in paths], cfg8, o_positions=positions,
                             method="double_ratio", r_ap=7, r_in=12, r_out=20)[0]
    rB = pt.reduce_to_stokes([str(p) for p in paths], cfg8, o_positions=positions,
                             method="lsq", r_ap=7, r_in=12, r_out=20)[0]
    assert rA.scalar_summary["q"] == pytest.approx(rB.scalar_summary["q"], abs=3e-3)
    assert rA.scalar_summary["u"] == pytest.approx(rB.scalar_summary["u"], abs=3e-3)


@pytest.mark.parametrize("q_true,u_true", [(0.03, -0.02)])
def test_pull_distribution_is_unit_normal(q_true, u_true, rng):
    """MC: q,u pulls (x̂−x_true)/σ ~ N(0,1) (correct error bars)."""
    n_trials = 4000
    pulls_q, pulls_u = [], []
    for _ in range(n_trials):
        bfs = make_beamfluxes(q_true, u_true, 2.0e5, ANGLES8, rng=rng)
        B = pt.lsq_modulation(bfs)
        pulls_q.append((B["q"] - q_true) / B["sigma_q"])
        pulls_u.append((B["u"] - u_true) / B["sigma_u"])
    pulls_q = np.array(pulls_q)
    pulls_u = np.array(pulls_u)
    assert abs(pulls_q.mean()) < 0.08
    assert abs(pulls_u.mean()) < 0.08
    assert pulls_q.std(ddof=1) == pytest.approx(1.0, abs=0.08)
    assert pulls_u.std(ddof=1) == pytest.approx(1.0, abs=0.08)


def test_calibration_applied_in_pipeline(cfg, rng, tmp_path):
    """Injecting IP in the simulator and supplying its calibration recovers truth."""
    q_true, u_true = 0.02, 0.0
    q0, u0 = 0.005, -0.004  # instrumental polarization to inject and remove
    positions = [(128.0, 128.0)]
    scene = pt.make_scene(positions, [(1, q_true, u_true, 0)], [6e6])
    paths = pt.simulate_sequence(scene, cfg, out_dir=tmp_path, exptime_s=15.0,
                                 seeing_arcsec=2.0, sky_e_per_px=60.0, rng=rng,
                                 shape=(256, 256), ip=(q0, u0))
    calib = pt.PolCalibration(q0=q0, u0=u0)
    res = pt.reduce_to_stokes([str(p) for p in paths], cfg, o_positions=positions,
                              method="double_ratio", calibration=calib,
                              r_ap=7, r_in=12, r_out=20)[0]
    s = res.scalar_summary
    assert abs(s["q"] - q_true) < 4 * s["sigma_q"] + 1e-3
    assert abs(s["u"] - u_true) < 4 * s["sigma_u"] + 1e-3
