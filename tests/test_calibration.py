"""Standard-star calibration: IP, PA zero-point, efficiency (research map §4)."""

import numpy as np
import pytest

import poltools as pt
from poltools import calibration as cal


def test_ip_recovered_from_unpolarized_standards():
    q0_true, u0_true = 0.004, -0.003
    rng = np.random.default_rng(1)
    q = rng.normal(q0_true, 1e-4, 50)
    u = rng.normal(u0_true, 1e-4, 50)
    q0, u0, cov = cal.fit_instrumental_polarization(q, u)
    assert q0 == pytest.approx(q0_true, abs=5e-4)
    assert u0 == pytest.approx(u0_true, abs=5e-4)
    # subtraction removes it
    qc, uc = cal.apply_ip(q0_true, u0_true, q0, u0)
    assert abs(qc) < 5e-4 and abs(uc) < 5e-4


def test_pa_zeropoint_and_rotation():
    dtheta_true = 9.23  # like DBIP
    # a source with known literature PA, measured at PA - dtheta
    theta_lit = np.array([20.0, 75.0, 130.0])
    theta_meas = (theta_lit - dtheta_true) % 180.0
    dtheta, sig = cal.fit_pa_zeropoint(theta_meas, theta_lit)
    assert dtheta == pytest.approx(dtheta_true, abs=1e-6)
    # rotating a measured vector by 2*dtheta restores the literature PA
    p = 0.05
    th_m = np.deg2rad(20.0 - dtheta_true)
    q, u = p * np.cos(2 * th_m), p * np.sin(2 * th_m)
    qc, uc = cal.apply_pa_zeropoint(q, u, dtheta)
    pa = 0.5 * np.rad2deg(np.arctan2(uc, qc)) % 180.0
    assert pa == pytest.approx(20.0, abs=1e-6)


def test_efficiency():
    eff = cal.fit_efficiency([0.045, 0.090], [0.050, 0.100])
    assert eff == pytest.approx(0.9, abs=1e-9)
    q, u = cal.apply_efficiency(0.045, 0.0, eff)
    assert q == pytest.approx(0.05, abs=1e-9)


def test_polcalibration_bundle_roundtrip():
    """A full PolCalibration inverts injected IP+PA+efficiency."""
    q0, u0, dtheta, eff = 0.003, -0.002, 7.0, 0.95
    # true source
    p_true, pa_true = 0.04, 30.0
    th = np.deg2rad(pa_true)
    qt, ut = p_true * np.cos(2 * th), p_true * np.sin(2 * th)
    # forward: apply efficiency, rotate by -2*dtheta (instrument frame), add IP
    a = -2 * np.deg2rad(dtheta)
    qm = eff * (qt * np.cos(a) - ut * np.sin(a)) + q0
    um = eff * (qt * np.sin(a) + ut * np.cos(a)) + u0
    calib = pt.PolCalibration(q0=q0, u0=u0, dtheta_deg=dtheta, efficiency=eff)
    qc, uc = calib.apply(qm, um)
    assert qc == pytest.approx(qt, abs=1e-9)
    assert uc == pytest.approx(ut, abs=1e-9)
