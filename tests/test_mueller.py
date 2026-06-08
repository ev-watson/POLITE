"""Mueller forward-model checks."""

import numpy as np
import pytest

import poltools as pt


@pytest.mark.parametrize("theta", [0.0, 11.3, 22.5, 45.0, 67.5, 90.0, 123.4])
@pytest.mark.parametrize("qu", [(0.0, 0.0), (0.05, 0.0), (0.0, -0.04), (0.03, -0.02)])
def test_ideal_hwp_formula(theta, qu):
    """oe_intensities reproduces I'=½[I ± (Q cos4θ + U sin4θ)] exactly."""
    q, u = qu
    S = pt.stokes_vector(1.0, q, u)
    Io, Ie = pt.oe_intensities(S, theta)
    ac = q * np.cos(4 * np.deg2rad(theta)) + u * np.sin(4 * np.deg2rad(theta))
    assert Ie == pytest.approx(0.5 * (1 + ac), abs=1e-12)
    assert Io == pytest.approx(0.5 * (1 - ac), abs=1e-12)
    # flux conservation: o + e == I
    assert (Io + Ie) == pytest.approx(1.0, abs=1e-12)


def test_hwp_matrix_structure():
    """Ideal-HWP Mueller matrix is [[1,0,0,0],[0,c4,s4,0],[0,s4,-c4,0],[0,0,0,-1]]."""
    for phi in [10.0, 30.0, 50.0]:
        M = pt.M_hwp(phi)
        c4, s4 = np.cos(4 * np.deg2rad(phi)), np.sin(4 * np.deg2rad(phi))
        expect = np.array([[1, 0, 0, 0],
                           [0, c4, s4, 0],
                           [0, s4, -c4, 0],
                           [0, 0, 0, -1]], dtype=float)
        assert np.allclose(M, expect, atol=1e-12)


def test_rotator_orthogonal_and_inverse():
    R = pt.M_rotator(31.0)
    Rinv = pt.M_rotator(-31.0)
    assert np.allclose(R @ Rinv, np.eye(4), atol=1e-12)


def test_retarder_depolarization_window():
    """Real-retarder depolarization of recovered p is ≤0.2% for δ∈[176.4,183.6]°
    (Masiero 2007)."""
    q, u = 0.04, -0.03
    p_true = np.hypot(q, u)
    S = pt.stokes_vector(1.0, q, u)
    angles = [i * 22.5 for i in range(8)]
    from poltools._types import BeamFlux
    for delta in (176.4, 180.0, 183.6):
        bfs = []
        for th in angles:
            Io, Ie = pt.oe_intensities(S, th, retardance_deg=delta)
            bfs.append(BeamFlux(th, Io, Ie, 0.0, 0.0))
        B = pt.lsq_modulation(bfs)
        p_meas = np.hypot(B["q"], B["u"])
        assert abs(p_meas - p_true) / p_true <= 2.0e-3


def test_system_mueller_matches_oe_intensities():
    """The full-chain Mueller matrix (analyzer along +Q') reproduces the e-beam
    intensity from oe_intensities at every HWP angle."""
    q, u = 0.05, -0.03
    S = pt.stokes_vector(1.0, q, u)
    for th in (0.0, 22.5, 45.0, 67.5):
        M = pt.system_mueller(th)              # analyzer_deg=0 -> +Q' (e) arm
        I_e = float((M @ S)[0])
        _, Ie_ref = pt.oe_intensities(S, th)
        assert I_e == pytest.approx(Ie_ref, abs=1e-12)


def test_qwp_ready_v_mode():
    """Architecture is V-ready: a QWP (δ=90) Mueller matrix is correct/non-trivial
    in the V row (even though the linear pipeline never solves V)."""
    M = pt.M_retarder(90.0, 0.0)
    # QWP with fast axis 0: maps U<->V (block), leaves Q fixed
    assert M[1, 1] == pytest.approx(1.0, abs=1e-12)
    assert abs(M[2, 3]) == pytest.approx(1.0, abs=1e-12)
    assert abs(M[3, 2]) == pytest.approx(1.0, abs=1e-12)
