"""Shared fixtures for poltools tests."""

import numpy as np
import pytest

import poltools as pt
from caltools import SensorConfig


@pytest.fixture
def rng():
    return np.random.default_rng(20260607)


@pytest.fixture
def sensor():
    # QHY268M-like, small frame for fast tests
    return SensorConfig(nx=256, ny=256, pixel_size_um=3.76, gain_e_per_adu=1.0,
                        temperature_c=-10.0, sensor_name="QHY268M")


@pytest.fixture
def cfg(sensor):
    return pt.PolConfig(
        sensor=sensor,
        beam=pt.BeamGeometry(separation_px=40.0, position_angle_deg=0.0),
        hwp_angles_deg=(0.0, 22.5, 45.0, 67.5),
        read_noise_e=3.5,
        full_well_e=51000.0,
    )


def make_beamfluxes(q, u, total_e, angles, rng=None, throughput=(1.0, 1.0),
                    retardance_deg=180.0):
    """Flux-level o/e BeamFlux list (optionally Poisson-noised)."""
    from poltools._types import BeamFlux
    S = pt.stokes_vector(1.0, q, u)
    t_o, t_e = throughput
    out = []
    for th in angles:
        Io, Ie = pt.oe_intensities(S, th, retardance_deg=retardance_deg)
        fo, fe = Io * total_e * t_o, Ie * total_e * t_e
        if rng is not None:
            fo = rng.poisson(max(fo, 0))
            fe = rng.poisson(max(fe, 0))
        so = np.sqrt(max(fo, 1.0))
        se = np.sqrt(max(fe, 1.0))
        out.append(BeamFlux(hwp_deg=th, f_o=float(fo), f_e=float(fe),
                            sig_o=float(so), sig_e=float(se)))
    return out
