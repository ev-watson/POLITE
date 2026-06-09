"""Per-filter α-BBO geometry, bad-region exclusion, PWI4 INSTROT, Savart cards.

Covers the optical-chain correction (2026-06-08): the α-BBO Savart split is
dispersive, so the o/e beam separation is per EFW band; the chipped corner is
excluded at reduction time; INSTROT (PWI4 rotator) is read per frame; and the
FITS writer records the Savart provenance cards.
"""

import warnings

import numpy as np
import pytest
from astropy.io import fits
from astropy.table import Table

import poltools as pt
from poltools import photometry as phot


ANGLES4 = (0.0, 22.5, 45.0, 67.5)


def _cfg_two_bands(sensor):
    """A PolConfig whose two EFW bands have DIFFERENT α-BBO separations."""
    filters = (
        pt.FilterConfig("Photometric B", pt.BeamGeometry(30.0, 0.0),
                        eff_wavelength_nm=440.0),
        pt.FilterConfig("Photometric R", pt.BeamGeometry(50.0, 0.0),
                        eff_wavelength_nm=640.0),
    )
    return pt.PolConfig(
        sensor=sensor,
        beam=pt.BeamGeometry(30.0, 0.0),
        hwp_angles_deg=ANGLES4,
        filters=filters,
        read_noise_e=3.5,
        full_well_e=51000.0,
    )


# --- registry / for_filter -------------------------------------------------

def test_default_efw_filters_slots():
    fs = pt.default_efw_filters(60.0)
    names = [f.name for f in fs]
    assert names == ["Photometric B", "Photometric V", "Photometric R",
                     "Clear", "Dark"]
    dark = [f for f in fs if f.name == "Dark"][0]
    assert dark.is_dark is True
    # placeholders until measured from data
    assert all(f.characterized is False for f in fs)
    assert all(f.beam.separation_px == 60.0 for f in fs)


def test_for_filter_selects_band_geometry(sensor):
    cfg = _cfg_two_bands(sensor)
    b = cfg.for_filter("Photometric B")
    r = cfg.for_filter("Photometric R")
    assert b.beam.separation_px == 30.0 and b.filter_name == "Photometric B"
    assert r.beam.separation_px == 50.0 and r.filter_name == "Photometric R"
    # registry carried over verbatim; active_filter resolves
    assert r.active_filter().eff_wavelength_nm == 640.0
    with pytest.raises(KeyError):
        cfg.for_filter("Halpha")


# --- group_by_filter -------------------------------------------------------

def test_group_by_filter(sensor, rng, tmp_path):
    cfg = _cfg_two_bands(sensor)
    scene = pt.make_scene([(64.0, 40.0)], [(1, 0.03, -0.02, 0)], [3e5])
    paths = []
    for band in ("Photometric B", "Photometric R"):
        cb = cfg.for_filter(band)
        ps = pt.simulate_sequence(scene, cb, out_dir=tmp_path / band, rng=rng,
                                  shape=(128, 128), exptime_s=2.0)
        paths += [str(p) for p in ps]
    groups = pt.group_by_filter(paths)
    assert set(groups) == {"Photometric B", "Photometric R"}
    assert len(groups["Photometric B"]) == len(ANGLES4)
    assert len(groups["Photometric R"]) == len(ANGLES4)


# --- region exclusion (chipped corner readiness) ---------------------------

def test_point_in_regions():
    regs = [(0, 0, 10, 10)]
    assert phot.point_in_regions(5, 5, regs) is True
    assert phot.point_in_regions(10, 0, regs) is True       # inclusive edge
    assert phot.point_in_regions(50, 5, regs) is False
    assert phot.point_in_regions(5, 5, None) is False        # no regions


def test_pair_oe_exclude_regions():
    # one o/e pair near origin, one near (80,80); exclude the second's corner
    beam = pt.BeamGeometry(separation_px=20.0, position_angle_deg=0.0)  # +y
    det = Table({
        "x_centroid": [30.0, 30.0, 80.0, 80.0],
        "y_centroid": [30.0, 50.0, 80.0, 100.0],
    })
    all_pairs = phot.pair_oe(det, beam)
    assert len(all_pairs) == 2
    kept = phot.pair_oe(det, beam, exclude_regions=[(70, 70, 110, 110)])
    assert len(kept) == 1
    assert kept[0][0] == (30.0, 30.0)


def test_reduce_to_stokes_exclude_regions_drops_source(sensor, rng, tmp_path):
    cfg = pt.PolConfig(sensor=sensor, beam=pt.BeamGeometry(20.0, 0.0),
                       hwp_angles_deg=ANGLES4, read_noise_e=3.5,
                       full_well_e=51000.0)
    positions = [(40.0, 40.0), (90.0, 40.0)]
    names = ["keep", "drop"]
    scene = pt.make_scene(positions, [(1, 0.03, -0.02, 0)] * 2, [3e5, 3e5], names)
    paths = [str(p) for p in pt.simulate_sequence(
        scene, cfg, out_dir=tmp_path, rng=rng, shape=(160, 96), exptime_s=3.0)]
    res = pt.reduce_to_stokes(
        paths, cfg, o_positions=positions, names=names, method="double_ratio",
        r_ap=5, r_in=7, r_out=10,
        exclude_regions=[(80, 30, 100, 70)],   # covers the "drop" source
    )
    got = {r.name for r in res}
    assert "keep" in got and "drop" not in got


# --- per-filter reduction (dispersion would break a single fixed offset) ----

def test_reduce_to_stokes_per_filter_geometry(sensor, rng, tmp_path):
    cfg = _cfg_two_bands(sensor)
    q_t, u_t = 0.03, -0.02
    pos = [(64.0, 40.0)]
    scene = pt.make_scene(pos, [(1, q_t, u_t, 0)], [6e5], ["src"])
    paths = []
    for band in ("Photometric B", "Photometric R"):
        cb = cfg.for_filter(band)            # B: sep 30, R: sep 50
        ps = pt.simulate_sequence(scene, cb, out_dir=tmp_path / band, rng=rng,
                                  shape=(128, 128), exptime_s=5.0,
                                  sky_e_per_px=20.0)
        paths += [str(p) for p in ps]
    # reduce ALL bands together: pipeline must apply each band's own offset
    res = pt.reduce_to_stokes(paths, cfg, o_positions=pos, names=["src"],
                              method="double_ratio", r_ap=4, r_in=6, r_out=9)
    assert len(res) == 2
    by_band = {r.metadata["filter"]: r for r in res}
    assert set(by_band) == {"Photometric B", "Photometric R"}
    for band, r in by_band.items():
        s = r.scalar_summary
        assert abs(s["q"] - q_t) < 0.01, (band, s["q"])
        assert abs(s["u"] - u_t) < 0.01, (band, s["u"])


# --- PWI4 INSTROT per-frame consistency ------------------------------------

def test_instrot_varies_warns(sensor, rng, tmp_path):
    cfg = pt.PolConfig(sensor=sensor, beam=pt.BeamGeometry(20.0, 0.0),
                       hwp_angles_deg=ANGLES4, read_noise_e=3.5,
                       full_well_e=51000.0)
    pos = [(48.0, 40.0)]
    scene = pt.make_scene(pos, [(1, 0.03, -0.02, 0)], [3e5], ["src"])
    paths = []
    for i, ang in enumerate(ANGLES4):
        frame = pt.render_frame(scene, cfg, ang, exptime_s=3.0, rng=rng,
                                shape=(96, 96), sky_e_per_px=20.0)
        p = pt.write_pol_fits(tmp_path / f"f{i}.fit", frame, ang, cfg,
                              seq_index=i, extra={"INSTROT": float(i)})  # drifts
        paths.append(str(p))
    with pytest.warns(UserWarning, match="INSTROT"):
        pt.reduce_to_stokes(paths, cfg, o_positions=pos, names=["src"],
                            method="double_ratio", r_ap=5, r_in=7, r_out=10)


def test_instrot_constant_no_warn(sensor, rng, tmp_path):
    cfg = pt.PolConfig(sensor=sensor, beam=pt.BeamGeometry(20.0, 0.0),
                       hwp_angles_deg=ANGLES4, read_noise_e=3.5,
                       full_well_e=51000.0)
    pos = [(48.0, 40.0)]
    scene = pt.make_scene(pos, [(1, 0.03, -0.02, 0)], [3e5], ["src"])
    paths = [str(p) for p in pt.simulate_sequence(
        scene, cfg, out_dir=tmp_path, rng=rng, shape=(96, 96), exptime_s=3.0,
        sky_e_per_px=20.0)]
    with warnings.catch_warnings():
        warnings.simplefilter("error")          # any INSTROT warning -> failure
        pt.reduce_to_stokes(paths, cfg, o_positions=pos, names=["src"],
                            method="double_ratio", r_ap=5, r_in=7, r_out=10)


# --- Savart provenance FITS cards ------------------------------------------

def test_write_pol_fits_savart_cards(sensor, rng, tmp_path):
    cfg = pt.PolConfig(sensor=sensor, beam=pt.BeamGeometry(60.0, 0.0),
                       filters=pt.default_efw_filters(60.0),
                       read_noise_e=3.5, full_well_e=51000.0).for_filter("Photometric V")
    scene = pt.make_scene([(40.0, 40.0)], [(1, 0.0, 0.0, 0)], [2e5])
    frame = pt.render_frame(scene, cfg, 0.0, exptime_s=2.0, rng=rng,
                            shape=(96, 96))
    p = pt.write_pol_fits(tmp_path / "v.fit", frame, 0.0, cfg)
    hdr = fits.getheader(str(p))
    assert hdr["SAVMAT"] == "alpha-BBO"
    assert hdr["SAVTHK"] == 18.0
    assert hdr["WAVELEN"] == 551.0
    assert hdr["FILTER"] == "Photometric V"
