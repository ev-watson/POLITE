"""CMOS-audit fixes (findings C-1..C-7) for the QHY268M / IMX571 path.

Covers the effective-N σ suppression (C-3; Stockmans et al. eq. 16, Source B),
the per-pixel read-noise map (C-1), bad/hot-pixel repair (C-2), the dark-frame
FPN injection (C-4), gain-mode provenance (C-5), and the saturation flag (C-6).
"""

import warnings

import numpy as np
import pytest

import poltools as pt
from poltools._types import PolConfig
from poltools.photometry import _MEDIAN_VAR_INFLATION, _effective_n


# --------------------------------------------------------------------------- #
# C-3 — effective-N σ suppression
# --------------------------------------------------------------------------- #
def test_effective_n_values():
    # N<=2 is exact for the median (frame / mean); N>=3 takes the pi/2 penalty.
    assert _effective_n(1, "median") == 1.0
    assert _effective_n(2, "median") == 2.0
    assert _effective_n(3, "median") == pytest.approx(3.0 / _MEDIAN_VAR_INFLATION)
    assert _effective_n(5, "median") == pytest.approx(5.0 / _MEDIAN_VAR_INFLATION)
    # the mean attains the full N at every N
    for n in (1, 2, 3, 5):
        assert _effective_n(n, "mean") == float(n)
    with pytest.raises(ValueError):
        _effective_n(3, "bogus")


def test_sigma_scales_with_effective_n(cfg, rng):
    """On a fixed image, the reported σ divides by sqrt(effective N) exactly."""
    pos = (128.0, 128.0)
    scene = pt.make_scene([pos], [(1, 0.0, 0.0, 0)], [3.0e5])
    img = pt.render_frame(scene, cfg, 0.0, exptime_s=10.0, seeing_arcsec=1.0,
                          sky_e_per_px=50.0, rng=rng, shape=(256, 256))
    dx, dy = cfg.beam.offset_xy()
    e_xy = (pos[0] + dx, pos[1] + dy)
    _, s1 = pt.measure_fluxes(img, [pos, e_xy], cfg, r_ap=8, r_in=12, r_out=18)
    _, s2 = pt.measure_fluxes(img, [pos, e_xy], cfg, r_ap=8, r_in=12, r_out=18,
                              n_combined=2, combine="median")
    _, s4 = pt.measure_fluxes(img, [pos, e_xy], cfg, r_ap=8, r_in=12, r_out=18,
                              n_combined=4, combine="median")
    assert np.allclose(s2, s1 / np.sqrt(2.0))                       # N=2 exact
    assert np.allclose(s4, s1 / np.sqrt(4.0 / _MEDIAN_VAR_INFLATION))
    # N=1 leaves the single-exposure σ untouched (back-compatible default)
    _, s1b = pt.measure_fluxes(img, [pos, e_xy], cfg, r_ap=8, r_in=12, r_out=18,
                               n_combined=1)
    assert np.allclose(s1b, s1)


def test_effective_n_sigma_matches_empirical_scatter(cfg):
    """MC validity: after median-combining N frames, the reported σ matches the
    true flux scatter — i.e. the pi/2 median penalty is the correct factor."""
    rng = np.random.default_rng(1234)
    pos = (64.0, 64.0)
    n_comb, n_trials = 5, 250
    scene = pt.make_scene([pos], [(1, 0.0, 0.0, 0)], [2.0e5])
    fluxes, sigmas = [], []
    for _ in range(n_trials):
        stack = [pt.render_frame(scene, cfg, 0.0, exptime_s=10.0,
                                 seeing_arcsec=2.0, sky_e_per_px=60.0, rng=rng,
                                 shape=(128, 128)) for _ in range(n_comb)]
        comb = np.median(np.array(stack, dtype=float), axis=0)
        f, s = pt.measure_fluxes(comb, [pos], cfg, r_ap=8, r_in=12, r_out=18,
                                 n_combined=n_comb, combine="median")
        fluxes.append(f[0])
        sigmas.append(s[0])
    emp = np.std(fluxes, ddof=1)
    rep = np.mean(sigmas)
    # reported and empirical agree to ~15% (correct error bars after combine)
    assert abs(rep - emp) / emp < 0.15


# --------------------------------------------------------------------------- #
# C-1 — per-pixel read-noise map
# --------------------------------------------------------------------------- #
def test_constant_ron_map_reproduces_scalar(cfg, rng):
    pos = (128.0, 128.0)
    scene = pt.make_scene([pos], [(1, 0.0, 0.0, 0)], [3.0e5])
    img = pt.render_frame(scene, cfg, 0.0, exptime_s=10.0, seeing_arcsec=1.0,
                          sky_e_per_px=50.0, rng=rng, shape=(256, 256))
    ron_map = np.full(img.shape, cfg.read_noise_e, dtype=float)
    _, s_scalar = pt.measure_fluxes(img, [pos], cfg, r_ap=8, r_in=12, r_out=18)
    _, s_map = pt.measure_fluxes(img, [pos], cfg, r_ap=8, r_in=12, r_out=18,
                                 ron_map=ron_map)
    assert np.allclose(s_scalar, s_map)


def test_elevated_ron_in_aperture_raises_sigma(cfg, rng):
    pos = (128.0, 128.0)
    scene = pt.make_scene([pos], [(1, 0.0, 0.0, 0)], [3.0e5])
    img = pt.render_frame(scene, cfg, 0.0, exptime_s=10.0, seeing_arcsec=1.0,
                          sky_e_per_px=50.0, rng=rng, shape=(256, 256))
    ron_map = np.full(img.shape, cfg.read_noise_e, dtype=float)
    ron_map[120:137, 120:137] = 30.0  # an RTN/hot patch over the aperture
    _, s_base = pt.measure_fluxes(img, [pos], cfg, r_ap=8, r_in=12, r_out=18)
    _, s_hi = pt.measure_fluxes(img, [pos], cfg, r_ap=8, r_in=12, r_out=18,
                                ron_map=ron_map)
    assert s_hi[0] > s_base[0]


def test_ron_map_shape_mismatch_raises(cfg, rng):
    pos = (128.0, 128.0)
    scene = pt.make_scene([pos], [(1, 0.0, 0.0, 0)], [3.0e5])
    img = pt.render_frame(scene, cfg, 0.0, exptime_s=10.0, rng=rng,
                          shape=(256, 256))
    with pytest.raises(ValueError):
        pt.measure_fluxes(img, [pos], cfg, r_ap=8, r_in=12, r_out=18,
                          ron_map=np.ones((10, 10)))


# --------------------------------------------------------------------------- #
# C-2 — bad/hot-pixel repair (the N=1 vulnerability)
# --------------------------------------------------------------------------- #
def test_bad_pixel_mask_repairs_hot_pixel(cfg, rng):
    """A hot pixel in the o-aperture inflates the flux at N=1; the mask repairs
    it back toward the clean value (median-combine cannot help at N=1)."""
    pos = (128.0, 128.0)
    scene = pt.make_scene([pos], [(1, 0.0, 0.0, 0)], [3.0e5])
    clean = pt.render_frame(scene, cfg, 0.0, exptime_s=10.0, seeing_arcsec=2.0,
                            sky_e_per_px=50.0, rng=rng, shape=(256, 256)
                            ).astype(float)
    f_clean, _ = pt.measure_fluxes(clean, [pos], cfg, r_ap=8, r_in=12, r_out=18)

    hot = clean.copy()
    hy, hx = 131, 131  # a few px off-core, inside the r_ap=8 aperture
    hot[hy, hx] = 60000.0
    f_hot, _ = pt.measure_fluxes(hot, [pos], cfg, r_ap=8, r_in=12, r_out=18)

    mask = np.zeros(clean.shape, dtype=bool)
    mask[hy, hx] = True
    f_fix, _ = pt.measure_fluxes(hot, [pos], cfg, r_ap=8, r_in=12, r_out=18,
                                 bad_pixel_mask=mask)

    assert f_hot[0] > 1.2 * f_clean[0]                       # hot pixel inflates
    assert abs(f_fix[0] - f_clean[0]) < 0.2 * abs(f_hot[0] - f_clean[0])
    assert abs(f_fix[0] - f_clean[0]) / f_clean[0] < 0.03    # back within ~3%


# --------------------------------------------------------------------------- #
# C-4 — 2-D dark frame (FPN / hot-pixel structure) in the simulator
# --------------------------------------------------------------------------- #
def test_dark_frame_injects_structure(cfg, rng):
    scene = pt.make_scene([], [], [])
    base = pt.render_frame(scene, cfg, 0.0, exptime_s=10.0, sky_e_per_px=0.0,
                           rng=np.random.default_rng(7), shape=(64, 64),
                           add_dark=False)
    dark = np.zeros((64, 64), dtype=float)
    dark[32, 32] = 5000.0  # a single hot dark pixel
    withdark = pt.render_frame(scene, cfg, 0.0, exptime_s=10.0, sky_e_per_px=0.0,
                               rng=np.random.default_rng(7), shape=(64, 64),
                               dark_frame=dark)
    assert int(withdark[32, 32]) - int(base[32, 32]) > 4000


def test_dark_frame_shape_mismatch_raises(cfg, rng):
    scene = pt.make_scene([], [], [])
    with pytest.raises(ValueError):
        pt.render_frame(scene, cfg, 0.0, exptime_s=10.0, rng=rng,
                        shape=(64, 64), dark_frame=np.zeros((10, 10)))


# --------------------------------------------------------------------------- #
# C-5 — gain-mode provenance carried through the config
# --------------------------------------------------------------------------- #
def test_config_provenance_and_sat_limit(sensor):
    cfg = PolConfig(sensor=sensor, beam=pt.BeamGeometry(separation_px=40.0),
                    read_noise_e=1.5, readout_mode="Mode1", gain_setting=56,
                    linearity_limit_e=40000.0)
    # sat_limit_e falls back to full_well when unset, else the linearity limit
    assert cfg.sat_limit_e() == 40000.0
    cfg_default = PolConfig(sensor=sensor, beam=pt.BeamGeometry(separation_px=40.0))
    assert cfg_default.sat_limit_e() == cfg_default.full_well_e
    # with_hwp_angles preserves every other field (dataclasses.replace)
    cfg2 = cfg.with_hwp_angles((0.0, 22.5, 45.0, 67.5, 90.0))
    assert cfg2.read_noise_e == 1.5
    assert cfg2.readout_mode == "Mode1"
    assert cfg2.gain_setting == 56
    assert cfg2.linearity_limit_e == 40000.0
    assert cfg2.hwp_angles_deg == (0.0, 22.5, 45.0, 67.5, 90.0)


# --------------------------------------------------------------------------- #
# C-6 — saturation / linearity flag
# --------------------------------------------------------------------------- #
def test_saturation_flag_on_measure_pair(cfg, rng):
    pos = (128.0, 128.0)
    dx, dy = cfg.beam.offset_xy()
    e_xy = (pos[0] + dx, pos[1] + dy)
    img = np.full((256, 256), 1050.0, dtype=float)  # bias + a little sky
    # saturate the o-core: (peak-bias)*gain >= full_well_e
    img[126:131, 126:131] = 60000.0
    bf = pt.measure_pair(img, pos, e_xy, cfg, hwp_deg=0.0, r_ap=8, r_in=12,
                         r_out=18)
    assert bf.saturated is True
    # a clean frame is not flagged
    clean = np.full((256, 256), 1050.0, dtype=float)
    clean[126:131, 126:131] = 5000.0
    bf2 = pt.measure_pair(clean, pos, e_xy, cfg, hwp_deg=0.0, r_ap=8, r_in=12,
                          r_out=18)
    assert bf2.saturated is False


def test_pipeline_warns_and_flags_saturation(sensor, rng, tmp_path):
    """A saturating source raises a warning and sets the result metadata flag."""
    cfg = PolConfig(sensor=sensor, beam=pt.BeamGeometry(separation_px=40.0),
                    hwp_angles_deg=(0.0, 22.5, 45.0, 67.5),
                    linearity_limit_e=30000.0)
    positions = [(128.0, 128.0)]
    scene = pt.make_scene(positions, [(1, 0.02, -0.01, 0)], [1.0e7])  # very bright
    paths = pt.simulate_sequence(scene, cfg, out_dir=tmp_path, exptime_s=15.0,
                                 seeing_arcsec=2.0, sky_e_per_px=60.0, rng=rng,
                                 shape=(256, 256))
    with pytest.warns(UserWarning, match="saturation"):
        res = pt.reduce_to_stokes([str(p) for p in paths], cfg,
                                  o_positions=positions, method="double_ratio",
                                  r_ap=7, r_in=12, r_out=20)[0]
    assert res.metadata["saturated"] is True
    assert len(res.metadata["saturated_angles"]) > 0
