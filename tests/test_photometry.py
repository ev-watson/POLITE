"""Photometry: detection, o/e pairing, and aperture flux recovery."""

import numpy as np

import poltools as pt


def test_detect_and_pair(cfg, rng):
    positions = [(60.0, 70.0), (160.0, 120.0)]
    scene = pt.make_scene(positions, [(1, 0.0, 0.0, 0)] * 2, [2e6, 2e6])
    img = pt.render_frame(scene, cfg, 0.0, exptime_s=10.0, seeing_arcsec=2.0,
                          sky_e_per_px=60.0, rng=rng, shape=(256, 256))
    det = pt.detect_sources(img, cfg.fwhm_px(2.0), threshold_sigma=8.0)
    pairs = pt.pair_oe(det, cfg.beam)
    assert len(pairs) == 2
    found_o = sorted(o for (o, e) in pairs)
    for fo, tp in zip(found_o, sorted(positions)):
        assert abs(fo[0] - tp[0]) < 1.5 and abs(fo[1] - tp[1]) < 1.5


def test_aperture_flux_recovers_injected(cfg, rng):
    """Net aperture flux ≈ injected electrons (within a few % for a clean PSF)."""
    pos = (128.0, 128.0)
    total_e = 3.0e5
    # unpolarized -> o and e each get half the flux at all angles.
    # Sharp seeing so a modest aperture captures ~all the PSF and the sky
    # annulus is free of the partner beam (sep=40px).
    scene = pt.make_scene([pos], [(1, 0.0, 0.0, 0)], [total_e])
    img = pt.render_frame(scene, cfg, 0.0, exptime_s=10.0, seeing_arcsec=1.0,
                          sky_e_per_px=50.0, rng=rng, shape=(256, 256),
                          bias_adu=1000.0)
    dx, dy = cfg.beam.offset_xy()
    flux, sig = pt.measure_fluxes(img, [pos, (pos[0] + dx, pos[1] + dy)], cfg,
                                  r_ap=8, r_in=12, r_out=18, bias_adu=1000.0)
    # each beam should hold ~half the total flux (gain=1 -> e- == ADU net)
    assert abs(flux[0] - total_e / 2) / (total_e / 2) < 0.03
    assert abs(flux[1] - total_e / 2) / (total_e / 2) < 0.03
    # uncertainty is positive and order sqrt(flux)
    assert np.all(sig > 0)
