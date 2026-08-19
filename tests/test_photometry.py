"""Photometry: detection, o/e pairing, and aperture flux recovery."""

from dataclasses import replace

import numpy as np
import pytest

import poltools as pt
import poltools.photometry as photometry


def test_measure_fluxes_requires_gain(cfg):
    """Conversion gain is a per-night characterization value, not header state:
    reduction fails closed (clear error) rather than using a silent nominal."""
    cfg_nogain = replace(cfg, sensor=replace(cfg.sensor, gain_e_per_adu=None))
    img = np.zeros((64, 64), dtype=float)
    with pytest.raises(ValueError, match="gain"):
        pt.measure_fluxes(img, [(32.0, 32.0)], cfg_nogain, r_ap=6, r_in=9, r_out=14)


def test_measure_fluxes_requires_read_noise(cfg):
    """Read noise is likewise per-night; absent it (and no ron_map) -> clear error."""
    cfg_noron = replace(cfg, read_noise_e=None)
    img = np.zeros((64, 64), dtype=float)
    with pytest.raises(ValueError, match="read noise"):
        pt.measure_fluxes(img, [(32.0, 32.0)], cfg_noron, r_ap=6, r_in=9, r_out=14)


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


def test_brightest_point_source_ranks_integrated_flux_not_hot_pixel():
    """A single saturated defect must not outrank a resolved stellar source."""
    yy, xx = np.indices((256, 256))
    image = np.full((256, 256), 1000.0)
    faint = 9000.0 * np.exp(-((xx - 70.0) ** 2 + (yy - 80.0) ** 2) / (2 * 9.0 ** 2))
    bright = 15000.0 * np.exp(-((xx - 180.0) ** 2 + (yy - 160.0) ** 2) / (2 * 12.0 ** 2))
    image += faint + bright
    image[20, 20] = 65535.0

    x, y = pt.brightest_point_source(image, source_scale_px=20.0)
    assert x == pytest.approx(180.0, abs=1.0)
    assert y == pytest.approx(160.0, abs=1.0)


def test_brightest_point_source_respects_bad_pixel_mask():
    yy, xx = np.indices((256, 256))
    image = 1000.0 + 12000.0 * np.exp(
        -((xx - 80.0) ** 2 + (yy - 90.0) ** 2) / (2 * 10.0 ** 2)
    ) + 18000.0 * np.exp(
        -((xx - 180.0) ** 2 + (yy - 160.0) ** 2) / (2 * 10.0 ** 2)
    )
    bad = np.zeros_like(image, dtype=bool)
    bad[130:191, 150:211] = True

    x, y = pt.brightest_point_source(image, source_scale_px=20.0, bad_pixel_mask=bad)
    assert x == pytest.approx(80.0, abs=1.0)
    assert y == pytest.approx(90.0, abs=1.0)


def _paired_scene(shape=(800, 800)):
    """Three dual-beam sources with a common, near-nominal detector split."""
    yy, xx = np.indices(shape)
    image = np.full(shape, 1000.0)
    offset = (18.0, 238.0)
    sources = ((180.0, 150.0, 9000.0), (500.0, 180.0, 15000.0),
               (270.0, 470.0, 11000.0))
    for x, y, amp in sources:
        for px, py in ((x, y), (x + offset[0], y + offset[1])):
            image += amp * np.exp(
                -((xx - px) ** 2 + (yy - py) ** 2) / (2 * 8.0 ** 2)
            )
    return image, offset


def test_propose_anchor_pair_measures_common_axis_and_brightest_pair():
    image, offset = _paired_scene()
    proposal = pt.propose_anchor_pair(
        image, source_scales_px=(8.0, 16.0), threshold_sigma=5.0,
    )
    assert proposal.supporting_pair_count >= 3
    assert len(proposal.matched_pairs) == 3
    assert proposal.separation_px == pytest.approx(np.hypot(*offset), abs=1.0)
    assert proposal.axis_angle_deg == pytest.approx(
        np.degrees(np.arctan2(offset[0], offset[1])), abs=1.0
    )
    expected = {(500, 180), (518, 418)}
    got = {tuple(round(v) for v in proposal.beam_a_xy),
           tuple(round(v) for v in proposal.beam_b_xy)}
    assert got == expected


def test_anchor_proposal_exposes_canonical_pair_anchors():
    image, _offset = _paired_scene()
    proposal = pt.propose_anchor_pair(
        image, source_scales_px=(8.0, 16.0), threshold_sigma=5.0,
    )
    assert proposal.as_pair_anchors() == {
        "a": proposal.beam_a_xy,
        "b": proposal.beam_b_xy,
    }
    assert proposal.beam_geometry.separation_px == pytest.approx(proposal.separation_px)


def test_propose_anchor_pair_fails_without_repeated_split():
    yy, xx = np.indices((512, 512))
    image = 1000.0 + 10000.0 * np.exp(
        -((xx - 150.0) ** 2 + (yy - 150.0) ** 2) / (2 * 8.0 ** 2)
    )
    with pytest.raises(ValueError, match="fewer than four|no candidate offsets"):
        pt.propose_anchor_pair(image, source_scales_px=(8.0, 16.0))


def test_track_matched_pair_uses_constellation_translation():
    """The continuing pair is chosen after a large detector translation."""
    previous = (
        ((100.0, 100.0), (118.0, 338.0)),
        ((400.0, 200.0), (418.0, 438.0)),
        ((800.0, 600.0), (818.0, 838.0)),
    )
    drift = np.array((163.0, -224.0))
    current = tuple(
        tuple(tuple(np.asarray(point) + drift) for point in pair)
        for pair in previous
    )
    # This pair is nearer to the old target than the true current target, but
    # does not share the constellation's translation and must not be selected.
    distractor = ((180.0, 60.0), (198.0, 298.0))
    current = (distractor,) + current

    pair, measured_drift, residual = pt.track_matched_pair(
        previous[0], previous, current,
    )

    assert pair == current[1]
    assert measured_drift == pytest.approx(drift)
    assert residual == pytest.approx(0.0)


def test_track_matched_pair_rejects_incoherent_constellation():
    previous = (
        ((100.0, 100.0), (118.0, 338.0)),
        ((400.0, 200.0), (418.0, 438.0)),
    )
    current = (
        ((120.0, 140.0), (138.0, 378.0)),
        ((500.0, 150.0), (518.0, 388.0)),
    )

    with pytest.raises(ValueError, match="common translation"):
        pt.track_matched_pair(previous[0], previous, current)


def test_track_matched_pair_requires_translation_to_continue_selected_pair():
    """A denser unrelated shift cannot override the selected pair's track."""
    previous = (
        ((0.0, 0.0), (18.0, 238.0)),
        ((1000.0, 0.0), (1018.0, 238.0)),
        ((3000.0, 0.0), (3018.0, 238.0)),
        ((6000.0, 0.0), (6018.0, 238.0)),
    )
    true_drift = np.array((163.0, -224.0))
    unrelated_drift = np.array((-500.0, 400.0))
    current = (
        tuple(tuple(np.asarray(point) + true_drift) for point in previous[0]),
        tuple(tuple(np.asarray(point) + true_drift) for point in previous[1]),
        tuple(tuple(np.asarray(point) + unrelated_drift) for point in previous[1]),
        tuple(tuple(np.asarray(point) + unrelated_drift) for point in previous[2]),
        tuple(tuple(np.asarray(point) + unrelated_drift) for point in previous[3]),
    )

    pair, measured_drift, residual = pt.track_matched_pair(
        previous[0], previous, current,
    )

    assert pair == current[0]
    assert measured_drift == pytest.approx(true_drift)
    assert residual == pytest.approx(0.0)


def test_select_trackable_pair_skips_default_pair_that_leaves_detector():
    def pair_at(x, y):
        return ((float(x), float(y)), (float(x + 18), float(y + 238)))

    previous = (pair_at(100, 100), pair_at(1000, 1000), pair_at(3000, 2000))
    drift = np.array((163.0, -224.0))
    current = tuple(
        tuple(tuple(np.asarray(point) + drift) for point in pair)
        for pair in previous[1:]
    )

    first, second, measured_drift, residual = pt.select_trackable_pair(
        previous, current, current[0],
    )

    assert first == previous[1]
    assert second == current[0]
    assert measured_drift == pytest.approx(drift)
    assert residual == pytest.approx(0.0)


def test_track_pair_sequence_continues_one_pair_and_returns_frame_diagnostics(
    monkeypatch,
):
    previous_pairs = (
        ((100.0, 100.0), (118.0, 338.0)),
        ((400.0, 200.0), (418.0, 438.0)),
    )
    drift = np.array((8.0, -12.0))
    current_pairs = tuple(
        tuple(tuple(np.asarray(point) + drift) for point in pair)
        for pair in previous_pairs
    )
    proposals = iter((
        pt.AnchorPairProposal(
            *previous_pairs[0], 238.0, 3.0, 2, 0.0, 1.0, 4, previous_pairs,
        ),
        pt.AnchorPairProposal(
            *current_pairs[0], 238.0, 3.0, 2, 0.0, 1.0, 4, current_pairs,
        ),
    ))
    monkeypatch.setattr(photometry, "propose_anchor_pair", lambda *_args, **_kwargs: next(proposals))
    frames = ({"filename": "first.fits"}, {"filename": "second.fits"})
    data = np.ones((4, 4))

    tracked = pt.track_pair_sequence(
        frames, load_data=lambda _frame: (data, None),
    )

    assert [entry.frame for entry in tracked] == list(frames)
    assert tracked[0].pair_xy == previous_pairs[0]
    assert tracked[0].shift_xy is None
    assert tracked[1].pair_xy == current_pairs[0]
    assert tracked[1].shift_xy == pytest.approx(drift)
    assert tracked[1].prediction_rms_px == pytest.approx(0.0)


def test_show_tracked_sequence_accepts_masked_data(monkeypatch):
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    data = np.arange(100, dtype=float).reshape(10, 10)
    data[0, 0] = np.nan
    mask = np.zeros_like(data, dtype=bool)
    mask[1, 1] = True
    tracked = (
        pt.TrackedPairFrame(
            frame={"filename": "camera_frame.fits"},
            beam_a_xy=(3.0, 3.0), beam_b_xy=(6.0, 6.0),
            shift_xy=None, prediction_rms_px=None,
        ),
    )
    monkeypatch.setattr(plt, "show", lambda: None)

    fig = pt.show_tracked_sequence(
        tracked, load_data=lambda _frame: (data, mask), half_size_px=4,
    )

    assert fig.axes[0].get_title().startswith("camera_frame.fits")
    plt.close(fig)


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


def test_measure_pair_matches_measure_fluxes(cfg, rng):
    """measure_pair (single o/e convenience) equals the underlying batched
    measure_fluxes and packs them into a BeamFlux."""
    pos = (128.0, 128.0)
    scene = pt.make_scene([pos], [(1, 0.0, 0.0, 0)], [3.0e5])
    img = pt.render_frame(scene, cfg, 0.0, exptime_s=10.0, seeing_arcsec=1.0,
                          sky_e_per_px=50.0, rng=rng, shape=(256, 256),
                          bias_adu=1000.0)
    dx, dy = cfg.beam.offset_xy()
    e_xy = (pos[0] + dx, pos[1] + dy)
    bf = pt.measure_pair(img, pos, e_xy, cfg, hwp_deg=22.5, r_ap=8, r_in=12,
                         r_out=18, bias_adu=1000.0)
    flux, sig = pt.measure_fluxes(img, [pos, e_xy], cfg, r_ap=8, r_in=12,
                                  r_out=18, bias_adu=1000.0)
    assert bf.hwp_deg == 22.5
    assert bf.f_o == pytest.approx(flux[0])
    assert bf.f_e == pytest.approx(flux[1])
    assert bf.sig_o == pytest.approx(sig[0])
    assert bf.sig_e == pytest.approx(sig[1])
