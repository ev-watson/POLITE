"""Simulator + I/O: FITS byte-compatibility and polarimetry metadata."""

import numpy as np
import pytest
from astropy.io import fits

import caltools as ct
import poltools as pt


def test_fits_roundtrip_byte_exact(cfg, rng, tmp_path):
    scene = pt.make_scene([(60.0, 70.0)], [(1, 0.03, -0.02, 0)], [5e5])
    frame = pt.render_frame(scene, cfg, 0.0, exptime_s=5.0, seeing_arcsec=2.0,
                            sky_e_per_px=40.0, rng=rng, shape=(128, 128))
    p = pt.write_pol_fits(tmp_path / "f.fit", frame, 0.0, cfg)
    back = ct.load_frame(str(p))
    # caltools reads physical ADU via memmap=False; must equal what we wrote
    assert np.array_equal(back.astype(np.uint16), frame)
    # BZERO present (unsigned-int convention for QHY/TheSkyX)
    hdr = fits.getheader(str(p))
    assert hdr["BZERO"] == 32768
    assert hdr["HWPANG"] == 0.0
    assert hdr["RETARD"] == cfg.retardance_deg
    assert hdr["EGAIN"] == cfg.sensor.gain_e_per_adu


def test_sequence_writes_all_angles_and_groups(cfg, rng, tmp_path):
    scene = pt.make_scene([(60.0, 70.0)], [(1, 0.0, 0.0, 0)], [5e5])
    paths = pt.simulate_sequence(scene, cfg, out_dir=tmp_path, rng=rng,
                                 shape=(128, 128), exptime_s=2.0)
    assert len(paths) == len(cfg.hwp_angles_deg)
    groups = pt.group_by_hwp_angle([str(p) for p in paths])
    assert set(groups.keys()) == set(round(a, 3) for a in cfg.hwp_angles_deg)
    seq = pt.group_pol_sequence([str(p) for p in paths])
    assert list(seq.keys()) == sorted(cfg.hwp_angles_deg)
    assert all(len(group) == 1 for group in seq.values())

    repeated = pt.group_pol_sequence([str(paths[0]), str(paths[0])])
    assert len(repeated[cfg.hwp_angles_deg[0]]) == 2


def test_missing_hwpang_raises_named_error(tmp_path):
    """A frame without HWPANG is a hard, file-named error (not a silent NaN that
    surfaces later as a cryptic angle-lookup miss)."""
    data = np.zeros((32, 32), dtype=np.uint16)
    p = tmp_path / "no_hwp.fit"
    fits.PrimaryHDU(data=data).writeto(p)
    with pytest.raises(ValueError):
        pt.read_pol_frame(str(p))
    with pytest.raises(ValueError):
        pt.group_by_hwp_angle([str(p)])
    with pytest.raises(ValueError):
        pt.group_pol_sequence([str(p)])


def test_make_scene_length_mismatch_raises():
    """make_scene rejects unequal-length inputs instead of silently truncating
    to the shortest via zip()."""
    with pytest.raises(ValueError):
        pt.make_scene([(1.0, 2.0), (3.0, 4.0)], [(1, 0.0, 0.0, 0)], [1e5, 2e5])


def test_render_saturation_clip(cfg, rng):
    # an extremely bright source saturates at the FULL-WELL level (51 ke-),
    # converted to ADU + bias, and never exceeds the 16-bit ADC max.
    bias = 1000.0
    scene = pt.make_scene([(64.0, 64.0)], [(1, 0.0, 0.0, 0)], [1e9])
    frame = pt.render_frame(scene, cfg, 0.0, exptime_s=100.0, seeing_arcsec=2.0,
                            sky_e_per_px=10.0, rng=rng, shape=(128, 128),
                            bias_adu=bias)
    assert frame.dtype == np.uint16
    assert frame.max() <= (1 << cfg.sensor.bitdepth) - 1   # never overflow ADC
    sat_adu = cfg.full_well_e / cfg.sensor.gain_e_per_adu + bias
    # core pixels pinned near the full-well plateau (within Poisson/read noise)
    assert frame.max() >= 0.98 * sat_adu
    assert (frame > 0.98 * sat_adu).sum() > 5   # a saturated core exists


def test_sidecar_roundtrip_preserves_measured_beam_geometry(tmp_path):
    det = pt.SessionDetectorConfig(
        beam_separation_px=239.5,
        beam_position_angle_deg=328.2,
        beam_geometry_characterized=True,
    )
    sidecar = pt.write_pol_config_sidecar(
        tmp_path / "pol_config.yaml", det, session_id="TEST"
    )
    cfg = pt.load_pol_config_sidecar(sidecar, filter_name="Photometric V")
    assert cfg.beam.separation_px == pytest.approx(239.5)
    assert cfg.beam.position_angle_deg == pytest.approx(328.2)
    assert cfg.active_filter().characterized is True


def test_fits_header_beam_geometry_is_characterized(tmp_path):
    data = np.zeros((32, 48), dtype=np.uint16)
    header = fits.Header({
        "FILTER": "Photometric V",
        "BEAMSEP": 239.5,
        "BEAMPA": 328.2,
        "EGAIN": 1.0,
    })
    path = tmp_path / "geometry.fits"
    fits.PrimaryHDU(data, header=header).writeto(path)
    cfg = pt.polconfig_from_fits_headers(path)
    assert cfg.beam.separation_px == pytest.approx(239.5)
    assert cfg.beam.position_angle_deg == pytest.approx(328.2)
    assert cfg.active_filter().characterized is True
