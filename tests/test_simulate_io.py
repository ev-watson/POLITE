"""Simulator + I/O: FITS byte-compatibility and polarimetry metadata."""

import numpy as np
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
