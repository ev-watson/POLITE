"""Master-frame I/O is scaled correctly and genuinely row-chunked."""

from __future__ import annotations

import numpy as np
import pytest
from astropy.io import fits

import caltools as ct


def _write_uint16(path, data):
    fits.PrimaryHDU(np.asarray(data, dtype=np.uint16)).writeto(path)
    return str(path)


def test_load_cube_chunked_reconstructs_scaled_uint16_with_roi(tmp_path):
    arrays = [
        (np.arange(48, dtype=np.uint16).reshape(6, 8) + 400 * i)
        for i in range(3)
    ]
    paths = [_write_uint16(tmp_path / f"f{i}.fits", a)
             for i, a in enumerate(arrays)]
    roi = (slice(1, 6, 2), slice(2, 8, 2))
    pieces = list(ct.load_cube_chunked(paths, chunk_rows=2, roi=roi))
    rebuilt = np.concatenate([chunk for _, chunk in pieces], axis=1)
    expected = np.stack([a[roi] for a in arrays]).astype(np.float32)
    np.testing.assert_array_equal(rebuilt, expected)
    assert [sl for sl, _ in pieces] == [slice(0, 2), slice(2, 3)]


def test_load_cube_chunked_validates_inputs():
    with pytest.raises(ValueError, match="at least one"):
        list(ct.load_cube_chunked([]))
    with pytest.raises(ValueError, match="positive"):
        list(ct.load_cube_chunked(["unused.fits"], chunk_rows=0))


def test_master_bias_rejects_unknown_reducer(tmp_path):
    path = _write_uint16(tmp_path / "bias.fits", np.ones((4, 4)))
    with pytest.raises(ValueError, match="median.*mean"):
        ct.master_bias([path], method="mode")


def test_master_flat_rejects_zero_signal(tmp_path):
    bias = np.full((4, 4), 500.0, dtype=np.float32)
    paths = [
        _write_uint16(tmp_path / f"flat{i}.fits", np.full((4, 4), 500))
        for i in range(3)
    ]
    with pytest.raises(ValueError, match="Cannot normalize"):
        ct.master_flat(paths, bias, normalize=True, chunk_rows=2)


def test_path_read_noise_matches_cube_implementation(tmp_path):
    rng = np.random.default_rng(12)
    arrays = [
        np.rint(1000 + rng.normal(0, 4, size=(9, 11))).astype(np.uint16)
        for _ in range(8)
    ]
    paths = [_write_uint16(tmp_path / f"bias{i}.fits", a)
             for i, a in enumerate(arrays)]
    cube = ct.load_cube(paths)
    expected_rn, expected_temporal = ct.read_noise_map(cube)
    got_rn, got_temporal = ct.read_noise_map_from_paths(paths, chunk_rows=3)
    np.testing.assert_allclose(got_rn, expected_rn, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(
        got_temporal, expected_temporal, rtol=1e-6, atol=1e-6
    )


def test_exposure_grouping_collapses_float_artifacts_not_real_milliseconds(
    tmp_path,
):
    paths = []
    for i, exposure in enumerate((0.2, 0.2000000000109, 0.201)):
        header = fits.Header({"IMAGETYP": "DARK", "EXPTIME": exposure})
        path = tmp_path / f"dark_{i}.fits"
        fits.PrimaryHDU(np.zeros((2, 2), dtype=np.uint16), header).writeto(path)
        paths.append(str(path))
    groups = ct.group_by_type_and_exposure(paths)
    assert len(groups[("DARK", 0.2)]) == 2
    assert len(groups[("DARK", 0.201)]) == 1
