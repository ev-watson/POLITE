"""Regression tests for outlier-robust QA gates (obs_utils.qa_lib)."""
import numpy as np
import pytest
from astropy.io import fits

from obs_utils.qa_lib import QAResult, run_bias_qa, run_sequence_audit


def _write_bias(tmp_path, n=4, level=300.0, ron_adu=3.5, egain=1.0,
                n_cosmic=0, cosmic_adu=5000.0, n_pinned=0, seed=0):
    rng = np.random.default_rng(seed)
    paths = []
    for i in range(n):
        img = rng.normal(level, ron_adu, (128, 128)).astype(np.float32)
        if n_cosmic:
            ys = rng.integers(0, 128, n_cosmic)
            xs = rng.integers(0, 128, n_cosmic)
            img[ys, xs] += cosmic_adu
        if n_pinned:
            ys = rng.integers(0, 128, n_pinned)
            xs = rng.integers(0, 128, n_pinned)
            img[ys, xs] = -1.0
        hdr = fits.Header()
        hdr["IMAGETYP"] = "BIAS"
        hdr["EGAIN"] = egain
        p = tmp_path / f"BIAS_{i:03d}.fits"
        fits.writeto(p, img, hdr, overwrite=True)
        paths.append(p)
    return paths


def test_level_property():
    assert QAResult("t", True).level == "PASS"
    assert QAResult("t", True, warnings=["w"]).level == "WARN"
    assert QAResult("t", False).level == "FAIL"


def test_bias_qa_robust_to_cosmic_rays(tmp_path):
    """Cosmic rays / hot pixels must not inflate RON into a false failure."""
    _write_bias(tmp_path, ron_adu=3.5, egain=1.0, n_cosmic=80)
    r = run_bias_qa([tmp_path], ron_target_e=3.5, ron_tol_e=0.5)
    assert r.passed, r.messages
    assert r.metrics["ron_e"] == pytest.approx(3.5, abs=0.3)


def test_bias_qa_gross_ron_fails(tmp_path):
    """A genuinely high read noise (well beyond tol) still FAILs."""
    _write_bias(tmp_path, ron_adu=8.0, egain=1.0)
    r = run_bias_qa([tmp_path], ron_target_e=3.5, ron_tol_e=0.5)
    assert not r.passed
    assert r.level == "FAIL"


def test_bias_qa_few_pinned_pixels_warn_not_fail(tmp_path):
    """A handful of pinned pixels should WARN, not block capture."""
    _write_bias(tmp_path, ron_adu=3.5, n_pinned=3)
    r = run_bias_qa([tmp_path], ron_target_e=3.5, ron_tol_e=0.5)
    assert r.passed
    assert r.level == "WARN"


def test_sequence_audit_honors_explicit_expected_angle_override(tmp_path):
    for i, angle in enumerate((0.0, 22.5, 45.0, 67.5)):
        header = fits.Header({
            "IMAGETYP": "LIGHT",
            "OBJECT": "standard",
            "POLSEQ": "custom_sequence",
            "FILTER": "Photometric V",
            "HWPANG": angle,
        })
        fits.PrimaryHDU(np.zeros((4, 4)), header).writeto(
            tmp_path / f"frame_{i}.fits"
        )
    default = run_sequence_audit(tmp_path)
    override = run_sequence_audit(
        tmp_path, expected_angles={"custom_sequence": 4}
    )
    assert not default.passed
    assert override.passed
