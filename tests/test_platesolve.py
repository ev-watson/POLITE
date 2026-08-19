"""Read-only PlateSolve3 adapter tests; no camera or PWI4 endpoint involved."""

from __future__ import annotations

import subprocess
from pathlib import Path

import numpy as np
import pytest
from astropy.io import fits

from obs_utils.platesolve import PlateSolveConfig, platesolve
from obs_utils.pointing import ModelBuildConfig, build_pointing_model, map_point


def _plate_solve_paths(tmp_path):
    frame = tmp_path / "raw_camera_frame.fits"
    executable = tmp_path / "ps3cli.exe"
    catalog = tmp_path / "Kepler"
    frame.write_bytes(b"not examined by the fake solver")
    executable.write_bytes(b"not executed by the fake runner")
    catalog.mkdir()
    return frame, executable, catalog


def test_platesolve_preserves_real_solver_fields_without_assuming_wcs_keys(tmp_path):
    frame, executable, catalog = _plate_solve_paths(tmp_path)
    seen = {}

    def runner(args, **kwargs):
        seen["args"] = args
        seen["kwargs"] = kwargs
        Path(args[-2]).write_text("solver_status=1\nfield_rotation=12.5\n", encoding="utf-8")
        return subprocess.CompletedProcess(args, 0, stdout="solver output", stderr="")

    result = platesolve(
        frame,
        0.224,
        PlateSolveConfig(executable, catalog, command_prefix=("mono",)),
        runner=runner,
    )

    assert seen["args"][:2] == ["mono", str(executable)]
    assert seen["args"][2] == str(frame)
    assert seen["args"][3] == "0.224"
    assert seen["kwargs"]["timeout"] == 120.0
    assert result.raw_fields == {"solver_status": 1.0, "field_rotation": 12.5}
    assert result.stdout == "solver output"


def test_platesolve_hands_a_synthetic_fits_to_solver_unchanged(tmp_path):
    """Exercise the FITS hand-off without claiming a catalogue-match test.

    The real PS3CLI executable and Kepler catalogue live on the observatory
    machine, not CI.  This verifies the adapter's complete local protocol with
    a realistic temporary image while a fake executable stands in for PS3CLI.
    """
    frame, executable, catalog = _plate_solve_paths(tmp_path)
    yy, xx = np.indices((96, 128))
    data = 1000.0 + sum(
        amplitude * np.exp(-((xx - x) ** 2 + (yy - y) ** 2) / 18.0)
        for x, y, amplitude in ((20.0, 30.0, 8000.0), (70.0, 42.0, 6000.0), (100.0, 78.0, 5000.0))
    )
    fits.PrimaryHDU(data.astype(np.float32)).writeto(frame, overwrite=True)
    original = frame.read_bytes()

    def runner(args, **_kwargs):
        image_argument = args.index(str(frame))
        with fits.open(args[image_argument]) as hdul:
            np.testing.assert_allclose(hdul[0].data, data)
        assert frame.read_bytes() == original
        Path(args[image_argument + 2]).write_text("solve_status=1\n", encoding="utf-8")
        return subprocess.CompletedProcess(args, 0, stdout="synthetic accepted", stderr="")

    result = platesolve(frame, 0.224, PlateSolveConfig(executable, catalog), runner=runner)

    assert result.raw_fields == {"solve_status": 1.0}
    assert frame.read_bytes() == original


def test_platesolve_rejects_missing_local_inputs_before_running(tmp_path):
    frame, executable, catalog = _plate_solve_paths(tmp_path)
    with pytest.raises(FileNotFoundError, match="executable"):
        platesolve(frame, 0.224, PlateSolveConfig(tmp_path / "missing.exe", catalog))
    with pytest.raises(ValueError, match="positive finite"):
        platesolve(frame, float("nan"), PlateSolveConfig(executable, catalog))


def test_platesolve_reports_nonzero_solver_exit(tmp_path):
    frame, executable, catalog = _plate_solve_paths(tmp_path)

    def runner(args, **_kwargs):
        return subprocess.CompletedProcess(args, 2, stdout="", stderr="no match")

    with pytest.raises(RuntimeError, match="code 2: no match"):
        platesolve(frame, 0.224, PlateSolveConfig(executable, catalog), runner=runner)


def test_pointing_model_functions_are_motion_free_until_ps3_is_commissioned(tmp_path):
    class MustNotMove:
        def mount_goto_alt_az(self, *_args):  # pragma: no cover - must not run
            raise AssertionError("pointing-model guard attempted a slew")

    config = PlateSolveConfig(tmp_path / "ps3cli.exe", tmp_path / "Kepler")
    with pytest.raises(RuntimeError, match="disabled"):
        map_point(
            MustNotMove(), 30.0, 90.0, lambda _path: None, config, 0.224,
            tmp_path / "image.fits",
        )

    image_path = tmp_path / "would-be-output" / "image.fits"
    with pytest.raises(RuntimeError, match="disabled"):
        build_pointing_model(
            MustNotMove(), config, ModelBuildConfig(0.224, image_path=image_path),
        )
    assert not image_path.parent.exists()
