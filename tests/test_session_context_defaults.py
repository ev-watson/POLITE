"""Detector operating point and ``camera:`` block loading.

The default is Mode 5 / gain 56 / offset 20 (see obs_utils.session_context).  A
missing block or an unrecognized key must WARN and continue -- running on the
defaults is legitimate, but it has to be visible, and ``offest: 20`` must not pass
for ``offset: 20`` in silence.
"""
import logging

import pytest

from obs_utils.session_context import SessionCaptureContext, session_context_from_yaml


def test_default_operating_point_is_mode5_gain56_offset20():
    ctx = SessionCaptureContext()
    assert (ctx.readout_mode, ctx.gain_setting, ctx.offset_setting) == (5, 56, 20)
    assert ctx.readout_mode_name == "Mode 5"


def test_missing_camera_block_warns_and_uses_defaults(caplog):
    with caplog.at_level(logging.WARNING):
        ctx = session_context_from_yaml(None)
    assert (ctx.readout_mode, ctx.gain_setting, ctx.offset_setting) == (5, 56, 20)
    assert "No camera: block" in caplog.text
    # The banner must name what it fell back to, not just that it fell back.
    assert "gain 56" in caplog.text and "offset 20" in caplog.text


def test_unknown_key_warns_by_name_and_is_ignored(caplog):
    with caplog.at_level(logging.WARNING):
        ctx = session_context_from_yaml({"offest": 99, "gain": 56})
    assert ctx.offset_setting == 20          # the typo did NOT set offset
    assert ctx.gain_setting == 56           # the good key still applied
    assert "offest" in caplog.text
    assert "unrecognized" in caplog.text


def test_retired_egain_ron_keys_are_reported_not_absorbed(caplog):
    """Conversion gain and read noise are reduction results, never plan inputs."""
    with caplog.at_level(logging.WARNING):
        session_context_from_yaml({"egain_e_per_adu": 1.0, "ron_e": 3.5})
    assert "egain_e_per_adu" in caplog.text and "ron_e" in caplog.text


def test_recognized_keys_still_load_without_warning(caplog):
    with caplog.at_level(logging.WARNING):
        ctx = session_context_from_yaml(
            {"readout_mode": 3, "readout_mode_name": "Mode 3", "gain": 0,
             "offset": 20, "cooler_setpoint_c": -15.0, "cooler_policy": "exact"}
        )
    # Mode 3 / gain 0 -- the high-full-well alternative operating point.
    assert (ctx.readout_mode, ctx.gain_setting, ctx.offset_setting) == (3, 0, 20)
    assert ctx.readout_mode_name == "Mode 3"
    assert ctx.cooler_policy == "exact"
    assert caplog.text == ""


def test_bad_cooler_policy_still_raises():
    with pytest.raises(ValueError, match="cooler_policy"):
        session_context_from_yaml({"cooler_policy": "manual"})
