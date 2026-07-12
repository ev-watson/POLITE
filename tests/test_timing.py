from __future__ import annotations

from datetime import datetime, timezone

import pytest

from obs_utils.timing import (
    ParsedW32tmStatus,
    TimingConfig,
    evaluate_w32tm_status,
    lab_timing_snapshot,
    parse_w32tm_status,
)


W32TM_SAMPLE = """\
Leap Indicator: 0(no warning)
Stratum: 3 (secondary reference - syncd by (sn)(IPv4 address))
Precision: -23 (119.209ns per tick)
Root Delay: 0.0156250s
Root Dispersion: 0.0312500s
ReferenceId: 0x0A0B0C0D
Last Successful Sync Time: 7/8/2026 3:45:12 PM
Source: 0.pool.ntp.org
Poll Interval: 10 (1024s)

Phase Offset: 0.1234567s
"""


def test_parse_w32tm_status_extracts_fields():
    now = datetime(2026, 7, 8, 22, 50, 0, tzinfo=timezone.utc)
    parsed = parse_w32tm_status(W32TM_SAMPLE, now=now)

    assert parsed.source == "0.pool.ntp.org"
    assert parsed.phase_offset_s == pytest.approx(0.1234567)
    assert parsed.last_sync_local is not None
    assert parsed.sync_age_s is not None
    assert parsed.sync_age_s > 0


def test_evaluate_w32tm_warn_only_never_raises():
    config = TimingConfig(abort_on_failure=False, warn_offset_s=0.5, abort_offset_s=2.0)
    parsed = ParsedW32tmStatus(phase_offset_s=5.0, sync_age_s=100.0)

    warnings = evaluate_w32tm_status(parsed, config)

    assert warnings
    assert any("offset" in w.lower() for w in warnings)


def test_evaluate_w32tm_abort_when_enabled():
    config = TimingConfig(abort_on_failure=True, abort_offset_s=2.0)
    parsed = ParsedW32tmStatus(phase_offset_s=3.0, sync_age_s=10.0)

    with pytest.raises(Exception):
        evaluate_w32tm_status(parsed, config)


def test_lab_snapshot_uses_lab_local():
    status = lab_timing_snapshot(TimingConfig())

    assert status.timesrc == "LAB-LOCAL"
    assert status.timeunc_s == 1.5
    assert status.ntpoffs_s is None
    assert status.ntpage_s is None
