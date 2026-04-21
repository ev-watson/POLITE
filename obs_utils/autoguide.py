from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Iterable, Optional, Union
import random

try:
    from alpaca.camera import GuideDirections
except Exception:  # pragma: no cover - fallback if alpyca is not importable at load time
    from enum import IntEnum

    class GuideDirections(IntEnum):
        North = 0
        South = 1
        East = 2
        West = 3

from alpyca_tools.camera_device import CameraDevice

from .alpaca import ImagingSession
from .pwi4_client import PWI4


@dataclass
class GuidePulse:
    direction: GuideDirections
    duration_ms: int


def _parse_direction(direction: Union[GuideDirections, str, int]) -> GuideDirections:
    if isinstance(direction, GuideDirections):
        return direction
    if isinstance(direction, int):
        return GuideDirections(direction)

    text = str(direction).strip().lower()
    mapping = {
        "n": GuideDirections.North,
        "north": GuideDirections.North,
        "s": GuideDirections.South,
        "south": GuideDirections.South,
        "e": GuideDirections.East,
        "east": GuideDirections.East,
        "w": GuideDirections.West,
        "west": GuideDirections.West,
    }
    if text not in mapping:
        raise ValueError(f"Unknown guide direction: {direction}")
    return mapping[text]


def _ensure_can_pulse(camera: CameraDevice) -> None:
    try:
        can_pulse = bool(camera.CanPulseGuide)
    except Exception:
        can_pulse = False
    if not can_pulse:
        raise RuntimeError(
            "Camera does not support pulse guiding (CanPulseGuide is false)"
        )


def pulse_guide(
    camera: CameraDevice,
    direction: Union[GuideDirections, str, int],
    duration_ms: int,
) -> None:
    _ensure_can_pulse(camera)
    direction_enum = _parse_direction(direction)
    try:
        camera.PulseGuide(direction_enum, int(duration_ms))
    except Exception as exc:
        raise RuntimeError(
            "PulseGuide failed; driver may not implement it"
        ) from exc


def dither_mount_offset_arcsec(
    pwi4: PWI4,
    ra_arcsec: float,
    dec_arcsec: float,
    settle_s: float = 0.0,
) -> None:
    """
    Dither using PWI4 mount offsets (no PulseGuide required).

    This is a good fit when the guide camera driver exposes the guide sensor for imaging
    but does not implement ASCOM/Alpaca PulseGuide.
    """
    kwargs = {}
    if ra_arcsec != 0:
        kwargs["ra_add_arcsec"] = float(ra_arcsec)
    if dec_arcsec != 0:
        kwargs["dec_add_arcsec"] = float(dec_arcsec)
    if not kwargs:
        return

    pwi4.mount_offset(**kwargs)
    if settle_s > 0:
        time.sleep(settle_s)


def apply_pulses(
    camera: CameraDevice,
    pulses: Iterable[GuidePulse],
    settle_s: float = 0.0,
) -> None:
    for pulse in pulses:
        pulse_guide(camera, pulse.direction, pulse.duration_ms)
        if settle_s > 0:
            time.sleep(settle_s)


def _pulse_duration_ms(
    offset_arcsec: float,
    guide_rate_arcsec_per_s: float,
    min_duration_ms: int,
    max_duration_ms: int,
) -> int:
    if guide_rate_arcsec_per_s <= 0:
        raise ValueError("guide_rate_arcsec_per_s must be > 0")
    duration_ms = int(round(1000.0 * abs(offset_arcsec) / guide_rate_arcsec_per_s))
    return max(min_duration_ms, min(max_duration_ms, duration_ms))


def offsets_to_pulses(
    ra_offset_arcsec: float,
    dec_offset_arcsec: float,
    guide_rate_arcsec_per_s: float,
    min_duration_ms: int = 50,
    max_duration_ms: int = 2000,
    invert_ra: bool = False,
    invert_dec: bool = False,
) -> list[GuidePulse]:
    pulses: list[GuidePulse] = []

    if ra_offset_arcsec != 0:
        ra_sign = -1.0 if invert_ra else 1.0
        ra_dir = GuideDirections.West if ra_offset_arcsec * ra_sign > 0 else GuideDirections.East
        ra_dur = _pulse_duration_ms(
            ra_offset_arcsec,
            guide_rate_arcsec_per_s,
            min_duration_ms,
            max_duration_ms,
        )
        pulses.append(GuidePulse(direction=ra_dir, duration_ms=ra_dur))

    if dec_offset_arcsec != 0:
        dec_sign = -1.0 if invert_dec else 1.0
        dec_dir = GuideDirections.North if dec_offset_arcsec * dec_sign > 0 else GuideDirections.South
        dec_dur = _pulse_duration_ms(
            dec_offset_arcsec,
            guide_rate_arcsec_per_s,
            min_duration_ms,
            max_duration_ms,
        )
        pulses.append(GuidePulse(direction=dec_dir, duration_ms=dec_dur))

    return pulses


def autoguide_from_offsets(
    session: ImagingSession,
    ra_offset_arcsec: float,
    dec_offset_arcsec: float,
    guide_rate_arcsec_per_s: float,
    min_duration_ms: int = 50,
    max_duration_ms: int = 2000,
    invert_ra: bool = False,
    invert_dec: bool = False,
    settle_s: float = 0.0,
    require_guide_camera: bool = True,
) -> None:
    if require_guide_camera and session.guide_camera is None:
        raise RuntimeError("Guide camera is required for autoguiding but is not connected")
    camera = session.guide_camera or session.camera
    pulses = offsets_to_pulses(
        ra_offset_arcsec=ra_offset_arcsec,
        dec_offset_arcsec=dec_offset_arcsec,
        guide_rate_arcsec_per_s=guide_rate_arcsec_per_s,
        min_duration_ms=min_duration_ms,
        max_duration_ms=max_duration_ms,
        invert_ra=invert_ra,
        invert_dec=invert_dec,
    )
    apply_pulses(camera, pulses, settle_s=settle_s)


def random_dither_mount_offset_arcsec(
    pwi4: PWI4,
    max_arcsec: float = 10.0,
    settle_s: float = 1.0,
) -> tuple[float, float]:
    ra = random.uniform(-max_arcsec, max_arcsec)
    dec = random.uniform(-max_arcsec, max_arcsec)
    dither_mount_offset_arcsec(pwi4, ra_arcsec=ra, dec_arcsec=dec, settle_s=settle_s)
    return ra, dec
