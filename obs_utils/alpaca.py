from __future__ import annotations

import time
from dataclasses import dataclass
from typing import List, Optional, Union

from alpaca.filterwheel import FilterWheel
from alpaca.rotator import Rotator

from alpyca_tools.camera_device import CameraDevice


@dataclass
class ImagingSession:
    camera: CameraDevice
    guide_camera: Optional[CameraDevice] = None
    filter_wheel: Optional[FilterWheel] = None
    filter_names: Optional[List[str]] = None
    # Optec Pyxis 2" carrying the half-wave plate, bridged via INDIGO's
    # indigo_agent_alpaca as an Alpaca Rotator. None when no HWP stage is wired
    # (the PWI4 field rotator is handled separately, not here).
    rotator: Optional[Rotator] = None

    def close(self) -> None:
        for dev in (self.rotator, self.filter_wheel, self.guide_camera, self.camera):
            if dev is None:
                continue
            try:
                dev.Connected = False
            except Exception:
                pass


def connect_camera(host: str, device_number: int) -> CameraDevice:
    cam = CameraDevice(host, device_number)
    cam.Connected = True
    return cam


def connect_filter_wheel(host: str, device_number: int) -> FilterWheel:
    fw = FilterWheel(host, device_number)
    fw.Connected = True
    return fw


def connect_rotator(host: str, device_number: int) -> Rotator:
    """Connect an ASCOM Alpaca Rotator.

    Used for the Optec Pyxis when it is bridged through INDIGO's
    ``indigo_agent_alpaca`` (INDIGO Rotator -> Alpaca IRotatorV3).
    """
    rot = Rotator(host, device_number)
    rot.Connected = True
    return rot


def move_rotator_absolute(
    rotator: Rotator,
    position_deg: float,
    poll_s: float = 0.5,
    timeout_s: float = 120.0,
) -> float:
    """Move the rotator to an absolute sky position angle and block until settled.

    ``MoveAbsolute`` is asynchronous per the Alpaca spec; we poll ``IsMoving``.
    Returns the final reported ``Position`` in degrees.
    """
    rotator.MoveAbsolute(float(position_deg))
    t0 = time.time()
    while rotator.IsMoving:
        if (time.time() - t0) > timeout_s:
            raise TimeoutError("Rotator move timed out")
        time.sleep(poll_s)
    return float(rotator.Position)


def open_imaging_session(
    host: str,
    camera_index: int,
    guide_camera_index: Optional[int] = None,
    filterwheel_index: Optional[int] = None,
    filter_names: Optional[List[str]] = None,
    rotator_index: Optional[int] = None,
) -> ImagingSession:
    camera = connect_camera(host, camera_index)
    guide_camera = None
    filter_wheel = None
    rotator = None

    if guide_camera_index is not None:
        guide_camera = connect_camera(host, guide_camera_index)

    if filterwheel_index is not None:
        filter_wheel = connect_filter_wheel(host, filterwheel_index)

    if rotator_index is not None:
        rotator = connect_rotator(host, rotator_index)

    return ImagingSession(
        camera=camera,
        guide_camera=guide_camera,
        filter_wheel=filter_wheel,
        filter_names=filter_names,
        rotator=rotator,
    )


def set_filter_position(
    filter_wheel: FilterWheel,
    position: Union[int, str],
    names_override: Optional[List[str]] = None,
    poll_s: float = 0.5,
    timeout_s: float = 30.0,
) -> int:
    if isinstance(position, str):
        names = list(filter_wheel.Names)
        if position in names:
            position = names.index(position)
        elif names_override and position in names_override:
            position = names_override.index(position)
        else:
            raise ValueError(
                f"Filter '{position}' not in filter wheel names: {names or names_override}"
            )

    filter_wheel.Position = int(position)
    t0 = time.time()
    while True:
        if filter_wheel.Position == int(position):
            return int(position)
        if (time.time() - t0) > timeout_s:
            raise TimeoutError("Filter wheel move timed out")
        time.sleep(poll_s)
