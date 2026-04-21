from __future__ import annotations

from enum import IntEnum


class CameraState(IntEnum):
    Idle = 0
    Waiting = 1
    Exposing = 2
    Reading = 3
    Download = 4
    Error = 5
