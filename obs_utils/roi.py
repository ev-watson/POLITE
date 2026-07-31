"""Shared camera-ROI policy and fail-closed application helpers.

The Alpaca server always retains the camera's full raw frame as its default.
This module is the session-side policy layer: a night may choose a smaller
science ROI only after it has been measured and explicitly validated.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Mapping, Optional


@dataclass(frozen=True)
class EffectiveArea:
    """SDK-reported usable detector rectangle in unbinned sensor pixels."""

    startx: int
    starty: int
    numx: int
    numy: int

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "EffectiveArea":
        return cls(**{key: int(value[key]) for key in ("startx", "starty", "numx", "numy")})

    @property
    def endx(self) -> int:
        return self.startx + self.numx

    @property
    def endy(self) -> int:
        return self.starty + self.numy


@dataclass(frozen=True)
class CameraROI:
    """One ROI in the driver's binned-coordinate convention."""

    startx: int = 0
    starty: int = 0
    numx: Optional[int] = None
    numy: Optional[int] = None
    binx: int = 1
    biny: int = 1

    def resolved(self, camera_x_size: int, camera_y_size: int) -> "CameraROI":
        if self.binx < 1 or self.biny < 1:
            raise ValueError("ROI binning must be positive")
        return CameraROI(
            startx=int(self.startx), starty=int(self.starty),
            numx=(camera_x_size // self.binx if self.numx is None else int(self.numx)),
            numy=(camera_y_size // self.biny if self.numy is None else int(self.numy)),
            binx=int(self.binx), biny=int(self.biny),
        )

    def validate(
        self,
        camera_x_size: int,
        camera_y_size: int,
        *,
        effective_area: Optional[EffectiveArea] = None,
    ) -> "CameraROI":
        """Resolve and reject an ROI outside raw or effective detector bounds."""
        roi = self.resolved(camera_x_size, camera_y_size)
        assert roi.numx is not None and roi.numy is not None
        maxx, maxy = camera_x_size // roi.binx, camera_y_size // roi.biny
        if roi.startx < 0 or roi.starty < 0 or roi.numx < 1 or roi.numy < 1:
            raise ValueError(f"Invalid ROI: {roi}")
        if roi.startx + roi.numx > maxx or roi.starty + roi.numy > maxy:
            raise ValueError(f"ROI {roi} exceeds binned camera bounds {maxx}x{maxy}")
        if effective_area is not None:
            x0, x1 = roi.startx * roi.binx, (roi.startx + roi.numx) * roi.binx
            y0, y1 = roi.starty * roi.biny, (roi.starty + roi.numy) * roi.biny
            if x0 < effective_area.startx or x1 > effective_area.endx or y0 < effective_area.starty or y1 > effective_area.endy:
                raise ValueError(f"ROI {roi} falls outside effective area {effective_area}")
        return roi


def camera_effective_area(camera: Any) -> EffectiveArea:
    """Read the QHY SDK rectangle through the server's standard Alpaca Action."""
    raw = camera.Action("POLITE.EffectiveArea", "")
    if isinstance(raw, str):
        raw = json.loads(raw)
    if not isinstance(raw, Mapping):
        raise RuntimeError(f"Invalid POLITE.EffectiveArea response: {raw!r}")
    return EffectiveArea.from_mapping(raw)


def camera_roi(camera: Any) -> CameraROI:
    """Read the current driver ROI without assuming a full frame."""
    return CameraROI(
        startx=int(camera.StartX), starty=int(camera.StartY),
        numx=int(camera.NumX), numy=int(camera.NumY),
        binx=int(camera.BinX), biny=int(camera.BinY),
    )


def apply_roi(
    camera: Any,
    roi: CameraROI,
    *,
    effective_area: Optional[EffectiveArea] = None,
) -> CameraROI:
    """Apply a measured ROI and require exact camera read-back.

    The QHY server clamps invalid requests.  Exact verification converts that
    otherwise silent behavior into a visible, fail-closed acquisition error.
    """
    requested = roi.validate(
        int(camera.CameraXSize), int(camera.CameraYSize), effective_area=effective_area,
    )
    camera.BinX, camera.BinY = requested.binx, requested.biny
    camera.StartX, camera.StartY = requested.startx, requested.starty
    camera.NumX, camera.NumY = requested.numx, requested.numy
    actual = camera_roi(camera)
    if actual != requested:
        raise RuntimeError(f"Camera ROI read-back {actual} does not match requested {requested}")
    return actual
