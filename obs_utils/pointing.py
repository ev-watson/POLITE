from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

from alpyca_tools.fits_writer import FitsHeaderConfig

from .imaging import CaptureRequest, ImagingSession, capture_fits_file, select_filter
from .platesolve import PlateSolveConfig
from .pwi4_client import PWI4


ImageCaptureFn = Callable[[Path], None]


@dataclass
class ModelBuildConfig:
    image_arcsec_per_pixel: float
    num_alt: int = 3
    # PWI4 calls this field altitude, but POLITE uses 0 deg=zenith and
    # 90 deg=horizon. Keep model points inside the shed-safe 3--42 deg window.
    min_alt: float = 3.0
    max_alt: float = 42.0
    num_az: int = 6
    min_az: float = 5.0
    max_az: float = 355.0
    image_path: Path = Path("image.fits")
    poll_s: float = 0.2


def create_point_list(
    num_alt: int,
    min_alt: float,
    max_alt: float,
    num_az: int,
    min_az: float,
    max_az: float,
) -> list[tuple[float, float]]:
    points: list[tuple[float, float]] = []
    for i in range(num_az):
        azm = min_az + (max_az - min_az) * i / float(num_az)
        for j in range(num_alt):
            alt = min_alt + (max_alt - min_alt) * j / float(num_alt - 1)
            points.append((alt, azm))
    return points


def take_image_virtualcam(pwi4: PWI4, out_path: Path) -> None:
    pwi4.virtualcamera_take_image_and_save(str(out_path))


def take_image_pointing(
    session: ImagingSession,
    out_path: Path,
    filter_name: str = "Clear/Luminance",
) -> None:
    if session.filter_wheel is not None:
        select_filter(session, filter_name)

    request = CaptureRequest(
        exposure_s=2.0,
        is_light=True,
        binx=3,
        biny=3,
    )
    header = FitsHeaderConfig(imagetyp="LIGHT", object_name="POINTING", filter_name=filter_name)
    capture_fits_file(session, request, header, out_path)


def map_point(
    pwi4: PWI4,
    alt_deg: float,
    az_deg: float,
    take_image_fn: ImageCaptureFn,
    platesolve_cfg: PlateSolveConfig,
    arcsec_per_pixel: float,
    image_path: Path,
    poll_s: float = 0.2,
) -> None:
    raise RuntimeError(
        "Automatic pointing-model construction is disabled until a real POLITE "
        "PS3CLI result confirms its coordinate field names and units. Use the "
        "read-only obs_utils.platesolve.platesolve commissioning path first; "
        "this function will not slew, capture, or edit PWI4's model."
    )


def build_pointing_model(
    pwi4: PWI4,
    platesolve_cfg: PlateSolveConfig,
    config: ModelBuildConfig,
    session: Optional[ImagingSession] = None,
    take_image_fn: Optional[ImageCaptureFn] = None,
) -> None:
    raise RuntimeError(
        "Automatic pointing-model construction is disabled pending PlateSolve3 "
        "commissioning. This function will not create a model, slew, or capture."
    )
