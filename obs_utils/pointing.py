from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

from alpyca_tools.fits_writer import FitsHeaderConfig

from .imaging import CaptureRequest, ImagingSession, capture_fits_file, select_filter
from .platesolve import PlateSolveConfig, platesolve
from .pwi4_client import PWI4


ImageCaptureFn = Callable[[Path], None]


@dataclass
class ModelBuildConfig:
    image_arcsec_per_pixel: float
    num_alt: int = 3
    min_alt: float = 20.0
    max_alt: float = 80.0
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
    pwi4.mount_goto_alt_az(alt_deg, az_deg)
    while pwi4.status().mount.is_slewing:
        time.sleep(poll_s)

    status = pwi4.status()
    azm_error = abs(status.mount.azimuth_degs - az_deg)
    alt_error = abs(status.mount.altitude_degs - alt_deg)
    if azm_error > 0.1 or alt_error > 0.1:
        raise RuntimeError(
            "Mount stopped too far from target: "
            f"az={status.mount.azimuth_degs:.4f}, alt={status.mount.altitude_degs:.4f}"
        )

    pwi4.mount_tracking_on()
    take_image_fn(image_path)

    result = platesolve(
        image_file=image_path,
        arcsec_per_pixel=arcsec_per_pixel,
        config=platesolve_cfg,
    )

    pwi4.mount_model_add_point(result["ra_j2000_hours"], result["dec_j2000_degrees"])


def build_pointing_model(
    pwi4: PWI4,
    platesolve_cfg: PlateSolveConfig,
    config: ModelBuildConfig,
    session: Optional[ImagingSession] = None,
    take_image_fn: Optional[ImageCaptureFn] = None,
) -> None:
    config.image_path.parent.mkdir(parents=True, exist_ok=True)
    points = create_point_list(
        config.num_alt,
        config.min_alt,
        config.max_alt,
        config.num_az,
        config.min_az,
        config.max_az,
    )

    if take_image_fn is None:
        if session is None:
            raise ValueError("session is required when take_image_fn is not provided")
        take_image_fn = lambda path: take_image_pointing(session, path)

    for alt, azm in points:
        map_point(
            pwi4,
            alt,
            azm,
            take_image_fn,
            platesolve_cfg,
            arcsec_per_pixel=config.image_arcsec_per_pixel,
            image_path=config.image_path,
            poll_s=config.poll_s,
        )
