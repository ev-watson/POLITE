from __future__ import annotations

import logging
import time
from typing import Iterable, Optional

from .config import SlewLimits, SkyRegionLimit
from .pwi4_client import PWI4


logger = logging.getLogger(__name__)


def connect_mount(pwi4: PWI4, poll_s: float = 1.0) -> None:
    if pwi4.status().mount.is_connected:
        return
    logger.info("Connecting to mount")
    pwi4.mount_connect()
    while not pwi4.status().mount.is_connected:
        time.sleep(poll_s)


def enable_motors(pwi4: PWI4, poll_s: float = 1.0) -> None:
    status = pwi4.status()
    if not status.mount.axis0.is_enabled:
        logger.info("Enabling axis 0")
        pwi4.mount_enable(0)
    if not status.mount.axis1.is_enabled:
        logger.info("Enabling axis 1")
        pwi4.mount_enable(1)

    while True:
        status = pwi4.status()
        if status.mount.axis0.is_enabled and status.mount.axis1.is_enabled:
            break
        time.sleep(poll_s)


def home_mount(pwi4: PWI4, poll_s: float = 1.0, settle_tol_deg: float = 0.001) -> None:
    logger.info("Finding home")
    pwi4.mount_find_home()
    last_axis0 = -99999.0
    last_axis1 = -99999.0
    while True:
        status = pwi4.status()
        delta0 = status.mount.axis0.position_degs - last_axis0
        delta1 = status.mount.axis1.position_degs - last_axis1
        if abs(delta0) < settle_tol_deg and abs(delta1) < settle_tol_deg:
            break
        last_axis0 = status.mount.axis0.position_degs
        last_axis1 = status.mount.axis1.position_degs
        time.sleep(poll_s)


def wait_for_slew(pwi4: PWI4, poll_s: float = 0.2) -> None:
    while pwi4.status().mount.is_slewing:
        time.sleep(poll_s)


def load_pointing_model(pwi4: PWI4, filename: str) -> None:
    logger.info("Loading pointing model: %s", filename)
    pwi4.mount_model_load(filename)


def set_slew_time_constant(pwi4: PWI4, value_s: float) -> None:
    logger.info("Setting slew time constant: %.3f s", value_s)
    pwi4.mount_set_slew_time_constant(value_s)


def apply_slew_rate_limit(pwi4: PWI4, max_deg_s: float, enforce: bool = False) -> None:
    status = pwi4.status()
    axis0 = status.mount.axis0.max_velocity_degs_per_sec
    axis1 = status.mount.axis1.max_velocity_degs_per_sec

    if axis0 is None or axis1 is None:
        logger.warning("Mount max velocity not reported; cannot verify slew rate limits")
        return

    if axis0 > max_deg_s or axis1 > max_deg_s:
        msg = (
            f"Mount max velocity exceeds limit: axis0={axis0:.2f} deg/s, "
            f"axis1={axis1:.2f} deg/s, limit={max_deg_s:.2f} deg/s"
        )
        if enforce:
            raise RuntimeError(msg)
        logger.warning(msg)


def _altaz_allowed(alt_deg: float, az_deg: float, regions: Iterable[SkyRegionLimit]) -> bool:
    for region in regions:
        if (
            region.alt_min_deg <= alt_deg <= region.alt_max_deg
            and region.az_min_deg <= az_deg <= region.az_max_deg
        ):
            return True
    return False


def slew_altaz(
    pwi4: PWI4,
    alt_deg: float,
    az_deg: float,
    limits: Optional[SlewLimits] = None,
) -> None:
    if limits and limits.enforce_regions and limits.regions:
        if not _altaz_allowed(alt_deg, az_deg, limits.regions):
            raise ValueError(f"Target Alt/Az {alt_deg:.2f}, {az_deg:.2f} outside allowed regions")
    pwi4.mount_goto_alt_az(alt_deg, az_deg)


def slew_radec_j2000(
    pwi4: PWI4,
    ra_hours: float,
    dec_deg: float,
    limits: Optional[SlewLimits] = None,
) -> None:
    if limits and limits.enforce_regions and limits.regions:
        logger.warning("Sky region limits are defined in Alt/Az; RA/Dec checks are not implemented")
    pwi4.mount_goto_ra_dec_j2000(ra_hours, dec_deg)
