from __future__ import annotations

import logging
from typing import Iterable, Optional

from .config import SlewLimits, SkyRegionLimit
from .pwi4_client import PWI4
from .waits import wait_until


logger = logging.getLogger(__name__)


# Deadlines for the PWI4 waits below. They are generous -- a healthy mount beats
# every one of them by a wide margin -- because their job is not to time the
# hardware but to make sure a stuck device eventually raises. Notebooks share
# one kernel with the camera, wheel, and HWP, so an unbounded poll here takes
# those down with it (see obs_utils/waits.py).
CONNECT_TIMEOUT_S = 30.0
ENABLE_TIMEOUT_S = 60.0
HOME_TIMEOUT_S = 300.0
SLEW_TIMEOUT_S = 300.0


def connect_mount(pwi4: PWI4, poll_s: float = 1.0, timeout_s: float = CONNECT_TIMEOUT_S) -> None:
    if pwi4.status().mount.is_connected:
        return
    logger.info("Connecting to mount")
    pwi4.mount_connect()
    wait_until(
        lambda: bool(pwi4.status().mount.is_connected),
        timeout_s=timeout_s,
        poll_s=poll_s,
        what="mount connect",
        detail="Check that the mount is powered and PWI4 shows it connected.",
        on_error="retry",
    )


def enable_motors(pwi4: PWI4, poll_s: float = 1.0, timeout_s: float = ENABLE_TIMEOUT_S) -> None:
    status = pwi4.status()
    if not status.mount.axis0.is_enabled:
        logger.info("Enabling axis 0")
        pwi4.mount_enable(0)
    if not status.mount.axis1.is_enabled:
        logger.info("Enabling axis 1")
        pwi4.mount_enable(1)

    def both_enabled() -> bool:
        st = pwi4.status()
        return bool(st.mount.axis0.is_enabled and st.mount.axis1.is_enabled)

    wait_until(
        both_enabled,
        timeout_s=timeout_s,
        poll_s=poll_s,
        what="mount axis enable",
        detail=(
            "One axis never energized. obs_utils.night_safety.verify_mount "
            "reports which one."
        ),
    )


def home_mount(
    pwi4: PWI4,
    poll_s: float = 1.0,
    settle_tol_deg: float = 0.001,
    timeout_s: float = HOME_TIMEOUT_S,
) -> None:
    """Home both axes, waiting until the reported positions stop changing.

    .. warning::
       The settle test below is **not** a proof that homing happened. It breaks
       when two consecutive polls report the same position, which a *dead* drive
       satisfies on the second poll -- so this can return "success" in ~1 s
       without the axis having moved at all. The timeout added here bounds the
       hang, not the false pass. Use
       :func:`obs_utils.night_safety.verify_mount`, which checks the axes are
       energized first, before trusting a home. Flagged 2026-07-30; fixing the
       settle logic is a separate decision.
    """
    logger.info("Finding home")
    pwi4.mount_find_home()
    last_axis0 = -99999.0
    last_axis1 = -99999.0

    def settled() -> bool:
        nonlocal last_axis0, last_axis1
        status = pwi4.status()
        delta0 = status.mount.axis0.position_degs - last_axis0
        delta1 = status.mount.axis1.position_degs - last_axis1
        if abs(delta0) < settle_tol_deg and abs(delta1) < settle_tol_deg:
            return True
        last_axis0 = status.mount.axis0.position_degs
        last_axis1 = status.mount.axis1.position_degs
        return False

    wait_until(
        settled,
        timeout_s=timeout_s,
        poll_s=poll_s,
        what="mount homing",
        detail="The axes never stopped moving. Stop the mount and check PWI4.",
    )


def wait_for_slew(pwi4: PWI4, poll_s: float = 0.2, timeout_s: float = SLEW_TIMEOUT_S) -> None:
    wait_until(
        lambda: not pwi4.status().mount.is_slewing,
        timeout_s=timeout_s,
        poll_s=poll_s,
        what="mount slew",
        detail="The mount is still slewing. Check for a rate limit or a blocked axis.",
    )


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
