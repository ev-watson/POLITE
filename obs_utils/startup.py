from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Optional

from .alpaca import ImagingSession, open_imaging_session
from .alpaca_servers import (
    AlpacaServerStatus,
    start_observatory_alpaca_servers,
    uses_local_observatory_alpaca_servers,
)
from .config import AlpacaConfig, Pwi4Config, SlewLimits, default_sky_regions
from .logging import LogPaths, LoggingConfig, setup_logging
from .mount import (
    apply_slew_rate_limit,
    connect_mount,
    enable_motors,
    home_mount,
    load_pointing_model,
    set_slew_time_constant,
)
from .pwi4_client import PWI4
from .timing import NtpStatus, TimingConfig, acquire_session_timing_snapshot


logger = logging.getLogger(__name__)


@dataclass
class StartupConfig:
    pwi4: Pwi4Config = field(default_factory=Pwi4Config)
    alpaca: AlpacaConfig = field(default_factory=AlpacaConfig)
    pointing_model_filename: Optional[str] = "DefaultModel.pxp"
    slew_limits: SlewLimits = field(default_factory=SlewLimits)
    slew_time_constant_s: Optional[float] = None
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    timing: TimingConfig = field(default_factory=TimingConfig)
    # In the observatory's Windows two-server layout, start ASCOM Remote and
    # the QHY SDK-direct Alpaca server before connecting any devices. Lab and
    # remote-host configurations are detected and left unchanged.
    launch_local_alpaca_servers: bool = True


@dataclass
class StartupState:
    pwi4: PWI4
    imaging: ImagingSession
    slew_limits: SlewLimits
    log_paths: Optional[LogPaths] = None
    ntp_status: Optional[NtpStatus] = None
    alpaca_servers: Optional[AlpacaServerStatus] = None


def _connect_rotator(pwi4: PWI4, poll_s: float = 0.5) -> None:
    status = pwi4.status()
    if not getattr(status.rotator, "exists", True):
        logger.info("Rotator not present; skipping")
        return

    if not status.rotator.is_connected:
        logger.info("Connecting rotator")
        pwi4.rotator_connect()

    status = pwi4.status()
    if not status.rotator.is_enabled:
        logger.info("Enabling rotator")
        pwi4.rotator_enable()

    while True:
        status = pwi4.status()
        if status.rotator.is_connected and status.rotator.is_enabled:
            break
        if not status.rotator.exists:
            break
        time.sleep(poll_s)


def _connect_focuser(pwi4: PWI4, poll_s: float = 0.5) -> None:
    status = pwi4.status()
    if not getattr(status.focuser, "exists", True):
        logger.info("Focuser not present; skipping")
        return

    if not status.focuser.is_connected:
        logger.info("Connecting focuser")
        pwi4.focuser_connect()

    status = pwi4.status()
    if not status.focuser.is_enabled:
        logger.info("Enabling focuser")
        pwi4.focuser_enable()

    while True:
        status = pwi4.status()
        if status.focuser.is_connected and status.focuser.is_enabled:
            break
        if not status.focuser.exists:
            break
        time.sleep(poll_s)


def startup_observatory(config: StartupConfig) -> StartupState:
    log_paths = None
    log_cfg = config.logging
    if log_cfg and log_cfg.enabled:
        log_paths = setup_logging(
            level=log_cfg.level,
            base_dir=log_cfg.base_dir,
            session_name=log_cfg.session_name,
            use_utc=log_cfg.use_utc,
            to_console=log_cfg.to_console,
            to_file=log_cfg.to_file,
            reset_handlers=log_cfg.reset_handlers,
        )
        logger.info("Logging to %s", log_paths.log_file)

    alpaca_servers = None
    if config.launch_local_alpaca_servers and uses_local_observatory_alpaca_servers(
        config.alpaca.host, config.alpaca.camera_host
    ):
        # The helper only confirms the two HTTP servers.  The usual imaging
        # session below remains responsible for connecting camera, EFW, and HWP.
        alpaca_servers = start_observatory_alpaca_servers(
            ascom_endpoint=config.alpaca.host,
            qhy_endpoint=config.alpaca.camera_host,
        )

    if not config.slew_limits.regions:
        config.slew_limits.regions = default_sky_regions()

    pwi4 = PWI4(host=config.pwi4.host, port=config.pwi4.port)

    connect_mount(pwi4)
    enable_motors(pwi4)
    home_mount(pwi4)

    _connect_rotator(pwi4)
    _connect_focuser(pwi4)

    if config.slew_time_constant_s is not None:
        set_slew_time_constant(pwi4, config.slew_time_constant_s)

    if config.slew_limits.max_slew_rate_deg_s is not None:
        apply_slew_rate_limit(
            pwi4,
            config.slew_limits.max_slew_rate_deg_s,
            enforce=config.slew_limits.enforce_rate,
        )

    if config.slew_limits.regions:
        for region in config.slew_limits.regions:
            logger.info(
                "Slew region enabled: %s alt=[%.1f, %.1f] az=[%.1f, %.1f]",
                region.name,
                region.alt_min_deg,
                region.alt_max_deg,
                region.az_min_deg,
                region.az_max_deg,
            )

    if config.pointing_model_filename:
        load_pointing_model(pwi4, config.pointing_model_filename)

    imaging = open_imaging_session(
        host=config.alpaca.host,
        camera_index=config.alpaca.camera_index,
        guide_camera_index=config.alpaca.guide_camera_index,
        filterwheel_index=config.alpaca.filterwheel_index,
        filter_names=config.alpaca.filter_names,
        rotator_index=config.alpaca.rotator_index,
        camera_host=config.alpaca.camera_host,
    )

    ntp_status = acquire_session_timing_snapshot(config.timing)
    if ntp_status.timesrc == "WIN-NTP":
        logger.info(
            "Timing snapshot: %s offset=%s sync_age=%s ref=%s",
            ntp_status.timesrc,
            ntp_status.ntpoffs_s,
            ntp_status.ntpage_s,
            ntp_status.timeref,
        )
    else:
        logger.info("Timing snapshot: %s (TIMEUNC=%.1f s)", ntp_status.timesrc, ntp_status.timeunc_s)

    logger.info("Startup complete")
    return StartupState(
        pwi4=pwi4,
        imaging=imaging,
        slew_limits=config.slew_limits,
        log_paths=log_paths,
        ntp_status=ntp_status,
        alpaca_servers=alpaca_servers,
    )
