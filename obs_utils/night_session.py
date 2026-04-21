from __future__ import annotations

import logging
import string
from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional, Union

from alpyca_tools.fits_writer import FitsHeaderConfig

from .imaging import CaptureRequest, capture_fits_file, select_filter
from .mount import slew_altaz, slew_radec_j2000, wait_for_slew
from .startup import StartupConfig, startup_observatory


logger = logging.getLogger(__name__)


@dataclass
class FramePlan:
    frame_type: str
    exposure_s: float
    count: int
    filter: Optional[Union[int, str]] = None
    object_name: Optional[str] = None
    binx: int = 1
    biny: int = 1
    startx: int = 0
    starty: int = 0
    numx: Optional[int] = None
    numy: Optional[int] = None
    gain: Optional[int] = None
    offset: Optional[int] = None
    readout_mode: Optional[int] = None
    sub_exposure_duration: Optional[float] = None
    fast_readout: Optional[bool] = None


@dataclass
class TargetPlan:
    name: str
    ra_hours: Optional[float] = None
    dec_deg: Optional[float] = None
    alt_deg: Optional[float] = None
    az_deg: Optional[float] = None
    frames: List[FramePlan] = field(default_factory=list)
    track: bool = True


@dataclass
class NightSessionConfig:
    startup: StartupConfig
    targets: List[TargetPlan] = field(default_factory=list)
    calibration_frames: List[FramePlan] = field(default_factory=list)
    calibration_stage: str = "after"
    base_data_dir: Union[str, Path] = "data"
    session_name: Optional[str] = None
    use_utc: bool = True
    observer: Optional[str] = None
    telescope: Optional[str] = None
    observatory: Optional[str] = None
    instrument: Optional[str] = None
    auto_metadata: bool = True
    stop_on_error: bool = True


def _slugify(text: str) -> str:
    allowed = set(string.ascii_letters + string.digits + "-_")
    cleaned = "".join(ch if ch in allowed else "_" for ch in text.strip())
    cleaned = cleaned.strip("_")
    return cleaned or "target"


def _frame_is_light(frame_type: str) -> bool:
    ft = frame_type.upper()
    if ft in ("DARK", "BIAS"):
        return False
    return True


def _build_filename(
    frame_type: str,
    exposure_s: float,
    index: int,
    object_name: Optional[str],
    filter_name: Optional[str],
) -> str:
    parts = [frame_type.upper()]
    if object_name:
        parts.append(_slugify(object_name))
    if filter_name:
        parts.append(f"f{_slugify(filter_name)}")
    parts.append(f"exp{exposure_s:g}s")
    parts.append(f"{index:03d}")
    return "_".join(parts) + ".fits"


def _safe_attr(obj: object, attr: str) -> Optional[str]:
    try:
        value = getattr(obj, attr)
    except Exception:
        return None
    if value is None:
        return None
    value = str(value).strip()
    return value or None


def _resolve_filter_name(
    session,
    filter_value: Optional[Union[int, str]],
    use_current: bool,
) -> Optional[str]:
    if filter_value is None:
        if use_current and session.filter_wheel is not None:
            try:
                pos = int(session.filter_wheel.Position)
                names = list(session.filter_wheel.Names)
                if not names and session.filter_names:
                    names = list(session.filter_names)
                if 0 <= pos < len(names):
                    return str(names[pos])
            except Exception:
                return None
        return None
    if isinstance(filter_value, str):
        return filter_value
    if session.filter_wheel is None:
        if session.filter_names and 0 <= int(filter_value) < len(session.filter_names):
            return str(session.filter_names[int(filter_value)])
        return str(filter_value)
    try:
        names = list(session.filter_wheel.Names)
        if not names and session.filter_names:
            names = list(session.filter_names)
        if 0 <= int(filter_value) < len(names):
            return str(names[int(filter_value)])
    except Exception:
        return str(filter_value)
    return str(filter_value)


def _format_hours(hours: Optional[float]) -> Optional[str]:
    if hours is None:
        return None
    return f"{hours:.6f}"


def _format_degs(deg: Optional[float]) -> Optional[str]:
    if deg is None:
        return None
    return f"{deg:.6f}"


def _auto_pointing_fields(pwi4) -> tuple[Optional[str], Optional[str], Optional[str]]:
    try:
        status = pwi4.status()
    except Exception:
        return None, None, None

    ra = _format_hours(status.mount.ra_j2000_hours)
    dec = _format_degs(status.mount.dec_j2000_degs)
    ha = None

    try:
        lmst = status.site.lmst_hours
        ra_app = status.mount.ra_apparent_hours
        if lmst is not None and ra_app is not None:
            ha_val = lmst - ra_app
            while ha_val < -12.0:
                ha_val += 24.0
            while ha_val >= 12.0:
                ha_val -= 24.0
            ha = _format_hours(ha_val)
    except Exception:
        pass

    return ra, dec, ha


def _build_header_config(
    session,
    pwi4,
    plan: FramePlan,
    target: Optional[TargetPlan],
    config: NightSessionConfig,
    filter_name: Optional[str],
) -> FitsHeaderConfig:
    object_name = plan.object_name or (target.name if target else None)
    if object_name is None:
        object_name = plan.frame_type.upper()

    instrument = config.instrument
    if config.auto_metadata:
        instrument = instrument or _safe_attr(session.camera, "SensorName")
        instrument = instrument or _safe_attr(session.camera, "Name")
        instrument = instrument or _safe_attr(session.camera, "DriverInfo")

    ra = None
    dec = None
    ha = None
    if config.auto_metadata and pwi4 is not None:
        ra, dec, ha = _auto_pointing_fields(pwi4)

    if ra is None and target and target.ra_hours is not None:
        ra = _format_hours(target.ra_hours)
    if dec is None and target and target.dec_deg is not None:
        dec = _format_degs(target.dec_deg)

    return FitsHeaderConfig(
        imagetyp=plan.frame_type.upper(),
        object_name=object_name,
        observer=config.observer,
        telescope=config.telescope,
        observatory=config.observatory,
        instrument=instrument,
        filter_name=filter_name,
        ra=ra,
        dec=dec,
        ha=ha,
    )


def _run_frames(
    session,
    pwi4,
    frames: List[FramePlan],
    base_dir: Path,
    target: Optional[TargetPlan],
    config: NightSessionConfig,
) -> None:
    for plan in frames:
        frame_type = plan.frame_type.upper()
        frame_dir = base_dir / frame_type
        frame_dir.mkdir(parents=True, exist_ok=True)

        filter_name = _resolve_filter_name(session, plan.filter, config.auto_metadata)
        if plan.filter is not None:
            logger.info("Selecting filter: %s", plan.filter)
            select_filter(session, plan.filter)

        for idx in range(1, plan.count + 1):
            request = CaptureRequest(
                exposure_s=plan.exposure_s,
                is_light=_frame_is_light(frame_type),
                binx=plan.binx,
                biny=plan.biny,
                startx=plan.startx,
                starty=plan.starty,
                numx=plan.numx,
                numy=plan.numy,
                gain=plan.gain,
                offset=plan.offset,
                readout_mode=plan.readout_mode,
                sub_exposure_duration=plan.sub_exposure_duration,
                fast_readout=plan.fast_readout,
            )

            header = _build_header_config(
                session=session,
                pwi4=pwi4,
                plan=plan,
                target=target,
                config=config,
                filter_name=filter_name,
            )

            filename = _build_filename(
                frame_type=frame_type,
                exposure_s=plan.exposure_s,
                index=idx,
                object_name=header.object_name,
                filter_name=filter_name,
            )
            out_path = frame_dir / filename

            logger.info(
                "Capturing %s %d/%d exp=%.3fs -> %s",
                frame_type,
                idx,
                plan.count,
                plan.exposure_s,
                out_path,
            )
            try:
                capture_fits_file(session, request, header, out_path)
                logger.info("Saved %s", out_path)
            except Exception:
                logger.exception("Capture failed for %s", out_path)
                if config.stop_on_error:
                    raise


def _slew_to_target(pwi4, target: TargetPlan, limits) -> None:
    if (target.alt_deg is None) ^ (target.az_deg is None):
        raise ValueError(f"Target {target.name} must define both alt_deg and az_deg")
    if (target.ra_hours is None) ^ (target.dec_deg is None):
        raise ValueError(f"Target {target.name} must define both ra_hours and dec_deg")

    if target.alt_deg is not None and target.az_deg is not None:
        slew_altaz(pwi4, target.alt_deg, target.az_deg, limits=limits)
    elif target.ra_hours is not None and target.dec_deg is not None:
        slew_radec_j2000(pwi4, target.ra_hours, target.dec_deg, limits=limits)
    else:
        return
    wait_for_slew(pwi4)
    if target.track:
        pwi4.mount_tracking_on()


def _should_run_calibrations(stage: str, config_stage: str) -> bool:
    if config_stage == "both":
        return True
    return config_stage == stage


def run_night_session(config: NightSessionConfig) -> None:
    if config.calibration_stage not in ("before", "after", "both"):
        raise ValueError("calibration_stage must be 'before', 'after', or 'both'")

    log_cfg = config.startup.logging
    if config.session_name and log_cfg.session_name is None:
        log_cfg = replace(log_cfg, session_name=config.session_name)
    if log_cfg.use_utc != config.use_utc:
        log_cfg = replace(log_cfg, use_utc=config.use_utc)
    startup_cfg = replace(config.startup, logging=log_cfg)

    state = startup_observatory(startup_cfg)

    if state.log_paths is not None:
        date_dir = state.log_paths.date_dir.name
        session_id = state.log_paths.log_file.stem
    else:
        now = datetime.now(timezone.utc) if config.use_utc else datetime.now()
        date_dir = now.strftime("%Y-%m-%d")
        session_id = now.strftime("session_%Y%m%d_%H%M%S")

    base_dir = Path(config.base_data_dir).expanduser().resolve()
    session_dir = base_dir / date_dir / session_id
    session_dir.mkdir(parents=True, exist_ok=True)
    logger.info("Session data directory: %s", session_dir)

    if config.calibration_frames and _should_run_calibrations("before", config.calibration_stage):
        cal_dir = session_dir / "calibrations"
        logger.info("Running calibration frames (before targets)")
        _run_frames(state.imaging, state.pwi4, config.calibration_frames, cal_dir, None, config)

    for target in config.targets:
        logger.info("Target: %s", target.name)
        _slew_to_target(state.pwi4, target, limits=state.slew_limits)
        target_dir = session_dir / "targets" / _slugify(target.name)
        if not target.frames:
            logger.warning("No frames defined for target %s", target.name)
        _run_frames(state.imaging, state.pwi4, target.frames, target_dir, target, config)

    if config.calibration_frames and _should_run_calibrations("after", config.calibration_stage):
        cal_dir = session_dir / "calibrations"
        logger.info("Running calibration frames (after targets)")
        _run_frames(state.imaging, state.pwi4, config.calibration_frames, cal_dir, None, config)

    logger.info("Night session complete")
    state.imaging.close()
