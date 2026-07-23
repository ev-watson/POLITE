from __future__ import annotations

import logging
import math
import string
import time
from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Union

from alpyca_tools.camera_ops import ExposureSettings, configure_camera
from alpyca_tools.fits_writer import DetectorCards, FitsHeaderConfig, PolarimetryCards

from .block_log import (
    JULIAN_ELEV_M,
    JULIAN_LAT_DEG,
    JULIAN_LON_DEG,
    BlockLogger,
    moon_separation_deg,
    pwi4_snapshot,
)
from .horizons import cache_ephemeris, fetch_ephemeris, resolve_target_id
from .imaging import CaptureRequest, capture_fits_file, select_filter, select_hwp_angle
from .mount import slew_altaz, slew_radec_j2000, wait_for_slew
from .night_display import NightReporter, estimated_duration_s, total_frame_count
from .pol_config import polconfig_snapshot, write_pol_config_sidecar
from .qa_gates import QAGate, dispatch_qa_gate, gates_after_cal, gates_after_target
from .session_context import SessionCaptureContext
from .startup import StartupConfig, startup_observatory
from .timing import FilterWheelState, NtpStatus


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
    # Polarimetry: when hwp_angle_deg is set, the Pyxis half-wave plate is moved
    # to this angle before the frame(s) and HWPANG/POLSEQ cards are written.
    hwp_angle_deg: Optional[float] = None
    retardance_deg: Optional[float] = None
    pol_seq_id: Optional[str] = None
    pol_seq_index: Optional[int] = None
    source_brick: Optional[str] = None


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
    capture_context: Optional[SessionCaptureContext] = None
    naming: Literal["legacy", "polite"] = "polite"
    calibration_before: List[FramePlan] = field(default_factory=list)
    calibration_after: List[FramePlan] = field(default_factory=list)
    qa_gates: List[QAGate] = field(default_factory=list)
    catalog: Dict[str, Dict[str, Any]] = field(default_factory=dict)

    @property
    def calibration_frames(self) -> List[FramePlan]:
        return list(self.calibration_before) + list(self.calibration_after)


_FILTER_SLUGS = {
    "Photometric V": "V",
    "Photometric R": "R",
    "Photometric B": "B",
    "Clear": "Clear",
    "Dark": "Dark",
}


def _pwi4_instrot_deg(pwi4) -> Optional[float]:
    """Read PWI4 field-rotator angle [deg] for INSTROT."""
    if pwi4 is None:
        return None
    try:
        status = pwi4.status()
        if hasattr(status, "rotator") and status.rotator.exists:
            return float(status.rotator.field_angle_degs)
    except Exception:
        logger.debug("Could not read PWI4 field rotator angle", exc_info=True)
    return None


def _filter_slug(filter_name: Optional[str], frame_type: str) -> str:
    if frame_type.upper() in ("DARK", "BIAS"):
        return frame_type.capitalize()
    if filter_name is None:
        return "None"
    return _FILTER_SLUGS.get(filter_name, _slugify(filter_name))


def _build_polite_filename(
    date_str: str,
    object_name: Optional[str],
    filter_name: Optional[str],
    frame_type: str,
    exposure_s: float,
    index: int,
) -> str:
    target = _slugify(object_name) if object_name else frame_type.upper()
    filt = _filter_slug(filter_name, frame_type)
    return f"{date_str}_{target}_{filt}_{exposure_s:g}s_{index:03d}.fits"


def _detector_cards(ctx: Optional[SessionCaptureContext], plan: FramePlan) -> Optional[DetectorCards]:
    if ctx is None:
        return None
    gain = plan.gain if plan.gain is not None else ctx.gain_setting
    offset = plan.offset if plan.offset is not None else ctx.offset_setting
    mode = plan.readout_mode if plan.readout_mode is not None else ctx.readout_mode
    return DetectorCards(
        gain_setting=gain,
        readout_mode=mode,
        readout_mode_name=ctx.readout_mode_name,
        offset_setting=offset,
        cooler_setpoint_c=ctx.cooler_setpoint_c,
        pixel_size_um=ctx.pixel_size_um,
    )


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


@dataclass
class _PointingFields:
    """Mount pointing snapshot in FITS-ready forms (sexagesimal + numeric)."""

    ra_sex: Optional[str] = None
    dec_sex: Optional[str] = None
    ha_sex: Optional[str] = None
    lst_sex: Optional[str] = None
    ra_deg: Optional[float] = None
    dec_deg: Optional[float] = None
    alt_deg: Optional[float] = None
    az_deg: Optional[float] = None
    airmass: Optional[float] = None


def _auto_pointing_fields(pwi4) -> _PointingFields:
    """Snapshot mount pointing into sexagesimal + numeric FITS pointing fields."""
    from .obs_math import airmass_kasten_young, deg_to_dms, hours_to_hms

    fields = _PointingFields()
    try:
        status = pwi4.status()
    except Exception:
        return fields

    mount = status.mount
    ra_h = getattr(mount, "ra_j2000_hours", None)
    dec_d = getattr(mount, "dec_j2000_degs", None)
    if ra_h is not None:
        fields.ra_deg = float(ra_h) * 15.0
        fields.ra_sex = hours_to_hms(ra_h, wrap=True)
    if dec_d is not None:
        fields.dec_deg = float(dec_d)
        fields.dec_sex = deg_to_dms(dec_d)

    fields.alt_deg = getattr(mount, "altitude_degs", None)
    fields.az_deg = getattr(mount, "azimuth_degs", None)
    fields.airmass = airmass_kasten_young(fields.alt_deg)

    try:
        lmst = status.site.lmst_hours
        if lmst is not None:
            fields.lst_sex = hours_to_hms(lmst, wrap=True)
            ra_app = getattr(mount, "ra_apparent_hours", None)
            if ra_app is not None:
                ha_val = lmst - ra_app
                while ha_val < -12.0:
                    ha_val += 24.0
                while ha_val >= 12.0:
                    ha_val -= 24.0
                fields.ha_sex = hours_to_hms(ha_val, signed=True, wrap=False)
    except Exception:
        pass

    return fields


def _build_header_config(
    session,
    pwi4,
    plan: FramePlan,
    target: Optional[TargetPlan],
    config: NightSessionConfig,
    filter_name: Optional[str],
    ntp_status: Optional[NtpStatus] = None,
    filter_wheel_state: Optional[FilterWheelState] = None,
    achieved_hwp_deg: Optional[float] = None,
    instrot_deg: Optional[float] = None,
) -> FitsHeaderConfig:
    object_name = plan.object_name or (target.name if target else None)
    if object_name is None:
        object_name = plan.frame_type.upper()

    ctx = config.capture_context
    instrument = config.instrument
    if config.auto_metadata:
        instrument = instrument or _safe_attr(session.camera, "SensorName")
        instrument = instrument or _safe_attr(session.camera, "Name")
        instrument = instrument or _safe_attr(session.camera, "DriverInfo")

    observer = config.observer or (ctx.observer if ctx else None)
    telescope = config.telescope or (ctx.telescope if ctx else None)
    observatory = config.observatory or (ctx.observatory if ctx else None)

    pf = _PointingFields()
    if config.auto_metadata and pwi4 is not None:
        pf = _auto_pointing_fields(pwi4)

    if pf.ra_sex is None and target and target.ra_hours is not None:
        from .obs_math import hours_to_hms

        pf.ra_deg = float(target.ra_hours) * 15.0
        pf.ra_sex = hours_to_hms(target.ra_hours, wrap=True)
    if pf.dec_sex is None and target and target.dec_deg is not None:
        from .obs_math import deg_to_dms

        pf.dec_deg = float(target.dec_deg)
        pf.dec_sex = deg_to_dms(target.dec_deg)

    polarimetry = None
    hwp = achieved_hwp_deg if achieved_hwp_deg is not None else plan.hwp_angle_deg
    if hwp is not None or plan.pol_seq_id is not None:
        wavlen = ctx.filter_wavelength_nm(filter_name) if ctx else None
        polarimetry = PolarimetryCards(
            hwp_angle_deg=hwp,
            retardance_deg=plan.retardance_deg if plan.retardance_deg is not None else 180.0,
            instrument_rotator_deg=instrot_deg,
            pol_seq_id=plan.pol_seq_id,
            pol_seq_index=plan.pol_seq_index,
            eff_wavelength_nm=wavlen,
            hwp_uncert_deg=ctx.hwp_uncert_deg if ctx else None,
        )

    return FitsHeaderConfig(
        imagetyp=plan.frame_type.upper(),
        object_name=object_name,
        observer=observer,
        telescope=telescope,
        observatory=observatory,
        instrument=instrument,
        origin=ctx.origin if ctx else None,
        filter_name=filter_name,
        ra=pf.ra_sex,
        dec=pf.dec_sex,
        ha=pf.ha_sex,
        ra_deg=pf.ra_deg,
        dec_deg=pf.dec_deg,
        alt_deg=pf.alt_deg,
        az_deg=pf.az_deg,
        airmass=pf.airmass,
        lst=pf.lst_sex,
        radesys="FK5" if pf.ra_sex is not None else None,
        site_lat_deg=ctx.site_lat_deg if ctx else None,
        site_lon_deg=ctx.site_lon_deg if ctx else None,
        site_elev_m=ctx.site_elev_m if ctx else None,
        focal_len_mm=ctx.focal_length_mm if ctx else None,
        focal_ratio=ctx.focal_ratio if ctx else None,
        aperture_mm=ctx.aperture_mm if ctx else None,
        pixscale=ctx.plate_scale_arcsec if ctx else None,
        polarimetry=polarimetry,
        detector=_detector_cards(ctx, plan),
        timing=ntp_status.to_provenance() if ntp_status is not None else None,
        filter_wheel_state=filter_wheel_state,
    )


def _apply_session_camera(session, ctx: SessionCaptureContext) -> SessionCaptureContext:
    """Apply session settings and return the context with its effective setpoint.

    ``exact`` is for a controlled temperature experiment: the requested target
    is always sent. ``at_or_below`` is for normal science: if the connected
    detector is already colder than the requested target, preserve that lower
    temperature rather than warming it toward the nominal setpoint.
    """
    camera = session.camera
    settings = ExposureSettings(
        exposure_s=0.0,
        is_light=False,
        gain=ctx.gain_setting,
        offset=ctx.offset_setting,
        readout_mode=ctx.readout_mode,
    )
    configure_camera(camera, settings)
    try:
        if getattr(camera, "CanSetCCDTemperature", False):
            requested_c = float(ctx.cooler_setpoint_c)
            effective_c = requested_c
            if ctx.cooler_policy == "at_or_below":
                try:
                    current_c = float(camera.CCDTemperature)
                except Exception:
                    current_c = float("nan")
                if math.isfinite(current_c) and current_c < requested_c:
                    effective_c = current_c
                    logger.info(
                        "Cooler at %+.2f C, colder than requested %+.1f C; "
                        "preserving the lower temperature.",
                        current_c, requested_c,
                    )
            camera.SetCCDTemperature = effective_c
            if effective_c == requested_c:
                logger.info("Cooler setpoint -> %+.1f C (%s)", effective_c, ctx.cooler_policy)
            else:
                logger.info(
                    "Cooler setpoint -> %+.2f C (at_or_below; requested %+.1f C)",
                    effective_c, requested_c,
                )
            ctx = replace(ctx, cooler_setpoint_c=effective_c)
    except Exception:
        logger.warning("Could not set cooler setpoint", exc_info=True)
    return ctx


def _read_camera(camera, attr: str):
    try:
        return getattr(camera, attr)
    except Exception:
        return None


def _verify_session_hardware(imaging, ctx: SessionCaptureContext, expected_filter_names) -> None:
    """Fail closed when the commanded camera/EFW setup differs from the plan."""
    camera = imaging.camera
    mismatches = []
    for label, commanded, actual in (
        ("gain", ctx.gain_setting, _read_camera(camera, "Gain")),
        ("offset", ctx.offset_setting, _read_camera(camera, "Offset")),
        ("readout_mode", ctx.readout_mode, _read_camera(camera, "ReadoutMode")),
    ):
        if actual is not None and int(actual) != int(commanded):
            mismatches.append(f"{label}: commanded {commanded}, camera reports {actual}")
    if mismatches:
        raise RuntimeError("Camera settings mismatch: " + "; ".join(mismatches))

    wheel = imaging.filter_wheel
    expected = [str(name) for name in (expected_filter_names or [])]
    if wheel is not None and expected:
        try:
            live = [str(name) for name in wheel.Names]
        except Exception:
            live = []
        if live and live != expected:
            raise RuntimeError(
                "EFW driver Names do not match the configured physical carousel: "
                f"driver={live}, expected={expected}"
            )
        logger.info("EFW names verified: %s", live or expected)

    logger.info(
        "Camera settings verified: gain=%s offset=%s readout=%s cooler=%+.2f C (%s)",
        ctx.gain_setting, ctx.offset_setting, ctx.readout_mode,
        ctx.cooler_setpoint_c, ctx.cooler_policy,
    )


def wait_for_cooler(
    camera,
    setpoint_c: float,
    *,
    tolerance_c: float = 0.1,
    stable_s: float = 30.0,
    timeout_s: float = 600.0,
    poll_s: float = 15.0,
) -> float:
    """Observe the QHY cooler until it is stably at the commanded setpoint.

    The setpoint has already been issued once. This gate reads telemetry only,
    leaving QHY's internal cooler PID in control. A normal science run aborts
    on timeout rather than silently collecting data at an unknown temperature.
    """
    if not _read_camera(camera, "CanSetCCDTemperature"):
        logger.warning("Camera cannot set temperature; skipping cooler gate.")
        return float("nan")

    logger.info(
        "Waiting for %+.2f C (tol %.2f C, hold %.0f s, timeout %.0f s)",
        setpoint_c, tolerance_c, stable_s, timeout_s,
    )
    started = time.monotonic()
    stable_since = None
    while True:
        value = _read_camera(camera, "CCDTemperature")
        temp_c = float(value) if value is not None else float("nan")
        power = _read_camera(camera, "CoolerPower")
        elapsed = time.monotonic() - started
        logger.info(
            "Cooler T=%+6.2f C target=%+.2f C power=%s%% elapsed=%4.0f s",
            temp_c, setpoint_c, "n/a" if power is None else f"{float(power):.0f}", elapsed,
        )
        if math.isfinite(temp_c) and abs(temp_c - setpoint_c) <= tolerance_c:
            if stable_since is None:
                stable_since = time.monotonic()
            elif time.monotonic() - stable_since >= stable_s:
                logger.info("Cooler stable at %+.2f C", temp_c)
                return temp_c
        else:
            stable_since = None
        if elapsed >= timeout_s:
            raise TimeoutError(
                f"Cooler did not stabilize at {setpoint_c:+.2f} C within {timeout_s:.0f} s "
                f"(last temperature {temp_c:+.2f} C)"
            )
        time.sleep(poll_s)


def plan_total_frames(config: "NightSessionConfig") -> int:
    """Total individual exposures the night will capture, respecting cal stage."""
    stage = config.calibration_stage
    total = sum(total_frame_count(t.frames) for t in config.targets)
    if config.calibration_before and stage in ("before", "both"):
        total += total_frame_count(config.calibration_before)
    if config.calibration_after and stage in ("after", "both"):
        total += total_frame_count(config.calibration_after)
    return total


def plan_estimated_duration_s(config: "NightSessionConfig") -> float:
    """Exposure-aware duration estimate for the live night-progress display."""
    stage = config.calibration_stage
    frames = []
    for target in config.targets:
        frames.extend(target.frames)
    if config.calibration_before and stage in ("before", "both"):
        frames.extend(config.calibration_before)
    if config.calibration_after and stage in ("after", "both"):
        frames.extend(config.calibration_after)
    return estimated_duration_s(frames)


def _run_frames(
    session,
    pwi4,
    frames: List[FramePlan],
    base_dir: Path,
    target: Optional[TargetPlan],
    config: NightSessionConfig,
    ntp_status: Optional[NtpStatus] = None,
    *,
    date_str: Optional[str] = None,
    block_log: Optional[list] = None,
    block_id: Optional[str] = None,
    reporter: Optional[NightReporter] = None,
) -> None:
    ctx = config.capture_context
    for plan in frames:
        frame_type = plan.frame_type.upper()
        if config.naming == "polite":
            frame_dir = base_dir
        else:
            frame_dir = base_dir / frame_type
        frame_dir.mkdir(parents=True, exist_ok=True)

        filter_name = _resolve_filter_name(session, plan.filter, config.auto_metadata)
        filter_wheel_state: Optional[FilterWheelState] = None
        if plan.filter is not None:
            logger.info("Selecting filter: %s", plan.filter)
            filter_wheel_state = select_filter(session, plan.filter)

        achieved_hwp: Optional[float] = None
        if plan.hwp_angle_deg is not None:
            logger.info("Setting HWP angle: %.2f deg", plan.hwp_angle_deg)
            achieved_hwp = select_hwp_angle(session, plan.hwp_angle_deg)

        for idx in range(1, plan.count + 1):
            gain = plan.gain
            offset = plan.offset
            readout_mode = plan.readout_mode
            if ctx is not None:
                gain = ctx.gain_setting if gain is None else gain
                offset = ctx.offset_setting if offset is None else offset
                readout_mode = ctx.readout_mode if readout_mode is None else readout_mode

            request = CaptureRequest(
                exposure_s=plan.exposure_s,
                is_light=_frame_is_light(frame_type),
                binx=plan.binx,
                biny=plan.biny,
                startx=plan.startx,
                starty=plan.starty,
                numx=plan.numx,
                numy=plan.numy,
                gain=gain,
                offset=offset,
                readout_mode=readout_mode,
                sub_exposure_duration=plan.sub_exposure_duration,
                fast_readout=plan.fast_readout,
            )

            instrot = _pwi4_instrot_deg(pwi4)
            header = _build_header_config(
                session=session,
                pwi4=pwi4,
                plan=plan,
                target=target,
                config=config,
                filter_name=filter_name,
                ntp_status=ntp_status,
                filter_wheel_state=filter_wheel_state,
                achieved_hwp_deg=achieved_hwp,
                instrot_deg=instrot,
            )

            object_name = header.object_name
            if config.naming == "polite" and date_str:
                filename = _build_polite_filename(
                    date_str, object_name, filter_name, frame_type,
                    plan.exposure_s, idx,
                )
            else:
                filename = _build_filename(
                    frame_type=frame_type,
                    exposure_s=plan.exposure_s,
                    index=idx,
                    object_name=object_name,
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
                if reporter is not None:
                    reporter.frame_captured(
                        frame_type, filt=filter_name, hwp=achieved_hwp,
                        exp=plan.exposure_s, idx=idx, count=plan.count,
                    )
            except Exception:
                logger.exception("Capture failed for %s", out_path)
                if config.stop_on_error:
                    raise

    if block_log is not None and block_id and target and ctx:
        from .pol_config import session_pol_config_to_polconfig
        filt = filter_name if frames else None
        if filt:
            pc = session_pol_config_to_polconfig(ctx, filt)
            block_log.append({
                "block_id": block_id,
                "target": target.name,
                "filter": filt,
                "pol_config": polconfig_snapshot(pc),
            })


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
    else:
        try:
            pwi4.mount_tracking_off()
        except Exception:
            logger.warning("Could not disable mount tracking for %s", target.name)


def _maybe_update_asteroid_ephemeris(
    target: TargetPlan,
    catalog_entry: Optional[Dict[str, Any]],
    cache_path: Path,
) -> TargetPlan:
    """Fetch Horizons ephemeris for known asteroids and update RA/Dec."""
    hid = resolve_target_id(target.name, catalog_entry)
    if hid is None:
        return target
    try:
        from astropy.time import Time
        row = fetch_ephemeris(
            hid, Time.now(),
            lon_deg=JULIAN_LON_DEG, lat_deg=JULIAN_LAT_DEG, elev_m=JULIAN_ELEV_M,
            target_name=target.name,
        )
        cache_ephemeris(cache_path, row)
        logger.info(
            "Horizons %s: RA=%.4fh Dec=%.3f deg phase=%.1f deg",
            target.name, row.ra_hours, row.dec_deg, row.phase_angle_deg or 0,
        )
        return replace(
            target,
            ra_hours=row.ra_hours,
            dec_deg=row.dec_deg,
        )
    except Exception:
        logger.warning("Horizons fetch failed for %s; using catalog coords", target.name, exc_info=True)
        return target


def _run_cal_frames(
    state,
    config: NightSessionConfig,
    frames: List[FramePlan],
    session_dir: Path,
    polite_date: str,
    block_logger: Optional[BlockLogger],
    block_log: list,
    label: str,
    reporter: Optional[NightReporter] = None,
) -> None:
    if not frames:
        return
    cal_dir = session_dir / "calibrations" if config.naming == "legacy" else session_dir
    logger.info("Running calibration frames (%s)", label)
    if reporter is not None:
        reporter.start_block(f"calibration ({label})", subtitle=f"{total_frame_count(frames)} frames")
    bricks = sorted({f.source_brick for f in frames if f.source_brick})
    extra = {"stage": label, "bricks": bricks}
    if block_logger:
        with block_logger.block(f"cal_{label}", brick=",".join(bricks), extra=extra):
            _run_frames(
                state.imaging, state.pwi4, frames, cal_dir, None, config,
                ntp_status=state.ntp_status, date_str=polite_date,
                block_log=block_log, block_id=f"cal_{label}", reporter=reporter,
            )
    else:
        _run_frames(
            state.imaging, state.pwi4, frames, cal_dir, None, config,
            ntp_status=state.ntp_status, date_str=polite_date,
            block_log=block_log, block_id=f"cal_{label}", reporter=reporter,
        )
    for brick in bricks:
        for gate in gates_after_cal(config.qa_gates, brick):
            dispatch_qa_gate(gate, session_dir=session_dir, cal_brick=brick)


def _flush_pol_config_sidecar(
    session_dir: Path,
    ctx: SessionCaptureContext,
    *,
    session_id: str,
    block_log: list,
    final: bool = False,
) -> Path:
    """Write or refresh ``pol_config.yaml`` so mid-session QA gates can load PolConfig."""
    sidecar = write_pol_config_sidecar(
        session_dir / "pol_config.yaml",
        ctx,
        session_id=session_id,
        blocks=list(block_log),
    )
    if final:
        logger.info("Wrote PolConfig sidecar: %s", sidecar)
    else:
        logger.debug("Updated PolConfig sidecar: %s", sidecar)
    return sidecar


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

    if config.capture_context is not None:
        config.capture_context = _apply_session_camera(state.imaging, config.capture_context)
        _verify_session_hardware(
            state.imaging, config.capture_context, config.startup.alpaca.filter_names,
        )
        wait_for_cooler(
            state.imaging.camera,
            config.capture_context.cooler_setpoint_c,
            tolerance_c=config.capture_context.cooler_tolerance_c,
            stable_s=config.capture_context.cooler_stable_s,
            timeout_s=config.capture_context.cooler_timeout_s,
        )

    if state.log_paths is not None:
        date_dir = state.log_paths.date_dir.name
        session_id = state.log_paths.log_file.stem
    else:
        now = datetime.now(timezone.utc) if config.use_utc else datetime.now()
        date_dir = now.strftime("%Y-%m-%d")
        session_id = config.session_name or now.strftime("session_%Y%m%d_%H%M%S")

    base_dir = Path(config.base_data_dir).expanduser().resolve()
    if config.naming == "polite":
        # FITSDATA/YYYYMMDD flat layout per observation plan §9
        if config.session_name and len(str(config.session_name)) == 8:
            ymd = str(config.session_name)
        else:
            ymd = date_dir.replace("-", "")
        session_dir = base_dir / ymd
        polite_date = f"{ymd[:4]}-{ymd[4:6]}-{ymd[6:8]}"
    else:
        session_dir = base_dir / date_dir / session_id
        polite_date = date_dir
    session_dir.mkdir(parents=True, exist_ok=True)
    logger.info("Session data directory: %s", session_dir)

    block_log: list = []
    manifest_path = session_dir / "block_manifest.jsonl"
    block_logger = BlockLogger(manifest_path)
    horizons_cache = session_dir / "horizons_cache.json"
    catalog_cache: Dict[str, Any] = {}

    if config.capture_context is not None:
        _flush_pol_config_sidecar(
            session_dir,
            config.capture_context,
            session_id=config.session_name or session_id,
            block_log=block_log,
        )

    stage = config.calibration_stage
    total_frames = plan_total_frames(config)
    reporter = NightReporter(
        total_frames,
        title=config.session_name or session_id,
        estimated_duration_s=plan_estimated_duration_s(config),
    )
    with reporter:
        reporter.banner([
            ("session", session_id),
            ("data dir", session_dir),
            ("targets", ", ".join(t.name for t in config.targets) or "(none)"),
            ("cal stage", stage),
            ("total frames", total_frames),
        ])
        if config.calibration_before and stage in ("before", "both"):
            _run_cal_frames(
                state, config, config.calibration_before, session_dir, polite_date,
                block_logger, block_log, "before", reporter=reporter,
            )
            if config.capture_context is not None:
                _flush_pol_config_sidecar(
                    session_dir,
                    config.capture_context,
                    session_id=config.session_name or session_id,
                    block_log=block_log,
                )

        for target in config.targets:
            logger.info("Target: %s", target.name)
            target = _maybe_update_asteroid_ephemeris(
                target, config.catalog.get(target.name), horizons_cache,
            )

            snap = pwi4_snapshot(state.pwi4)
            moon_sep = None
            if target.ra_hours is not None and target.dec_deg is not None:
                moon_sep = moon_separation_deg(target.ra_hours, target.dec_deg)

            if config.naming == "polite":
                target_dir = session_dir
            else:
                target_dir = session_dir / "targets" / _slugify(target.name)

            reporter.start_block(
                target.name,
                subtitle=f"{total_frame_count(target.frames)} frames"
                + (f"  moon {moon_sep:.0f}°" if moon_sep is not None else ""),
            )
            block_extra = {**snap, "moon_sep_deg": moon_sep}
            with block_logger.block(
                _slugify(target.name),
                target=target.name,
                extra=block_extra,
            ):
                _slew_to_target(state.pwi4, target, limits=state.slew_limits)
                if not target.frames:
                    logger.warning("No frames defined for target %s", target.name)
                _run_frames(
                    state.imaging,
                    state.pwi4,
                    target.frames,
                    target_dir,
                    target,
                    config,
                    ntp_status=state.ntp_status,
                    date_str=polite_date,
                    block_log=block_log,
                    block_id=_slugify(target.name),
                    reporter=reporter,
                )

            if config.capture_context is not None:
                _flush_pol_config_sidecar(
                    session_dir,
                    config.capture_context,
                    session_id=config.session_name or session_id,
                    block_log=block_log,
                )

            for gate in gates_after_target(config.qa_gates, target.name):
                result = dispatch_qa_gate(gate, session_dir=session_dir, target_name=target.name)
                reporter.qa(result)

        if config.calibration_after and stage in ("after", "both"):
            _run_cal_frames(
                state, config, config.calibration_after, session_dir, polite_date,
                block_logger, block_log, "after", reporter=reporter,
            )
            if config.capture_context is not None:
                _flush_pol_config_sidecar(
                    session_dir,
                    config.capture_context,
                    session_id=config.session_name or session_id,
                    block_log=block_log,
                )

        for gate in config.qa_gates:
            if gate.handler == "sequence_audit" and not gate.after_target and not gate.after_cal:
                result = dispatch_qa_gate(gate, session_dir=session_dir)
                reporter.qa(result)

        if config.capture_context is not None:
            _flush_pol_config_sidecar(
                session_dir,
                config.capture_context,
                session_id=config.session_name or session_id,
                block_log=block_log,
                final=True,
            )

        reporter.note("✓ Night session complete", style="moss")
    logger.info("Night session complete")
    state.imaging.close()
