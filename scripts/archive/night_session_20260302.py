#!/usr/bin/env python3
"""
Night Session Script: March 2, 2026
Observatory: Julian, CA
Observation 2: Sensor characterization & calibration

Stacks:
  1. 9x 180s darks, 1x1 bin       (sensor noise estimate)
  2. 5x 60s darks, 1x1 bin        (sensor noise estimate)
  3. 80x 0.1s bias, 2x2 bin       (readout noise estimate)
  4. 40x sky flats, L/Clear, auto-expose 25k-35k ADU
  5. 20x 60s lights, L/Clear, illumination map (autoguided, dithered)
  6. 10x 30s lights, L/Clear, defocused dust map (no dithering)

Sequence:
  Phase 1 (before sunset): Darks + Bias
  Phase 2 (twilight):      Sky flats with auto-exposure
  Phase 3 (dark sky):      Illumination map (autoguided)
  Phase 4 (dark sky):      Defocused dust map
"""

from __future__ import annotations

import logging
import random
import sys
import time
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
from astropy.coordinates import AltAz, EarthLocation, get_sun
from astropy.time import Time
import astropy.units as u

from alpyca_tools.camera_ops import (
    ExposureSettings,
    configure_camera,
    download_image,
)
from alpyca_tools.fits_writer import FitsHeaderConfig, build_header, write_fits

from obs_utils.autoguide import (
    dither_mount_offset_arcsec,
    pulse_guide,
    random_dither_mount_offset_arcsec,
)
from obs_utils.config import AlpacaConfig, Pwi4Config, SkyRegionLimit, SlewLimits
from obs_utils.imaging import (
    CaptureRequest,
    capture_fits_file,
    capture_image_array,
    select_filter,
)
from obs_utils.logging import LoggingConfig
from obs_utils.mount import slew_altaz, wait_for_slew
from obs_utils.startup import StartupConfig, StartupState, startup_observatory

logger = logging.getLogger(__name__)


# ============================================================================
# SITE-DETERMINED VALUES — UPDATE BEFORE RUNNING
# ============================================================================

# Device indices (check ASCOM Remote Server at site)
CAMERA_INDEX = 0            # Main imaging camera (STX-16803)
GUIDE_CAMERA_INDEX = 1      # Internal guide chip
FILTERWHEEL_INDEX = 0       # SBIG AFW

# Filter names in AFW slot order (inspect at site)
FILTER_NAMES = ["Clear/Luminance", "Red", "Blue", "Green", "Halpha"]
LCLEAR_FILTER = "Clear/Luminance"  # Name matching the L/Clear position

# Focuser defocus for dust map (test at site)
DUST_DEFOCUS_STEPS = 1000   # Focuser steps to move from nominal focus

# Guide camera pixel scale (TC-237 guide chip at CDK20 f/6.8, ~0.44"/px est.)
GUIDE_PIXEL_SCALE_ARCSEC = 0.44


# ============================================================================
# FIXED PARAMETERS
# ============================================================================

# Site location
SITE_LAT_DEG = 33.08
SITE_LON_DEG = -116.60
SITE_ELEV_M = 1300.0

# Observation date — UTC date for the evening of March 2 PST
OBS_DATE = "2026-03-03"

# Minimum elevation angle (shed clearance)
MIN_ELEVATION_DEG = 55.0

# Sky flat parameters
FLAT_ADU_MIN = 25000
FLAT_ADU_MAX = 35000
FLAT_ADU_TARGET = 30000     # Midpoint target for exposure scaling
FLAT_COUNT = 40
FLAT_INITIAL_EXP_S = 2.0
FLAT_MIN_EXP_S = 0.1
FLAT_MAX_EXP_S = 30.0
FLAT_DITHER_ARCSEC = 15.0   # Large dither to shift stars between flats

# Stack quantity selections (from observation sheet ranges)
DARK_60S_COUNT = 5           # Sheet says 3-5
BIAS_COUNT = 80              # Sheet says 50-80

# Light frame exposures (from observation sheet ranges)
ILLUM_EXP_S = 60.0           # Sheet says 60-90
ILLUM_COUNT = 20
ILLUM_DITHER_ARCSEC = 2.7    # ~5 px at 0.537"/px (4-6 px range)

DUST_EXP_S = 30.0            # Sheet says 30-60
DUST_COUNT = 10

# Autoguiding parameters
GUIDE_EXPOSURE_S = 2.0       # Guide frame exposure time
GUIDE_RATE_ARCSEC_S = 15.0   # Guide speed (sidereal rate)
GUIDE_MIN_CORRECTION_ARCSEC = 0.5  # Dead zone — ignore drift below this
GUIDE_MAX_CORRECTION_MS = 2000     # Max pulse guide duration

# Sky flat pointing direction
SKYFLAT_ALT_DEG = 80.0
SKYFLAT_AZ_DEG = 80.0        # Anti-solar for evening twilight

# Zenith pointing (illumination map & dust map)
ZENITH_ALT_DEG = 85.0        # Slightly off-zenith for mount comfort
ZENITH_AZ_DEG = 0.0

# Session metadata
OBSERVER = ""
TELESCOPE = "PlaneWave CDK20"
OBSERVATORY = "Julian, CA"
INSTRUMENT = "SBIG STX-16803"

# Base data directory
BASE_DATA_DIR = "FITSDATA"
SESSION_DATE_DIR = "20260302"

# Interactive mode — pause for user confirmation between phases
INTERACTIVE = True

# Sun altitude thresholds for scheduling
SUN_ALT_FLAT_START_DEG = -2.0   # Begin attempting sky flats
SUN_ALT_ASTRO_TWILIGHT_DEG = -18.0  # Astronomical darkness


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def get_site_location() -> EarthLocation:
    """Return the observatory EarthLocation."""
    return EarthLocation(
        lat=SITE_LAT_DEG * u.deg,
        lon=SITE_LON_DEG * u.deg,
        height=SITE_ELEV_M * u.m,
    )


def compute_sun_altitude(location: EarthLocation, t: Time) -> float:
    """Return the sun's altitude in degrees at the given time and location."""
    altaz_frame = AltAz(obstime=t, location=location)
    sun_altaz = get_sun(t).transform_to(altaz_frame)
    return float(sun_altaz.alt.deg)


def wait_for_sun_altitude(
    location: EarthLocation,
    target_alt_deg: float,
    setting: bool = True,
    poll_s: float = 30.0,
) -> None:
    """Block until the sun crosses the target altitude.

    Parameters
    ----------
    location : EarthLocation
        Observatory position.
    target_alt_deg : float
        Sun altitude threshold in degrees (negative for below horizon).
    setting : bool
        If True, wait for the sun to drop *below* target_alt_deg.
        If False, wait for the sun to rise *above* target_alt_deg.
    poll_s : float
        Polling interval in seconds.
    """
    while True:
        now = Time.now()
        alt = compute_sun_altitude(location, now)
        if setting and alt <= target_alt_deg:
            logger.info(
                "Sun altitude %.1f° <= %.1f° — proceeding",
                alt, target_alt_deg,
            )
            return
        if not setting and alt >= target_alt_deg:
            logger.info(
                "Sun altitude %.1f° >= %.1f° — proceeding",
                alt, target_alt_deg,
            )
            return
        logger.info(
            "Sun altitude %.1f° — waiting for %.1f° (%s) ...",
            alt,
            target_alt_deg,
            "setting" if setting else "rising",
        )
        time.sleep(poll_s)


def log_twilight_times(location: EarthLocation) -> None:
    """Compute and log approximate twilight times for the observation date."""
    base = Time(OBS_DATE + "T00:00:00", scale="utc")
    # Scan from noon UTC (early morning PST) through the night
    times = base + np.linspace(0, 14, 1000) * u.hour

    sun_alts = np.array([compute_sun_altitude(location, t) for t in times])

    thresholds = {
        "Sunset (0°)": 0.0,
        "Civil twilight (-6°)": -6.0,
        "Nautical twilight (-12°)": -12.0,
        "Astronomical twilight (-18°)": -18.0,
    }
    for label, alt_thresh in thresholds.items():
        # Find first crossing below threshold
        crossings = np.where(np.diff(np.sign(sun_alts - alt_thresh)))[0]
        if len(crossings) > 0:
            t_cross = times[crossings[0]]
            # Convert to PST (UTC-8)
            pst_hour = (t_cross.datetime.hour - 8) % 24
            pst_min = t_cross.datetime.minute
            logger.info(
                "  %s: ~%02d:%02d PST (UTC %s)",
                label, pst_hour, pst_min, t_cross.iso[:19],
            )


def prompt_user(message: str) -> None:
    """If INTERACTIVE mode is on, pause and wait for user confirmation."""
    if not INTERACTIVE:
        return
    try:
        input(f"\n>>> {message} [Press Enter to continue, Ctrl+C to abort] ")
    except KeyboardInterrupt:
        logger.info("User aborted at interactive prompt")
        sys.exit(0)


def make_session_dir() -> Path:
    """Create and return the session output directory."""
    session_dir = Path(BASE_DATA_DIR) / SESSION_DATE_DIR
    session_dir.mkdir(parents=True, exist_ok=True)
    return session_dir


# ============================================================================
# PHASE 1: DARKS & BIAS
# ============================================================================

def run_calibration_frames(state: StartupState, session_dir: Path) -> None:
    """Capture dark and bias calibration frames (Stacks 1-3).

    No pointing required — shutter stays closed for all frames.
    """
    logger.info("=" * 60)
    logger.info("PHASE 1: Calibration Frames (Darks & Bias)")
    logger.info("=" * 60)

    imaging = state.imaging

    header_base = FitsHeaderConfig(
        observer=OBSERVER or None,
        telescope=TELESCOPE,
        observatory=OBSERVATORY,
        instrument=INSTRUMENT,
    )

    # --- Stack 1: 9x 180s darks, 1x1 bin ---
    logger.info("Stack 1: 9x 180s dark frames, 1x1 bin")
    dark_dir = session_dir / "DARK"
    dark_dir.mkdir(parents=True, exist_ok=True)

    for i in range(1, 10):
        request = CaptureRequest(exposure_s=180.0, is_light=False, binx=1, biny=1)
        header = FitsHeaderConfig(
            imagetyp="DARK",
            object_name="SensorNoise",
            observer=header_base.observer,
            telescope=header_base.telescope,
            observatory=header_base.observatory,
            instrument=header_base.instrument,
        )
        filename = f"DARK_SensorNoise_exp180s_{i:03d}.fits"
        out_path = dark_dir / filename
        logger.info("  Capturing dark %d/9 (180s) -> %s", i, out_path)
        try:
            capture_fits_file(imaging, request, header, out_path)
        except Exception:
            logger.exception("  Failed to capture dark %d/9", i)
            raise

    # --- Stack 2: 5x 60s darks, 1x1 bin ---
    logger.info("Stack 2: %dx 60s dark frames, 1x1 bin", DARK_60S_COUNT)

    for i in range(1, DARK_60S_COUNT + 1):
        request = CaptureRequest(exposure_s=60.0, is_light=False, binx=1, biny=1)
        header = FitsHeaderConfig(
            imagetyp="DARK",
            object_name="SensorNoise",
            observer=header_base.observer,
            telescope=header_base.telescope,
            observatory=header_base.observatory,
            instrument=header_base.instrument,
        )
        filename = f"DARK_SensorNoise_exp60s_{i:03d}.fits"
        out_path = dark_dir / filename
        logger.info(
            "  Capturing dark %d/%d (60s) -> %s", i, DARK_60S_COUNT, out_path,
        )
        try:
            capture_fits_file(imaging, request, header, out_path)
        except Exception:
            logger.exception("  Failed to capture dark %d/%d", i, DARK_60S_COUNT)
            raise

    # --- Stack 3: 80x 0.1s bias, 1x1 bin ---
    logger.info("Stack 3: %dx 0.1s bias frames, 1x1 bin", BIAS_COUNT)
    bias_dir = session_dir / "BIAS"
    bias_dir.mkdir(parents=True, exist_ok=True)

    for i in range(1, BIAS_COUNT + 1):
        request = CaptureRequest(exposure_s=0.1, is_light=False, binx=1, biny=1)
        header = FitsHeaderConfig(
            imagetyp="BIAS",
            object_name="ReadoutNoise",
            observer=header_base.observer,
            telescope=header_base.telescope,
            observatory=header_base.observatory,
            instrument=header_base.instrument,
        )
        filename = f"BIAS_ReadoutNoise_exp0.1s_{i:03d}.fits"
        out_path = bias_dir / filename
        if i % 10 == 1 or i == BIAS_COUNT:
            logger.info(
                "  Capturing bias %d/%d (0.1s, 1x1) -> %s",
                i, BIAS_COUNT, out_path,
            )
        try:
            capture_fits_file(imaging, request, header, out_path)
        except Exception:
            logger.exception("  Failed to capture bias %d/%d", i, BIAS_COUNT)
            raise

    logger.info("Phase 1 complete: darks and bias captured")


# ============================================================================
# PHASE 2: SKY FLATS
# ============================================================================

def auto_expose_sky_flats(
    state: StartupState,
    session_dir: Path,
    location: EarthLocation,
) -> int:
    """Capture sky flat frames with auto-exposure during twilight.

    Slews to the sky flat position, waits for appropriate sky brightness,
    then runs an auto-exposure loop adjusting exposure time to maintain
    the median ADU within the target range.

    Returns the number of flats captured.
    """
    logger.info("=" * 60)
    logger.info("PHASE 2: Sky Flats (auto-exposure)")
    logger.info("=" * 60)

    pwi4 = state.pwi4
    imaging = state.imaging

    flat_dir = session_dir / "FLAT"
    flat_dir.mkdir(parents=True, exist_ok=True)

    # Slew to sky flat position
    logger.info(
        "Slewing to sky flat position: alt=%.1f°, az=%.1f°",
        SKYFLAT_ALT_DEG, SKYFLAT_AZ_DEG,
    )
    slew_altaz(pwi4, SKYFLAT_ALT_DEG, SKYFLAT_AZ_DEG, limits=state.slew_limits)
    wait_for_slew(pwi4)
    pwi4.mount_tracking_on()

    # Select L/Clear filter
    logger.info("Selecting filter: %s", LCLEAR_FILTER)
    select_filter(imaging, LCLEAR_FILTER)

    # Wait for sun to approach flat-taking altitude
    logger.info(
        "Waiting for sun altitude <= %.1f° to begin flat sequence...",
        SUN_ALT_FLAT_START_DEG,
    )
    wait_for_sun_altitude(location, SUN_ALT_FLAT_START_DEG, setting=True, poll_s=30.0)

    # Auto-exposure loop
    current_exp = FLAT_INITIAL_EXP_S
    flats_captured = 0
    consecutive_bright = 0
    max_bright_retries = 20  # Give up if sky stays too bright for too many tries

    filter_name = LCLEAR_FILTER

    while flats_captured < FLAT_COUNT:
        logger.info(
            "  Flat attempt %d/%d: exposure=%.2fs",
            flats_captured + 1, FLAT_COUNT, current_exp,
        )

        # Capture test/flat frame
        request = CaptureRequest(
            exposure_s=current_exp, is_light=True, binx=1, biny=1,
        )
        try:
            data, dtype = capture_image_array(imaging, request)
        except Exception:
            logger.exception("  Flat capture failed — retrying")
            time.sleep(5)
            continue

        median_adu = float(np.median(data))
        logger.info("  Median ADU: %.0f (target: %d-%d)", median_adu, FLAT_ADU_MIN, FLAT_ADU_MAX)

        if median_adu > FLAT_ADU_MAX:
            # Sky too bright
            if current_exp <= FLAT_MIN_EXP_S:
                consecutive_bright += 1
                if consecutive_bright >= max_bright_retries:
                    logger.warning("  Sky still too bright after %d retries — aborting flats", max_bright_retries)
                    break
                logger.info("  Sky too bright at minimum exposure — waiting 30s...")
                time.sleep(30)
                continue
            # Reduce exposure
            new_exp = current_exp * (FLAT_ADU_TARGET / median_adu)
            current_exp = max(FLAT_MIN_EXP_S, min(FLAT_MAX_EXP_S, new_exp))
            consecutive_bright = 0
            logger.info("  Adjusting exposure to %.2fs (too bright)", current_exp)
            continue

        if median_adu < FLAT_ADU_MIN:
            # Sky too dim
            if current_exp >= FLAT_MAX_EXP_S:
                logger.warning(
                    "  Sky too dim at maximum exposure (%.1fs) — ending flat sequence",
                    FLAT_MAX_EXP_S,
                )
                break
            # Increase exposure
            new_exp = current_exp * (FLAT_ADU_TARGET / median_adu)
            current_exp = max(FLAT_MIN_EXP_S, min(FLAT_MAX_EXP_S, new_exp))
            consecutive_bright = 0
            logger.info("  Adjusting exposure to %.2fs (too dim)", current_exp)
            continue

        # Median is in range — save the frame
        consecutive_bright = 0
        flats_captured += 1

        header_cfg = FitsHeaderConfig(
            imagetyp="FLAT",
            object_name="SkyFlat",
            observer=OBSERVER or None,
            telescope=TELESCOPE,
            observatory=OBSERVATORY,
            instrument=INSTRUMENT,
            filter_name=filter_name,
        )
        header = build_header(imaging.camera, header_cfg, dtype, data.shape)
        filename = f"FLAT_SkyFlat_f{filter_name.replace('/', '_')}_exp{current_exp:g}s_{flats_captured:03d}.fits"
        out_path = flat_dir / filename

        write_fits(out_path, data, header)
        logger.info("  Saved flat %d/%d -> %s", flats_captured, FLAT_COUNT, out_path)

        # Adjust exposure for next frame (sky is dimming during evening)
        new_exp = current_exp * (FLAT_ADU_TARGET / median_adu)
        current_exp = max(FLAT_MIN_EXP_S, min(FLAT_MAX_EXP_S, new_exp))

        # Dither between frames
        if flats_captured < FLAT_COUNT:
            ra_dither = random.uniform(-FLAT_DITHER_ARCSEC, FLAT_DITHER_ARCSEC)
            dec_dither = random.uniform(-FLAT_DITHER_ARCSEC, FLAT_DITHER_ARCSEC)
            dither_mount_offset_arcsec(
                pwi4, ra_arcsec=ra_dither, dec_arcsec=dec_dither, settle_s=1.0,
            )

    logger.info("Phase 2 complete: %d/%d sky flats captured", flats_captured, FLAT_COUNT)
    return flats_captured


# ============================================================================
# AUTOGUIDING FUNCTIONS
# ============================================================================

def centroid_box(
    data: np.ndarray,
    cx: int,
    cy: int,
    box_size: int = 15,
) -> Tuple[float, float]:
    """Compute background-subtracted center-of-mass centroid in a box.

    Parameters
    ----------
    data : np.ndarray
        2D image array (row, col).
    cx, cy : int
        Approximate center column (x) and row (y) of the star.
    box_size : int
        Half-size of the box in pixels.

    Returns
    -------
    (x, y) : tuple of float
        Sub-pixel centroid position (column, row).
    """
    ny, nx = data.shape
    x0 = max(0, cx - box_size)
    x1 = min(nx, cx + box_size + 1)
    y0 = max(0, cy - box_size)
    y1 = min(ny, cy + box_size + 1)

    sub = data[y0:y1, x0:x1].astype(np.float64)
    sub -= np.median(sub)
    sub = np.clip(sub, 0.0, None)

    total = sub.sum()
    if total == 0:
        return float(cx), float(cy)

    ys, xs = np.mgrid[y0:y1, x0:x1]
    cent_x = float((xs * sub).sum() / total)
    cent_y = float((ys * sub).sum() / total)
    return cent_x, cent_y


def find_guide_star(
    guide_camera,
    exposure_s: float = GUIDE_EXPOSURE_S,
) -> Optional[Tuple[float, float]]:
    """Take a guide frame and find the brightest star.

    Returns the sub-pixel centroid (x, y) of the brightest star,
    or None if no suitable guide star is found.
    """
    settings = ExposureSettings(exposure_s=exposure_s, is_light=True)
    configure_camera(guide_camera, settings)
    guide_camera.StartExposure(exposure_s, True)

    # Wait for guide frame
    t0 = time.time()
    while not guide_camera.ImageReady:
        if time.time() - t0 > exposure_s + 30.0:
            logger.warning("Guide frame timed out")
            return None
        time.sleep(0.5)

    data, _info, _dtype = download_image(guide_camera)

    # Find brightest pixel
    bg = np.median(data)
    peak = np.max(data)

    # Require the peak to be significantly above background
    if peak < bg * 3:
        logger.warning(
            "No guide star found (peak=%.0f, bg=%.0f, ratio=%.1f)",
            peak, bg, peak / bg if bg > 0 else 0,
        )
        return None

    peak_idx = np.unravel_index(np.argmax(data), data.shape)
    peak_y, peak_x = int(peak_idx[0]), int(peak_idx[1])

    # Refine with centroid
    cx, cy = centroid_box(data, peak_x, peak_y, box_size=15)
    logger.info("Guide star found at (%.1f, %.1f), peak=%.0f, bg=%.0f", cx, cy, peak, bg)
    return cx, cy


def guided_exposure(
    state: StartupState,
    exposure_s: float,
    ref_centroid: Tuple[float, float],
) -> Tuple[np.ndarray, np.dtype]:
    """Capture a main camera exposure while running a guide correction loop.

    Parameters
    ----------
    state : StartupState
        Contains main camera, guide camera, and PWI4 handles.
    exposure_s : float
        Main camera exposure time in seconds.
    ref_centroid : (x, y)
        Reference guide star centroid position.

    Returns
    -------
    (data, dtype) : tuple
        The captured image array and its data type.
    """
    main_cam = state.imaging.camera
    guide_cam = state.imaging.guide_camera
    ref_x, ref_y = ref_centroid

    # Configure and start main exposure
    main_settings = ExposureSettings(exposure_s=exposure_s, is_light=True, binx=1, biny=1)
    configure_camera(main_cam, main_settings)
    main_cam.StartExposure(exposure_s, True)
    logger.info("  Main exposure started (%.1fs)", exposure_s)

    # Guide correction loop — runs while main camera is exposing
    guide_cycles = 0
    corrections_applied = 0

    while not main_cam.ImageReady:
        # Take a guide frame
        guide_settings = ExposureSettings(
            exposure_s=GUIDE_EXPOSURE_S, is_light=True,
        )
        configure_camera(guide_cam, guide_settings)
        guide_cam.StartExposure(GUIDE_EXPOSURE_S, True)

        # Wait for guide frame
        g_t0 = time.time()
        while not guide_cam.ImageReady:
            if time.time() - g_t0 > GUIDE_EXPOSURE_S + 15.0:
                logger.warning("  Guide frame timed out during cycle %d", guide_cycles)
                break
            # Also check if main exposure finished while we wait
            if main_cam.ImageReady:
                break
            time.sleep(0.3)

        if main_cam.ImageReady:
            break

        if not guide_cam.ImageReady:
            continue

        # Download and analyze guide frame
        try:
            g_data, _g_info, _g_dtype = download_image(guide_cam)
        except Exception:
            logger.warning("  Failed to download guide frame")
            continue

        # Find guide star in this frame
        bg = np.median(g_data)
        peak = np.max(g_data)
        if peak < bg * 2:
            logger.debug("  Guide star lost in cycle %d", guide_cycles)
            guide_cycles += 1
            continue

        peak_idx = np.unravel_index(np.argmax(g_data), g_data.shape)
        peak_y, peak_x = int(peak_idx[0]), int(peak_idx[1])
        cx, cy = centroid_box(g_data, peak_x, peak_y, box_size=15)

        # Compute drift in pixels
        dx_px = cx - ref_x
        dy_px = cy - ref_y

        # Convert to arcseconds
        dx_arcsec = dx_px * GUIDE_PIXEL_SCALE_ARCSEC
        dy_arcsec = dy_px * GUIDE_PIXEL_SCALE_ARCSEC
        drift_total = (dx_arcsec**2 + dy_arcsec**2) ** 0.5

        guide_cycles += 1

        if drift_total < GUIDE_MIN_CORRECTION_ARCSEC:
            logger.debug(
                "  Guide cycle %d: drift=%.2f\" (below threshold)",
                guide_cycles, drift_total,
            )
            continue

        # Apply corrections via pulse guide
        # RA correction: positive dx means star moved east, correct west
        if abs(dx_arcsec) >= GUIDE_MIN_CORRECTION_ARCSEC:
            ra_dur_ms = int(min(
                abs(dx_arcsec) / GUIDE_RATE_ARCSEC_S * 1000,
                GUIDE_MAX_CORRECTION_MS,
            ))
            ra_dir = "west" if dx_arcsec > 0 else "east"
            try:
                pulse_guide(guide_cam, ra_dir, ra_dur_ms)
            except Exception:
                logger.warning("  RA pulse guide failed")

        # Dec correction: positive dy means star moved north, correct south
        if abs(dy_arcsec) >= GUIDE_MIN_CORRECTION_ARCSEC:
            dec_dur_ms = int(min(
                abs(dy_arcsec) / GUIDE_RATE_ARCSEC_S * 1000,
                GUIDE_MAX_CORRECTION_MS,
            ))
            dec_dir = "south" if dy_arcsec > 0 else "north"
            try:
                pulse_guide(guide_cam, dec_dir, dec_dur_ms)
            except Exception:
                logger.warning("  Dec pulse guide failed")

        corrections_applied += 1
        logger.info(
            "  Guide cycle %d: drift=%.2f\" (dx=%.2f\", dy=%.2f\") — corrected",
            guide_cycles, drift_total, dx_arcsec, dy_arcsec,
        )

    # Download main image
    logger.info(
        "  Main exposure complete (%d guide cycles, %d corrections)",
        guide_cycles, corrections_applied,
    )
    data, _info, dtype = download_image(main_cam)
    return data, dtype


# ============================================================================
# PHASE 3: ILLUMINATION MAP
# ============================================================================

def run_illumination_map(
    state: StartupState,
    session_dir: Path,
    location: EarthLocation,
) -> None:
    """Capture illumination map frames at zenith with autoguiding and dithering.

    Stack 5: 20x 60s light frames, L/Clear, 1x1 bin.
    """
    logger.info("=" * 60)
    logger.info("PHASE 3: Illumination Map (autoguided, dithered)")
    logger.info("=" * 60)

    pwi4 = state.pwi4
    imaging = state.imaging

    # Wait for astronomical darkness
    logger.info("Waiting for astronomical twilight (sun < %.0f°)...", SUN_ALT_ASTRO_TWILIGHT_DEG)
    wait_for_sun_altitude(location, SUN_ALT_ASTRO_TWILIGHT_DEG, setting=True, poll_s=30.0)

    # Slew to near-zenith
    logger.info("Slewing to zenith: alt=%.1f°, az=%.1f°", ZENITH_ALT_DEG, ZENITH_AZ_DEG)
    slew_altaz(pwi4, ZENITH_ALT_DEG, ZENITH_AZ_DEG, limits=state.slew_limits)
    wait_for_slew(pwi4)
    pwi4.mount_tracking_on()

    # Select L/Clear filter
    logger.info("Selecting filter: %s", LCLEAR_FILTER)
    select_filter(imaging, LCLEAR_FILTER)

    light_dir = session_dir / "LIGHT"
    light_dir.mkdir(parents=True, exist_ok=True)

    filter_name = LCLEAR_FILTER
    use_guiding = True

    # Acquire guide star
    if imaging.guide_camera is None:
        logger.warning("No guide camera connected — falling back to tracking-only")
        use_guiding = False
    else:
        ref_centroid = find_guide_star(imaging.guide_camera, GUIDE_EXPOSURE_S)
        if ref_centroid is None:
            logger.warning("Could not acquire guide star — falling back to tracking-only")
            use_guiding = False

    # Capture frames
    for i in range(1, ILLUM_COUNT + 1):
        logger.info("Illumination map frame %d/%d (%.0fs)", i, ILLUM_COUNT, ILLUM_EXP_S)

        header_cfg = FitsHeaderConfig(
            imagetyp="LIGHT",
            object_name="IlluminationMap",
            observer=OBSERVER or None,
            telescope=TELESCOPE,
            observatory=OBSERVATORY,
            instrument=INSTRUMENT,
            filter_name=filter_name,
        )

        filename = f"LIGHT_IlluminationMap_f{filter_name.replace('/', '_')}_exp{ILLUM_EXP_S:g}s_{i:03d}.fits"
        out_path = light_dir / filename

        try:
            if use_guiding:
                data, dtype = guided_exposure(state, ILLUM_EXP_S, ref_centroid)
                header = build_header(imaging.camera, header_cfg, dtype, data.shape)
                write_fits(out_path, data, header)
            else:
                request = CaptureRequest(
                    exposure_s=ILLUM_EXP_S, is_light=True, binx=1, biny=1,
                )
                capture_fits_file(imaging, request, header_cfg, out_path)

            logger.info("  Saved -> %s", out_path)
        except Exception:
            logger.exception("  Failed to capture illumination frame %d/%d", i, ILLUM_COUNT)
            continue

        # Dither between frames
        if i < ILLUM_COUNT:
            ra_d, dec_d = random_dither_mount_offset_arcsec(
                pwi4, max_arcsec=ILLUM_DITHER_ARCSEC, settle_s=2.0,
            )
            logger.info("  Dithered by (%.1f\", %.1f\")", ra_d, dec_d)

            # Re-acquire guide star at new position
            if use_guiding:
                new_centroid = find_guide_star(imaging.guide_camera, GUIDE_EXPOSURE_S)
                if new_centroid is not None:
                    ref_centroid = new_centroid
                else:
                    logger.warning("  Guide star lost after dither — using last known position")

    logger.info("Phase 3 complete: illumination map captured")


# ============================================================================
# PHASE 4: DEFOCUSED DUST MAP
# ============================================================================

def wait_for_focuser(pwi4, poll_s: float = 0.5, timeout_s: float = 60.0) -> None:
    """Poll until the focuser stops moving."""
    t0 = time.time()
    while True:
        status = pwi4.status()
        if not status.focuser.is_moving:
            return
        if time.time() - t0 > timeout_s:
            logger.warning("Focuser move timed out after %.0fs", timeout_s)
            return
        time.sleep(poll_s)


def run_dust_map(state: StartupState, session_dir: Path) -> None:
    """Capture defocused dust map frames at zenith, no dithering.

    Stack 6: 10x 30s light frames, L/Clear, 1x1 bin, defocused.
    """
    logger.info("=" * 60)
    logger.info("PHASE 4: Defocused Dust Map")
    logger.info("=" * 60)

    pwi4 = state.pwi4
    imaging = state.imaging

    light_dir = session_dir / "LIGHT"
    light_dir.mkdir(parents=True, exist_ok=True)

    filter_name = LCLEAR_FILTER

    # Read current focuser position
    status = pwi4.status()
    nominal_position = status.focuser.position
    logger.info("Current focuser position: %.0f", nominal_position)

    # Defocus
    defocus_target = nominal_position + DUST_DEFOCUS_STEPS
    logger.info(
        "Defocusing: %.0f -> %.0f (+%d steps)",
        nominal_position, defocus_target, DUST_DEFOCUS_STEPS,
    )
    pwi4.focuser_goto(defocus_target)
    wait_for_focuser(pwi4)

    try:
        # Ensure we're pointing at zenith
        logger.info("Slewing to zenith: alt=%.1f°, az=%.1f°", ZENITH_ALT_DEG, ZENITH_AZ_DEG)
        slew_altaz(pwi4, ZENITH_ALT_DEG, ZENITH_AZ_DEG, limits=state.slew_limits)
        wait_for_slew(pwi4)
        pwi4.mount_tracking_on()

        # Select L/Clear filter
        select_filter(imaging, LCLEAR_FILTER)

        # Capture frames — no dithering
        for i in range(1, DUST_COUNT + 1):
            logger.info("Dust map frame %d/%d (%.0fs, defocused)", i, DUST_COUNT, DUST_EXP_S)

            request = CaptureRequest(
                exposure_s=DUST_EXP_S, is_light=True, binx=1, biny=1,
            )
            header_cfg = FitsHeaderConfig(
                imagetyp="LIGHT",
                object_name="DustMap",
                observer=OBSERVER or None,
                telescope=TELESCOPE,
                observatory=OBSERVATORY,
                instrument=INSTRUMENT,
                filter_name=filter_name,
            )
            filename = f"LIGHT_DustMap_f{filter_name.replace('/', '_')}_exp{DUST_EXP_S:g}s_{i:03d}.fits"
            out_path = light_dir / filename

            try:
                capture_fits_file(imaging, request, header_cfg, out_path)
                logger.info("  Saved -> %s", out_path)
            except Exception:
                logger.exception("  Failed to capture dust map frame %d/%d", i, DUST_COUNT)
                continue

    finally:
        # Always refocus even if captures fail
        logger.info("Refocusing: %.0f -> %.0f", defocus_target, nominal_position)
        pwi4.focuser_goto(nominal_position)
        wait_for_focuser(pwi4)
        logger.info("Focuser restored to nominal position")

    logger.info("Phase 4 complete: dust map captured")


# ============================================================================
# MAIN ORCHESTRATION
# ============================================================================

def main() -> None:
    """Orchestrate the full observation sequence for March 2, 2026."""

    # --- Create output directory first (needed for log file) ---
    session_dir = make_session_dir()

    # --- Setup logging ---
    log_file = session_dir / "session.log"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(log_file, mode="a"),
        ],
    )

    logger.info("=" * 60)
    logger.info("NIGHT SESSION: March 2, 2026 — Julian, CA")
    logger.info("Observation 2: Sensor Characterization & Calibration")
    logger.info("=" * 60)
    logger.info("Session output directory: %s", session_dir)
    logger.info("Log file: %s", log_file)

    location = get_site_location()

    # Log twilight times for reference
    logger.info("Computed twilight times for %s:", OBS_DATE)
    log_twilight_times(location)

    # Log current sun altitude
    now_alt = compute_sun_altitude(location, Time.now())
    logger.info("Current sun altitude: %.1f°", now_alt)

    # --- Observatory startup ---
    logger.info("Starting observatory initialization...")
    startup_cfg = StartupConfig(
        pwi4=Pwi4Config(host="localhost", port=8220),
        alpaca=AlpacaConfig(
            host="localhost:11111",
            camera_index=CAMERA_INDEX,
            guide_camera_index=GUIDE_CAMERA_INDEX,
            filterwheel_index=FILTERWHEEL_INDEX,
            filter_names=FILTER_NAMES,
        ),
        slew_limits=SlewLimits(
            regions=[
                SkyRegionLimit(
                    name="shed_clearance",
                    alt_min_deg=MIN_ELEVATION_DEG,
                    alt_max_deg=90.0,
                    az_min_deg=0.0,
                    az_max_deg=360.0,
                ),
            ],
            enforce_regions=True,
        ),
        logging=LoggingConfig(
            session_name="obs2_20260302",
            base_dir=str(Path(BASE_DATA_DIR) / SESSION_DATE_DIR / "logs"),
        ),
    )
    state = startup_observatory(startup_cfg)
    logger.info("Observatory startup complete")

    try:
        # --- Phase 1: Darks & Bias ---
        prompt_user("Ready to begin Phase 1 (darks & bias)?")
        run_calibration_frames(state, session_dir)

        # --- Phase 2: Sky Flats ---
        prompt_user("Ready to begin Phase 2 (sky flats)? Ensure telescope cover is open.")
        flats_captured = auto_expose_sky_flats(state, session_dir, location)
        if flats_captured < FLAT_COUNT:
            logger.warning(
                "Only captured %d/%d flats — continuing with remaining phases",
                flats_captured, FLAT_COUNT,
            )

        # --- Phase 3: Illumination Map ---
        prompt_user("Ready to begin Phase 3 (illumination map)?")
        run_illumination_map(state, session_dir, location)

        # --- Phase 4: Dust Map ---
        prompt_user("Ready to begin Phase 4 (defocused dust map)?")
        run_dust_map(state, session_dir)

    finally:
        # --- Shutdown ---
        logger.info("=" * 60)
        logger.info("SESSION COMPLETE — shutting down")
        logger.info("=" * 60)
        try:
            state.imaging.close()
            logger.info("Imaging session closed")
        except Exception:
            logger.exception("Error closing imaging session")

    logger.info("Night session finished successfully")


if __name__ == "__main__":
    main()
