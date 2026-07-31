from __future__ import annotations

"""Interactive (Jupyter) session manager for the POLITE observatory.

The night-session *scripts* connect, run a fixed plan, and exit. Notebooks
instead keep one long-lived kernel, so a device object created in one cell stays
connected for every later cell -- set filter slot 2 in one cell, slot 3 in the
next, no reconnect. This module wraps that pattern safely:

  * A module-level **singleton** so re-running the connect cell reuses the live
    connections instead of leaking Alpaca sessions or double-opening the Pyxis
    serial port (which is exclusive -- only one process may hold it).
  * ``connect()`` "tops up" only the subsystems that are not already live, so it
    is safe to re-run.
  * Thin action methods (``filter``/``hwp``/``expose``) over the existing
    ``obs_utils`` API, so cells read as one line each.
  * ``status()`` guards each subsystem independently -- a partial bring-up still
    reports what it can.
  * ``shutdown()`` / ``atexit`` release the instrument (camera, wheel, rotator)
    and the serial port. It deliberately does **not** park, disable, or
    disconnect the *mount* -- mount teardown is an explicit, separate action.

Typical lab use (INDIGO alpaca agent + serial HWP, no telescope)::

    from obs_utils import interactive as obs
    s = obs.connect(alpaca=True, pyxis_serial=True)   # cell 1, run once
    s.status()                                         # cell 2
    s.filter(2)                                        # cell 3
    s.filter(3)                                        # cell 4 -- no reconnect
    s.home_hwp(); s.hwp(90)                             # cell 5
    img = s.expose(1.0, dark=True)                     # cell 6
    obs.shutdown()                                      # last cell
"""

import atexit
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np

from . import user_config as _uc
from .alpaca import ImagingSession
from .config import AlpacaConfig, Pwi4Config, PyxisSerialConfig
from .imaging import (
    CaptureRequest,
    capture_fits_file,
    capture_image_array,
    select_filter,
    select_hwp_angle,
)
from .pwi4_client import PWI4
from .waits import wait_for_value, wait_until

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# Helpers for centring (see ObservatorySession.center_on_pair)
# --------------------------------------------------------------------------- #
def _default_pol_config():
    """The session's PolConfig, built from the current capture context."""
    from .pol_config import session_pol_config_to_polconfig
    from .session_context import SessionCaptureContext

    return session_pol_config_to_polconfig(SessionCaptureContext())


def _pixel_scale_arcsec(cfg: Any) -> Optional[float]:
    """Pixel scale [arcsec/px] from a PolConfig, or ``None`` if it does not carry one."""
    for owner in (cfg, getattr(cfg, "detector", None), getattr(cfg, "telescope", None)):
        scale = getattr(owner, "pixel_scale_arcsec", None)
        if isinstance(scale, (int, float)) and scale > 0:
            return float(scale)
    return None


def _detector_to_sky(
    dx_px: float,
    dy_px: float,
    scale_arcsec: float,
    sky_pa_deg: Optional[float],
    parity: int,
) -> Tuple[Optional[float], Optional[float]]:
    """Rotate a detector pixel offset into (east, north) arcsec.

    Returns ``(None, None)`` when ``sky_pa_deg`` is not supplied. That is the
    honest answer: how the detector is clocked relative to the sky has never been
    measured on this instrument, and a slew computed from an assumed orientation
    moves the target further away just as confidently as it would move it closer.
    Establish the mapping on the first pointed night -- offset a known amount,
    see which way the star went -- and pass it here.
    """
    if sky_pa_deg is None:
        return None, None
    pa = np.deg2rad(float(sky_pa_deg))
    x = float(dx_px) * (1 if int(parity) >= 0 else -1)
    east = (x * np.cos(pa) + float(dy_px) * np.sin(pa)) * scale_arcsec
    north = (-x * np.sin(pa) + float(dy_px) * np.cos(pa)) * scale_arcsec
    return float(east), float(north)


# --------------------------------------------------------------------------- #
# Session object
# --------------------------------------------------------------------------- #
@dataclass
class ObservatorySession:
    """Live bundle of the interactive connections.

    Any field may be ``None`` when its subsystem was not requested. Prefer the
    module-level :func:`connect` / :func:`session` helpers over building this
    directly, so the singleton and cleanup registration stay consistent.
    """

    imaging: Optional[ImagingSession] = None
    pwi4: Optional[PWI4] = None
    pyxis: Optional[Any] = None  # obs_utils.pyxis_gen3.PyxisGen3 (lazy import)
    alpaca_config: Optional[AlpacaConfig] = None
    pwi4_config: Optional[Pwi4Config] = None
    pyxis_config: Optional[PyxisSerialConfig] = None

    # ---- convenience accessors ------------------------------------------- #
    @property
    def camera(self):
        return None if self.imaging is None else self.imaging.camera

    @property
    def filter_wheel(self):
        return None if self.imaging is None else self.imaging.filter_wheel

    @property
    def hwp_rotator(self):
        """Alpaca (observatory) **half-wave plate** rotator, if wired via INDIGO/ASCOM.

        Named in full because this project has two rotators and confusing them
        is a data-losing mistake. This one is the Optec Pyxis: it modulates
        polarization and its angle is the independent variable of a pol
        sequence. The other is the PWI4 **field rotator**
        (:attr:`field_rotator`), which holds sky position angle and has nothing
        to do with polarimetry. ``self.imaging.rotator`` keeps the bare ASCOM
        name because at the driver layer the only Alpaca rotator here is the HWP.
        """
        return None if self.imaging is None else self.imaging.rotator

    @property
    def focuser(self):
        """PWI4 focuser status section, or ``None`` when no PWI4 client is up."""
        return self._pwi4_section("focuser")

    @property
    def field_rotator(self):
        """PWI4 **field rotator** status section (the instrument de-rotator).

        Not the half-wave plate -- see :attr:`hwp_rotator`.
        """
        return self._pwi4_section("rotator")

    def _pwi4_section(self, name: str):
        if self.pwi4 is None:
            return None
        try:
            return getattr(self.pwi4.status(), name)
        except Exception as exc:  # pragma: no cover - hardware/network dependent
            logger.warning("PWI4 %s status unavailable: %r", name, exc)
            return None

    def _require_pwi4(self) -> PWI4:
        if self.pwi4 is None:
            raise RuntimeError(
                "No PWI4 client. Run connect_mount() (or connect_all()) first."
            )
        return self.pwi4

    def _require_imaging(self) -> ImagingSession:
        if self.imaging is None:
            raise RuntimeError(
                "No Alpaca instrument connected. Run connect(alpaca=True) first."
            )
        return self.imaging

    # ---- filter wheel ----------------------------------------------------- #
    def filter(
        self,
        position: Union[int, str],
        poll_s: float = 0.5,
        timeout_s: float = 30.0,
    ) -> int:
        """Move the ZWO EFW to ``position`` (0-based slot index or filter name).

        Blocks until the wheel reports it has landed, then returns the slot.
        """
        session = self._require_imaging()
        if session.filter_wheel is None:
            raise RuntimeError("No filter wheel connected")
        landed = select_filter(session, position, poll_s=poll_s, timeout_s=timeout_s)
        logger.info("Filter -> slot %d (%s)", landed, self._filter_name(landed))
        return landed

    def _filter_name(self, slot: int) -> str:
        names = self._filter_names()
        return names[slot] if 0 <= slot < len(names) else str(slot)

    def _filter_names(self) -> List[str]:
        session = self.imaging
        if session is None:
            return []
        if session.filter_names:
            return list(session.filter_names)
        try:
            return list(session.filter_wheel.Names)  # type: ignore[union-attr]
        except Exception:
            return []

    def current_filter(self) -> Tuple[int, str]:
        """Return ``(slot, name)`` for the wheel's current position."""
        session = self._require_imaging()
        if session.filter_wheel is None:
            raise RuntimeError("No filter wheel connected")
        slot = int(session.filter_wheel.Position)
        return slot, self._filter_name(slot)

    # ---- half-wave plate -------------------------------------------------- #
    def hwp(
        self,
        angle_deg: float,
        poll_s: float = 0.5,
        timeout_s: float = 120.0,
    ) -> float:
        """Rotate the half-wave plate to an absolute angle [deg].

        Uses the native-serial Pyxis Gen3 when present (lab), otherwise the
        Alpaca rotator (observatory). Blocks until settled; returns the reported
        angle.
        """
        if self.pyxis is not None:
            final = self.pyxis.move_absolute(
                float(angle_deg), poll_s=poll_s, timeout_s=timeout_s
            )
            logger.info("HWP (serial) -> %.3f deg", final)
            return final
        session = self._require_imaging()
        if session.rotator is None:
            raise RuntimeError(
                "No HWP available: neither a serial Pyxis nor an Alpaca rotator "
                "is connected."
            )
        final = select_hwp_angle(session, float(angle_deg), poll_s=poll_s, timeout_s=timeout_s)
        logger.info("HWP (alpaca) -> %.3f deg", final)
        return final

    def home_hwp(self, poll_s: float = 0.5, timeout_s: float = 180.0) -> float:
        """Home the serial Pyxis Gen3 HWP (physical motion). Serial path only.

        The Gen3 rejects MOVEPA until it has been homed, so run this once per
        power cycle before :meth:`hwp`.
        """
        if self.pyxis is None:
            raise RuntimeError(
                "home_hwp() is for the serial Pyxis Gen3 only. The Alpaca rotator "
                "homes through its own driver."
            )
        final = self.pyxis.home(poll_s=poll_s, timeout_s=timeout_s)
        logger.info("HWP homed; Current PA = %.3f deg", final)
        return final

    # ---- focuser (PWI4) --------------------------------------------------- #
    def focus(self, position: float, *, timeout_s: float = 120.0, tol: float = 2.0) -> float:
        """Move the focuser to an absolute position [steps]; return where it landed."""
        pwi4 = self._require_pwi4()
        logger.info("Focuser -> %.1f", float(position))
        pwi4.focuser_goto(float(position))
        return wait_for_value(
            lambda: float(pwi4.status().focuser.position),
            float(position),
            tol=tol,
            timeout_s=timeout_s,
            what="focuser move",
            detail="Check the focuser is enabled and not at a travel limit.",
        )

    def focus_relative(self, delta: float, **kwargs: Any) -> float:
        """Jog the focuser by ``delta`` steps from where it is now."""
        pwi4 = self._require_pwi4()
        return self.focus(float(pwi4.status().focuser.position) + float(delta), **kwargs)

    def focus_stop(self) -> None:
        """Stop focuser motion immediately."""
        self._require_pwi4().focuser_stop()
        logger.info("Focuser stop commanded")

    def focuser_status(self) -> Dict[str, Any]:
        section = self.focuser
        if section is None:
            return {"connected": None}
        return {
            "exists": getattr(section, "exists", None),
            "connected": getattr(section, "is_connected", None),
            "enabled": getattr(section, "is_enabled", None),
            "position": getattr(section, "position", None),
            "moving": getattr(section, "is_moving", None),
        }

    def focus_sweep(
        self,
        positions: Sequence[float],
        exposure_s: float,
        *,
        aperture_px: Optional[float] = None,
        settle_s: float = 1.0,
        **capture_kwargs: Any,
    ) -> List[Tuple[float, Any]]:
        """Step the focuser through ``positions``, measuring one star at each.

        Returns ``[(position, StarProfile), ...]`` -- plot it with
        :func:`obs_utils.live.focus_curve` and **read the minimum yourself**.
        Nothing here picks a best focus or moves to one: the frames are cheap,
        the judgement is not, and an autofocus that quietly lands on a cosmic
        ray costs a whole sequence.

        The focuser is left wherever the last step put it.
        """
        from . import focus as focus_metrics

        pwi4 = self._require_pwi4()
        start = float(pwi4.status().focuser.position)
        logger.info(
            "Focus sweep: %d points %.0f..%.0f (from %.0f)",
            len(positions), min(positions), max(positions), start,
        )
        aperture = focus_metrics.DEFAULT_APERTURE_PX if aperture_px is None else aperture_px

        results: List[Tuple[float, Any]] = []
        for position in positions:
            self.focus(float(position))
            if settle_s:
                time.sleep(settle_s)
            frame = self.expose(exposure_s, **capture_kwargs)
            profile = focus_metrics.star_profile(frame, aperture_px=aperture)
            logger.info("  %8.1f  %s", float(position), profile.line())
            results.append((float(position), profile))
        return results

    # ---- field rotator (PWI4) --------------------------------------------- #
    # The instrument de-rotator. NOT the half-wave plate -- see hwp() above.
    def field_rotator_goto_field(self, angle_deg: float, **kwargs: Any) -> float:
        """Slew the field rotator to a sky position angle [deg]."""
        pwi4 = self._require_pwi4()
        logger.info("Field rotator -> field angle %.3f deg", float(angle_deg))
        pwi4.rotator_goto_field(float(angle_deg))
        return self._wait_field_rotator(
            lambda st: float(st.rotator.field_angle_degs), float(angle_deg), **kwargs
        )

    def field_rotator_goto_mech(self, angle_deg: float, **kwargs: Any) -> float:
        """Slew the field rotator to a mechanical angle [deg]."""
        pwi4 = self._require_pwi4()
        logger.info("Field rotator -> mech angle %.3f deg", float(angle_deg))
        pwi4.rotator_goto_mech(float(angle_deg))
        return self._wait_field_rotator(
            lambda st: float(st.rotator.mech_position_degs), float(angle_deg), **kwargs
        )

    def field_rotator_offset(self, delta_deg: float, **kwargs: Any) -> float:
        """Jog the field rotator by ``delta_deg`` from where it is now."""
        pwi4 = self._require_pwi4()
        logger.info("Field rotator jog %+.3f deg", float(delta_deg))
        target = float(pwi4.status().rotator.mech_position_degs) + float(delta_deg)
        pwi4.rotator_offset(float(delta_deg))
        return self._wait_field_rotator(
            lambda st: float(st.rotator.mech_position_degs), target, **kwargs
        )

    def field_rotator_stop(self) -> None:
        """Stop field-rotator motion immediately."""
        self._require_pwi4().rotator_stop()
        logger.info("Field rotator stop commanded")

    def _wait_field_rotator(
        self, read, target: float, *, timeout_s: float = 180.0, tol: float = 0.05
    ) -> float:
        pwi4 = self._require_pwi4()
        return wait_for_value(
            lambda: read(pwi4.status()),
            target,
            tol=tol,
            timeout_s=timeout_s,
            what="field rotator move",
            detail="Check the rotator is enabled and clear of its travel limit.",
        )

    def field_rotator_status(self) -> Dict[str, Any]:
        section = self.field_rotator
        if section is None:
            return {"connected": None}
        return {
            "exists": getattr(section, "exists", None),
            "connected": getattr(section, "is_connected", None),
            "enabled": getattr(section, "is_enabled", None),
            "mech_deg": getattr(section, "mech_position_degs", None),
            "field_deg": getattr(section, "field_angle_degs", None),
            "moving": getattr(section, "is_moving", None),
        }

    # ---- camera ----------------------------------------------------------- #
    def expose(
        self,
        exposure_s: float,
        out_path: Optional[Union[str, Path]] = None,
        *,
        dark: bool = False,
        header: Optional[Any] = None,
        poll_s: float = 0.5,
        timeout_s: float = 300.0,
        **capture_kwargs: Any,
    ) -> Union[np.ndarray, Path]:
        """Take one frame.

        Returns a NumPy array when ``out_path`` is omitted, otherwise writes a
        FITS file and returns its :class:`~pathlib.Path`. ``dark=True`` takes a
        shutter-closed frame. Extra keyword args (``gain``, ``offset``, ``binx``,
        ``numx`` ...) pass straight through to :class:`CaptureRequest`.
        """
        session = self._require_imaging()
        request = CaptureRequest(
            exposure_s=float(exposure_s),
            is_light=not dark,
            **capture_kwargs,
        )
        if out_path is None:
            data, _dtype = capture_image_array(
                session, request, poll_s=poll_s, timeout_s=timeout_s
            )
            logger.info(
                "Captured %s frame %.3fs -> array %s",
                "DARK" if dark else "LIGHT",
                exposure_s,
                data.shape,
            )
            return data

        if header is None:
            from alpyca_tools.fits_writer import FitsHeaderConfig

            header = FitsHeaderConfig(
                imagetyp="DARK" if dark else "LIGHT",
                instrument="QHY268M",
            )
        path = capture_fits_file(
            session, request, header, Path(out_path), poll_s=poll_s, timeout_s=timeout_s
        )
        logger.info("Wrote %s", path)
        return path

    # ---- cooler ----------------------------------------------------------- #
    def cool_to(
        self, setpoint_c: float, *, tol_c: float = 1.0, timeout_s: float = 1800.0
    ) -> float:
        """Set the cooler setpoint and wait for the sensor to reach it."""
        cam = self.camera
        if cam is None:
            raise RuntimeError("No camera connected")
        cam.CoolerOn = True
        cam.SetCCDTemperature = float(setpoint_c)
        logger.info("Cooler setpoint %.1f C; waiting for the sensor ...", float(setpoint_c))
        return wait_for_value(
            lambda: float(cam.CCDTemperature),
            float(setpoint_c),
            tol=tol_c,
            timeout_s=timeout_s,
            what="cooler settle",
            detail="Check the ambient load and that the setpoint is reachable.",
        )

    def warm_up(
        self,
        *,
        target_c: Optional[float] = None,
        rate_c_per_min: float = 2.0,
        step_c: float = 2.0,
        timeout_s: float = 3600.0,
    ) -> float:
        """Ramp the sensor to ambient in steps, then switch the cooler off.

        **CONJECTURED operational precaution:** a controlled warm-up is intended
        to reduce thermal-shock and condensation risk. This project has not yet
        verified a QHY manufacturer rate recommendation, so the default rate is
        adjustable guidance, not an instrument specification.

        ``target_c`` defaults to the camera's reported ambient (``HeatSinkTemperature``),
        falling back to +15 C if the driver does not report one. Returns the final
        sensor temperature.
        """
        cam = self.camera
        if cam is None:
            raise RuntimeError("No camera connected")

        if target_c is None:
            try:
                target_c = float(cam.HeatSinkTemperature)
            except Exception:
                target_c = 15.0
                logger.info("No ambient reading; ramping to %.1f C", target_c)

        current = float(cam.CCDTemperature)
        step_wait_s = max(1.0, 60.0 * abs(step_c) / max(rate_c_per_min, 0.1))
        logger.info(
            "Warm-up ramp: %.1f -> %.1f C in %.1f C steps at ~%.1f C/min",
            current, float(target_c), abs(step_c), rate_c_per_min,
        )

        deadline = time.monotonic() + timeout_s
        while current < float(target_c) - abs(step_c):
            current = min(current + abs(step_c), float(target_c))
            cam.SetCCDTemperature = current
            logger.info("  setpoint %.1f C", current)
            time.sleep(step_wait_s)
            if time.monotonic() >= deadline:
                logger.warning("Warm-up ramp hit its %.0f s budget; stopping the ramp", timeout_s)
                break
            current = float(cam.CCDTemperature)

        cam.CoolerOn = False
        final = float(cam.CCDTemperature)
        logger.info("Cooler off at %.1f C", final)
        return final

    # ---- centring (no astrometry -- see TEMPLATE-OVERHAUL-PLAN.md section 0) - #
    def center_on_pair(
        self,
        frame: Optional[np.ndarray] = None,
        *,
        exposure_s: float = 5.0,
        target_xy: Optional[Tuple[float, float]] = None,
        pixel_scale_arcsec: Optional[float] = None,
        pol_config: Optional[Any] = None,
        fwhm_px: float = 5.0,
        threshold_sigma: float = 5.0,
        tol_px: float = 4.0,
        sky_pa_deg: Optional[float] = None,
        parity: int = 1,
        apply: bool = False,
    ) -> Dict[str, Any]:
        """Centre the brightest Savart beam **pair** on ``target_xy``. No plate solve.

        The operational need at the telescope is "put my target in the middle",
        not "give me an absolute WCS". Our plate solver (``ps3cli``) takes an
        image file and nothing else, so the usual dual-beam fix -- detect
        sources, drop one beam of each pair, solve the deduped list -- is not
        reachable through it, and it has never been tried on a Savart-doubled
        field. This path sidesteps the question entirely: the Savart offset is
        already a characterized quantity, so the pair itself identifies the star.

        **Fail-closed on uncharacterized geometry.** At the nominal ~239 px
        separation a wrong pairing does not fail loudly -- it succeeds and looks
        right -- so this refuses to run on a beam geometry nobody has measured,
        exactly as ``poltools.reduce_to_stokes(detect=True)`` does.

        Reports the offset by default and only slews when ``apply=True``.

        ``sky_pa_deg`` and ``parity`` give the detector-to-sky mapping: the sky
        position angle of the detector's +y axis, and +1 or -1 for whether the
        field is mirrored. **Neither is calibrated on this instrument.** Without
        them the offset is reported in pixels and no slew is possible -- see the
        note at the conversion below.
        """
        from poltools.photometry import detect_sources, pair_oe

        cfg = pol_config if pol_config is not None else _default_pol_config()
        beam = getattr(cfg, "beam", cfg)
        if not getattr(cfg, "beam_geometry_characterized", False):
            raise RuntimeError(
                "Beam geometry is not characterized, so a 'pair' found at the "
                "nominal separation would be a guess that looks like a "
                "measurement.\nFIX: measure the separation on a real frame and "
                "attach it with PolConfig.with_beam_geometry(), or pass "
                "pol_config= explicitly."
            )

        image = self.expose(exposure_s) if frame is None else np.asarray(frame)
        if target_xy is None:
            target_xy = (image.shape[1] / 2.0, image.shape[0] / 2.0)

        pairs = pair_oe(
            detect_sources(image, fwhm_px, threshold_sigma=threshold_sigma),
            beam,
            tol_px=tol_px,
        )
        if not pairs:
            raise RuntimeError(
                f"No beam pair found at separation {beam.separation_px:.1f} px "
                f"(tol {tol_px} px). Either nothing is in the field, the exposure "
                "is too short, or the geometry is wrong."
            )

        (xo, yo), (xe, ye) = pairs[0]
        mid_x, mid_y = 0.5 * (xo + xe), 0.5 * (yo + ye)
        dx_px, dy_px = target_xy[0] - mid_x, target_xy[1] - mid_y

        result: Dict[str, Any] = {
            "pairs_found": len(pairs),
            "ordinary_xy": (xo, yo),
            "extraordinary_xy": (xe, ye),
            "midpoint_xy": (mid_x, mid_y),
            "target_xy": tuple(target_xy),
            "offset_px": (dx_px, dy_px),
            "applied": False,
        }

        scale = pixel_scale_arcsec if pixel_scale_arcsec is not None else _pixel_scale_arcsec(cfg)
        if scale is None:
            logger.warning("No pixel scale available; reporting the offset in pixels only.")
            return result

        # Detector pixels -> sky. The rotation between the two is NOT a known
        # quantity on this instrument: it depends on how the camera is clocked in
        # its adapter and on the field rotator angle, and nothing has measured it.
        # Guessing it would slew the wrong way with total confidence, so the sky
        # offset is only computed when the caller states the mapping.
        east_arcsec, north_arcsec = _detector_to_sky(dx_px, dy_px, scale, sky_pa_deg, parity)
        if east_arcsec is None:
            result["needs_sky_orientation"] = True
            logger.info(
                "Offset to centre: %+.1f, %+.1f px (|%.1f| arcsec). Sky direction "
                "unknown -- pass sky_pa_deg= and parity= to convert, or nudge with "
                "mount_offset by hand and watch which way the star moves.",
                dx_px, dy_px, float(np.hypot(dx_px, dy_px)) * scale,
            )
            return result

        result["offset_arcsec_east_north"] = (east_arcsec, north_arcsec)
        if not apply:
            logger.info(
                "Offset to centre: %+.1f, %+.1f px -> E %+.1f, N %+.1f arcsec. "
                "Re-run with apply=True to slew.",
                dx_px, dy_px, east_arcsec, north_arcsec,
            )
            return result

        pwi4 = self._require_pwi4()
        pwi4.mount_offset(ra_add_arcsec=east_arcsec, dec_add_arcsec=north_arcsec)
        result["applied"] = True
        logger.info("Applied mount offset E %+.1f, N %+.1f arcsec", east_arcsec, north_arcsec)
        return result

    # ---- status ----------------------------------------------------------- #
    def status(self, pretty: bool = True) -> Dict[str, Any]:
        """Collect a health summary. Each subsystem is probed independently so a
        partial bring-up still reports what it can.
        """
        info: Dict[str, Any] = {
            "instrument": self._instrument_status(),
            "hwp": self._hwp_status(),
            "mount": self._mount_status(),
            "focuser": self.focuser_status(),
            "field_rotator": self.field_rotator_status(),
        }
        if pretty:
            _print_status(info)
        return info

    def _instrument_status(self) -> Dict[str, Any]:
        out: Dict[str, Any] = {"connected": self.imaging is not None}
        if self.imaging is None:
            return out
        cam = self.imaging.camera
        try:
            out["camera_connected"] = bool(cam.Connected)
            out["sensor"] = (
                getattr(cam, "CameraXSize", None),
                getattr(cam, "CameraYSize", None),
            )
        except Exception as exc:  # pragma: no cover - hardware/network dependent
            out["camera_error"] = repr(exc)
        if self.imaging.filter_wheel is not None:
            try:
                slot = int(self.imaging.filter_wheel.Position)
                out["filter"] = {"slot": slot, "name": self._filter_name(slot)}
            except Exception as exc:  # pragma: no cover
                out["filter_error"] = repr(exc)
        return out

    def _hwp_status(self) -> Dict[str, Any]:
        if self.pyxis is not None:
            out: Dict[str, Any] = {"backend": "serial"}
            try:
                st = self.pyxis.get_status()
                out["current_pa_deg"] = self.pyxis.position_deg
                out["homed"] = self.pyxis._flag(st, "Is Homed")  # noqa: SLF001
                out["moving"] = self.pyxis._flag(st, "Is Moving")  # noqa: SLF001
            except Exception as exc:  # pragma: no cover
                out["error"] = repr(exc)
            return out
        rot = self.hwp_rotator
        if rot is not None:
            out = {"backend": "alpaca"}
            try:
                out["position_deg"] = float(rot.Position)
                out["moving"] = bool(getattr(rot, "IsMoving", False))
            except Exception as exc:  # pragma: no cover
                out["error"] = repr(exc)
            return out
        return {"backend": None}

    def _mount_status(self) -> Dict[str, Any]:
        if self.pwi4 is None:
            return {"connected": None}
        try:
            st = self.pwi4.status()
            mount = getattr(st, "mount", None)
            return {
                "connected": bool(getattr(mount, "is_connected", False)),
                "slewing": bool(getattr(mount, "is_slewing", False)),
                "tracking": bool(getattr(mount, "is_tracking", False)),
                "ra_hours": getattr(mount, "ra_j2000_hours", None),
                "dec_deg": getattr(mount, "dec_j2000_degs", None),
            }
        except Exception as exc:  # pragma: no cover
            return {"error": repr(exc)}

    # ---- teardown --------------------------------------------------------- #
    def close(self) -> None:
        """Release the instrument and the serial HWP.

        Does NOT touch the mount: parking/disabling the PlaneWave is a deliberate
        action, not a side effect of clearing a notebook session.
        """
        if self.imaging is not None:
            try:
                self.imaging.close()
                logger.info("Instrument disconnected (camera/wheel/rotator)")
            except Exception:  # pragma: no cover
                logger.exception("Error closing Alpaca instrument")
            self.imaging = None
        if self.pyxis is not None:
            try:
                self.pyxis.close()
                logger.info("Pyxis serial port closed")
            except Exception:  # pragma: no cover
                logger.exception("Error closing Pyxis serial port")
            self.pyxis = None
        # PWI4 is a stateless HTTP client; just drop the handle (mount untouched).
        self.pwi4 = None


# --------------------------------------------------------------------------- #
# Module-level singleton + lifecycle
# --------------------------------------------------------------------------- #
_SESSION: Optional[ObservatorySession] = None


def connect(
    *,
    alpaca: bool = True,
    mount: bool = False,
    pyxis_serial: bool = False,
    alpaca_config: Optional[AlpacaConfig] = None,
    pwi4_config: Optional[Pwi4Config] = None,
    pyxis_config: Optional[PyxisSerialConfig] = None,
    reuse: bool = True,
) -> ObservatorySession:
    """Connect the requested subsystems and return the live session.

    Idempotent: with ``reuse=True`` (default) an existing session is kept and
    only the *not-yet-connected* subsystems are added, so re-running the cell
    will not double-open the exclusive Pyxis serial port or leak Alpaca
    sessions. Pass ``reuse=False`` to tear the current session down and rebuild.

    Parameters mirror the site layout:
      * ``alpaca``       -- QHY268M camera + ZWO EFW (+ Alpaca HWP rotator if the
                            config sets ``rotator_index``), via the INDIGO alpaca
                            agent. Defaults to :data:`user_config.ALPACA_CONFIG`.
      * ``mount``        -- create the PWI4 HTTP client (no motion; the mount is
                            not slewed, enabled, or homed here).
      * ``pyxis_serial`` -- native-serial Pyxis Gen3 HWP (lab bench). Defaults to
                            :data:`user_config.PYXIS_CONFIG`.
    """
    global _SESSION

    if _SESSION is not None and not reuse:
        _SESSION.close()
        _SESSION = None

    session = _SESSION if _SESSION is not None else ObservatorySession()

    if alpaca and session.imaging is None:
        cfg = alpaca_config or _uc.ALPACA_CONFIG
        from .imaging import open_session  # local import keeps the surface tidy

        session.imaging = open_session(cfg)
        session.alpaca_config = cfg
        logger.info(
            "Alpaca instrument connected on %s (camera=%s wheel=%s rotator=%s)",
            cfg.host,
            cfg.camera_index,
            cfg.filterwheel_index,
            cfg.rotator_index,
        )

    if mount and session.pwi4 is None:
        cfg = pwi4_config or _uc.PWI4_CONFIG
        session.pwi4 = PWI4(host=cfg.host, port=cfg.port)
        session.pwi4_config = cfg
        logger.info("PWI4 client ready at %s:%d (mount not commanded)", cfg.host, cfg.port)

    if pyxis_serial and session.pyxis is None:
        cfg = pyxis_config or _uc.PYXIS_CONFIG
        # Lazy import so obs_utils.interactive imports even without pyserial.
        from .pyxis_gen3 import connect_pyxis_gen3

        session.pyxis = connect_pyxis_gen3(
            cfg.port, baud=cfg.baud, autodetect=cfg.autodetect_baud
        )
        session.pyxis_config = cfg
        logger.info("Pyxis Gen3 HWP connected on %s", cfg.port)

    _SESSION = session
    return session


# --------------------------------------------------------------------------- #
# Per-component connect (fault-isolated)
# --------------------------------------------------------------------------- #
# ``connect()`` brings up the whole Alpaca instrument in one call: if the camera
# is offline you get nothing, even when the wheel and rotator are fine. The
# helpers below connect exactly one device each and attach it to the shared
# singleton session, so an outage in one component never blocks the others.
# Each raises on its own failure -- run one per notebook cell and a failure is
# naturally isolated to that cell while the already-connected devices keep
# working. All are idempotent with ``reuse=True``.
def _ensure_session() -> ObservatorySession:
    global _SESSION
    if _SESSION is None:
        _SESSION = ObservatorySession()
    return _SESSION


def _ensure_imaging(session: ObservatorySession, cfg: AlpacaConfig) -> ImagingSession:
    if session.imaging is None:
        session.imaging = ImagingSession()
    if session.alpaca_config is None:
        session.alpaca_config = cfg
    return session.imaging


def connect_camera(
    *, alpaca_config: Optional[AlpacaConfig] = None, reuse: bool = True
) -> ObservatorySession:
    """Connect only the QHY268M camera and attach it to the session."""
    from .alpaca import connect_camera as _connect_camera

    session = _ensure_session()
    cfg = alpaca_config or _uc.ALPACA_CONFIG
    imaging = _ensure_imaging(session, cfg)
    if not (reuse and imaging.camera is not None):
        imaging.camera = _connect_camera(cfg.camera_host or cfg.host, cfg.camera_index)
        logger.info(
            "Camera connected on %s (device=%d)",
            cfg.camera_host or cfg.host,
            cfg.camera_index,
        )
    _SESSION = session
    return session


def connect_filter_wheel(
    *, alpaca_config: Optional[AlpacaConfig] = None, reuse: bool = True
) -> ObservatorySession:
    """Connect only the ZWO EFW filter wheel and attach it to the session."""
    from .alpaca import connect_filter_wheel as _connect_wheel

    session = _ensure_session()
    cfg = alpaca_config or _uc.ALPACA_CONFIG
    if cfg.filterwheel_index is None:
        raise RuntimeError("AlpacaConfig.filterwheel_index is None -- no wheel configured")
    imaging = _ensure_imaging(session, cfg)
    imaging.filter_names = imaging.filter_names or cfg.filter_names
    if not (reuse and imaging.filter_wheel is not None):
        imaging.filter_wheel = _connect_wheel(cfg.host, cfg.filterwheel_index)
        logger.info("Filter wheel connected on %s (device=%d)", cfg.host, cfg.filterwheel_index)
    _SESSION = session
    return session


def connect_hwp(
    *, alpaca_config: Optional[AlpacaConfig] = None, reuse: bool = True
) -> ObservatorySession:
    """Connect only the Alpaca **half-wave plate** rotator (observatory path).

    Not the field rotator -- that is :func:`connect_field_rotator`, on PWI4.

    For the lab bench's native-serial Pyxis Gen3, use :func:`connect_hwp_serial`
    instead: the serial port is exclusive and homes through its own driver.
    """
    from .alpaca import connect_rotator as _connect_alpaca_rotator

    session = _ensure_session()
    cfg = alpaca_config or _uc.ALPACA_CONFIG
    if cfg.rotator_index is None:
        raise RuntimeError(
            "AlpacaConfig.rotator_index is None -- no Alpaca HWP rotator configured. "
            "For the lab serial Pyxis use connect_hwp_serial()."
        )
    imaging = _ensure_imaging(session, cfg)
    if not (reuse and imaging.rotator is not None):
        imaging.rotator = _connect_alpaca_rotator(cfg.host, cfg.rotator_index)
        logger.info("Alpaca HWP rotator connected on %s (device=%d)", cfg.host, cfg.rotator_index)
    _SESSION = session
    return session


def connect_focuser(
    *, pwi4_config: Optional[Pwi4Config] = None, timeout_s: Optional[float] = None
) -> ObservatorySession:
    """Connect and enable the PWI4 focuser (no motion). Bounded in time."""
    from .startup import FOCUSER_TIMEOUT_S, connect_focuser as _connect_focuser

    session = connect_mount(pwi4_config=pwi4_config)
    fitted = _connect_focuser(
        session.pwi4, timeout_s=FOCUSER_TIMEOUT_S if timeout_s is None else timeout_s
    )
    logger.info("Focuser %s", "connected and enabled" if fitted else "not fitted")
    return session


def connect_field_rotator(
    *, pwi4_config: Optional[Pwi4Config] = None, timeout_s: Optional[float] = None
) -> ObservatorySession:
    """Connect and enable the PWI4 **field rotator** (no motion). Bounded in time.

    The instrument de-rotator, not the half-wave plate -- see :func:`connect_hwp`.
    """
    from .startup import FIELD_ROTATOR_TIMEOUT_S, connect_field_rotator as _connect_fr

    session = connect_mount(pwi4_config=pwi4_config)
    fitted = _connect_fr(
        session.pwi4, timeout_s=FIELD_ROTATOR_TIMEOUT_S if timeout_s is None else timeout_s
    )
    logger.info("Field rotator %s", "connected and enabled" if fitted else "not fitted")
    return session


def connect_all(
    *,
    alpaca: bool = True,
    mount: bool = True,
    focuser: bool = True,
    field_rotator: bool = True,
    hwp_serial: bool = False,
    alpaca_config: Optional[AlpacaConfig] = None,
    pwi4_config: Optional[Pwi4Config] = None,
    pyxis_config: Optional[PyxisSerialConfig] = None,
) -> ObservatorySession:
    """Bring up every configured device, one at a time, and report what came up.

    Fault-isolated: each device is attempted independently and a failure is
    logged, not raised, so one dead component cannot cost you the rest of the
    assembly. Check :meth:`ObservatorySession.status` afterwards -- this returns
    a session, not a promise that everything in it is live.

    **This never moves anything.** It connects and energizes only. That is the
    difference from :func:`obs_utils.startup.startup_observatory`, which also
    homes the mount and loads a pointing model; homing is physical motion across
    both axes and stays an explicit, separate step.

    ``hwp_serial`` selects the lab bench's native-serial Pyxis over the
    observatory's Alpaca HWP. The two are mutually exclusive: the serial port is
    exclusive, and the Alpaca driver would be talking to the same device.
    """
    session = _ensure_session()
    attempts = []
    if alpaca:
        attempts.append(("camera", lambda: connect_camera(alpaca_config=alpaca_config)))
        attempts.append(("filter wheel", lambda: connect_filter_wheel(alpaca_config=alpaca_config)))
        if not hwp_serial:
            attempts.append(("hwp (alpaca)", lambda: connect_hwp(alpaca_config=alpaca_config)))
    if hwp_serial:
        attempts.append(("hwp (serial)", lambda: connect_hwp_serial(pyxis_config=pyxis_config)))
    if mount:
        attempts.append(("mount (pwi4 client)", lambda: connect_mount(pwi4_config=pwi4_config)))
    if focuser:
        attempts.append(("focuser", lambda: connect_focuser(pwi4_config=pwi4_config)))
    if field_rotator:
        attempts.append(("field rotator", lambda: connect_field_rotator(pwi4_config=pwi4_config)))

    up, down = [], []
    for name, action in attempts:
        try:
            session = action()
            up.append(name)
        except Exception as exc:
            down.append(name)
            logger.warning("%s did NOT connect: %r", name, exc)

    print("connected :", ", ".join(up) if up else "(nothing)")
    if down:
        print("FAILED    :", ", ".join(down))
        print("Each device is independent -- the connected ones above are usable.")
    return session


def connect_hwp_serial(
    *, pyxis_config: Optional[PyxisSerialConfig] = None, reuse: bool = True
) -> ObservatorySession:
    """Connect only the native-serial Pyxis Gen3 HWP (lab bench)."""
    return connect(alpaca=False, pyxis_serial=True, pyxis_config=pyxis_config, reuse=reuse)


def connect_mount(
    *, pwi4_config: Optional[Pwi4Config] = None, reuse: bool = True
) -> ObservatorySession:
    """Create only the PWI4 client (no mount motion, enabling, or homing)."""
    return connect(alpaca=False, mount=True, pwi4_config=pwi4_config, reuse=reuse)


def session() -> ObservatorySession:
    """Return the current live session, or raise if nothing is connected."""
    if _SESSION is None:
        raise RuntimeError("No active session. Call obs_utils.interactive.connect() first.")
    return _SESSION


def current() -> Optional[ObservatorySession]:
    """Return the current session or ``None`` (non-raising)."""
    return _SESSION


def shutdown() -> None:
    """Release the instrument + serial HWP and clear the singleton (mount untouched)."""
    global _SESSION
    if _SESSION is not None:
        _SESSION.close()
        _SESSION = None
        logger.info("Interactive session shut down")


def adopt_startup(state: Any) -> ObservatorySession:
    """Wrap a :class:`obs_utils.startup.StartupState` from ``startup_observatory``
    as the interactive singleton, so the observatory notebook can run the full
    mount bring-up and then use the convenience methods here.
    """
    global _SESSION
    if _SESSION is not None:
        _SESSION.close()
    _SESSION = ObservatorySession(
        imaging=getattr(state, "imaging", None),
        pwi4=getattr(state, "pwi4", None),
    )
    return _SESSION


# Best-effort cleanup when the kernel exits. Notebooks do not guarantee this
# runs on a hard kernel kill, so an explicit shutdown() cell is still preferred.
atexit.register(shutdown)


# --------------------------------------------------------------------------- #
# Pretty-printer
# --------------------------------------------------------------------------- #
def _print_status(info: Dict[str, Any]) -> None:
    inst = info["instrument"]
    hwp = info["hwp"]
    mount = info["mount"]

    print("POLITE interactive session")
    print("-" * 52)

    if not inst.get("connected"):
        print("instrument : (not connected)")
    else:
        sensor = inst.get("sensor")
        cam = "connected" if inst.get("camera_connected") else "?"
        size = f"{sensor[0]}x{sensor[1]}" if sensor and sensor[0] else "?"
        print(f"camera     : {cam}  ({size})")
        filt = inst.get("filter")
        if filt:
            print(f"filter     : slot {filt['slot']}  ({filt['name']})")
        elif "filter_error" in inst:
            print(f"filter     : error {inst['filter_error']}")

    backend = hwp.get("backend")
    if backend is None:
        print("hwp        : (none)")
    elif backend == "serial":
        pa = hwp.get("current_pa_deg")
        homed = hwp.get("homed")
        pa_s = f"{pa:.3f} deg" if isinstance(pa, (int, float)) else "?"
        print(f"hwp        : serial  PA={pa_s}  homed={homed}  moving={hwp.get('moving')}")
    else:
        pos = hwp.get("position_deg")
        pos_s = f"{pos:.3f} deg" if isinstance(pos, (int, float)) else "?"
        print(f"hwp        : alpaca  pos={pos_s}  moving={hwp.get('moving')}")

    if mount.get("connected") is None and "error" not in mount:
        print("mount      : (no PWI4 client)")
    elif "error" in mount:
        print(f"mount      : error {mount['error']}")
    else:
        ra = mount.get("ra_hours")
        dec = mount.get("dec_deg")
        coord = ""
        if isinstance(ra, (int, float)) and isinstance(dec, (int, float)):
            coord = f"  ra={ra:.4f}h dec={dec:.3f}d"
        print(
            f"mount      : connected={mount.get('connected')} "
            f"slewing={mount.get('slewing')} tracking={mount.get('tracking')}{coord}"
        )

    _print_pwi4_axis("focuser", info.get("focuser"), ("position", "pos", "{:.1f}"))
    _print_pwi4_axis(
        "field rot", info.get("field_rotator"), ("field_deg", "field", "{:.3f} deg")
    )


def _print_pwi4_axis(label: str, section: Optional[Dict[str, Any]], value) -> None:
    """One status line for a PWI4 auxiliary axis (focuser / field rotator)."""
    key, caption, fmt = value
    if not section or section.get("connected") is None:
        print(f"{label:<11}: (no PWI4 client)")
        return
    if section.get("exists") is False:
        print(f"{label:<11}: (not fitted)")
        return
    reading = section.get(key)
    shown = fmt.format(reading) if isinstance(reading, (int, float)) else "?"
    print(
        f"{label:<11}: connected={section.get('connected')} "
        f"enabled={section.get('enabled')} {caption}={shown} "
        f"moving={section.get('moving')}"
    )
