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
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

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

logger = logging.getLogger(__name__)


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
    def rotator(self):
        """Alpaca (observatory) HWP rotator, if wired via INDIGO/ASCOM."""
        return None if self.imaging is None else self.imaging.rotator

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

    # ---- status ----------------------------------------------------------- #
    def status(self, pretty: bool = True) -> Dict[str, Any]:
        """Collect a health summary. Each subsystem is probed independently so a
        partial bring-up still reports what it can.
        """
        info: Dict[str, Any] = {
            "instrument": self._instrument_status(),
            "hwp": self._hwp_status(),
            "mount": self._mount_status(),
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
        rot = self.rotator
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


def connect_rotator(
    *, alpaca_config: Optional[AlpacaConfig] = None, reuse: bool = True
) -> ObservatorySession:
    """Connect only the Alpaca HWP rotator (observatory path) and attach it.

    For the lab bench's native-serial Pyxis Gen3, use ``connect(pyxis_serial=True)``
    instead -- the serial port is exclusive and homed through its own driver.
    """
    from .alpaca import connect_rotator as _connect_rotator

    session = _ensure_session()
    cfg = alpaca_config or _uc.ALPACA_CONFIG
    if cfg.rotator_index is None:
        raise RuntimeError(
            "AlpacaConfig.rotator_index is None -- no Alpaca rotator configured. "
            "For the lab serial Pyxis use connect(pyxis_serial=True)."
        )
    imaging = _ensure_imaging(session, cfg)
    if not (reuse and imaging.rotator is not None):
        imaging.rotator = _connect_rotator(cfg.host, cfg.rotator_index)
        logger.info("Alpaca rotator connected on %s (device=%d)", cfg.host, cfg.rotator_index)
    _SESSION = session
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
    print("-" * 40)

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
