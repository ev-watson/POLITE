from __future__ import annotations

"""Native serial driver for the Optec Pyxis 2" GEN3 rotator (HWP modulator).

The Pyxis 2" Gen3 controller speaks a *framed* text protocol that INDIGO's
legacy ``indigo_rotator_optec`` driver cannot handle (it expects the old
``CWAKUP``/``CCLINK`` -> ``!`` handshake and times out). POLITE therefore drives
the Pyxis directly over its FTDI serial link, bypassing INDIGO/Alpaca for the
rotator only. The camera + filter wheel stay on INDIGO->Alpaca; the mount +
field rotator stay on PWI4.

Protocol source (verified extract):
  reference-sheets/hardware/pyxis-command-processing-rev2.md
  ("Pyxis 2" GEN3 Programmers Reference Manual", rev 2.0, Nov 2018,
   firmware 3.0.1+); the generalized rev-3 rotator manual is the companion doc.

Frame syntax (no spaces)::

    <D d ii CCCCCC payload>

    D       target device: 'R' (rotator) or 'H' (hub)
    d       device id, always '1' for the Pyxis rotator
    ii      2-digit transaction id (00-99), echoed back after '!'
    CCCCCC  6-char command id (e.g. GETSTA, MOVEPA, DOHOME)
    payload variable-length parameter string (e.g. target PA x1000)

Responses: first line ``!ii`` (the echoed transaction id), then optional data
lines, terminated by a line ``END`` (GET/DO commands) or ``SET`` (SET commands).
``GETDNN`` emits a *doubled* ``END`` -- handled by flushing the input buffer
before each command. Errors replace ``!ii`` with ``ERROR ID = n`` / ``ERROR
TEXT = ...`` / ``END`` (see manual Appendix B).

Position angles are *instrumental* PA, degrees x1000 with no decimal point
(180.000 deg -> 180000), range 0..359999.

The class also exposes a small ASCOM-Alpaca ``IRotatorV3``-compatible shim
(``MoveAbsolute`` / ``IsMoving`` / ``Position`` / ``Connected``) so a
``PyxisGen3`` instance can stand in for an ``alpaca.rotator.Rotator`` in
``obs_utils.alpaca.move_rotator_absolute`` and ``imaging.select_hwp_angle``
without changing those call sites.
"""

import logging
import time
from typing import Dict, List, Optional, Sequence

try:  # pyserial is only needed to actually talk to hardware
    import serial
except ImportError:  # pragma: no cover - exercised only on hosts without pyserial
    serial = None  # type: ignore[assignment]

logger = logging.getLogger(__name__)

# Baud is not specified by the manual (the protocol is connection-agnostic).
# 19200 is what INDIGO's legacy Optec driver opens at and the most likely value;
# the rest are probed if ``autodetect`` is on. Order = most to least likely.
DEFAULT_BAUD_CANDIDATES: Sequence[int] = (19200, 115200, 9600, 38400, 57600)

_TERMINATORS = ("END", "SET")


class PyxisError(RuntimeError):
    """The controller returned an error frame (``ERROR ID = n`` ...)."""

    def __init__(self, code: Optional[int], text: str, command: Optional[str] = None):
        self.code = code
        self.text = text
        self.command = command
        msg = f"Pyxis error {code}: {text}"
        if command:
            msg += f" (command {command!r})"
        super().__init__(msg)


class PyxisTimeout(TimeoutError):
    """No terminated response was received within the read timeout."""


class PyxisGen3:
    """Direct-serial controller for an Optec Pyxis 2" Gen3 rotator.

    Typical use::

        rot = connect_pyxis_gen3("/dev/cu.usbserial-OP7XD6WD")
        rot.home()
        rot.move_absolute(90.0)
        print(rot.get_status())
        rot.close()
    """

    def __init__(
        self,
        port: str,
        baud: int = 19200,
        *,
        device: str = "R",
        device_id: str = "1",
        read_timeout_s: float = 2.0,
        write_timeout_s: float = 2.0,
        open_settle_s: float = 0.2,
        toggle_dtr: Optional[bool] = None,
    ) -> None:
        self._port = port
        self._baud = int(baud)
        self._device = device
        self._device_id = device_id
        self._read_timeout_s = float(read_timeout_s)
        self._write_timeout_s = float(write_timeout_s)
        self._open_settle_s = float(open_settle_s)
        self._toggle_dtr = toggle_dtr
        self._ser = None  # type: ignore[assignment]
        self._tid = 0

    # ------------------------------------------------------------------ #
    # Connection lifecycle
    # ------------------------------------------------------------------ #
    def open(self) -> "PyxisGen3":
        if serial is None:
            raise ImportError(
                "pyserial is required for PyxisGen3 (pip install pyserial)"
            )
        if self._ser is not None and self._ser.is_open:
            return self
        ser = serial.Serial()
        ser.port = self._port
        ser.baudrate = self._baud
        ser.bytesize = serial.EIGHTBITS
        ser.parity = serial.PARITY_NONE
        ser.stopbits = serial.STOPBITS_ONE
        ser.timeout = self._read_timeout_s
        ser.write_timeout = self._write_timeout_s
        # DTR toggling on open can reset some FTDI cables (see handoff notes on
        # the earlier pyserial "hang on open"). Leave pyserial's default unless
        # the caller explicitly asks to force a level.
        if self._toggle_dtr is not None:
            try:
                ser.dtr = self._toggle_dtr
                ser.rts = self._toggle_dtr
            except (OSError, ValueError):  # pragma: no cover - platform dependent
                logger.debug("Could not preset DTR/RTS before open; continuing")
        ser.open()
        time.sleep(self._open_settle_s)
        ser.reset_input_buffer()
        ser.reset_output_buffer()
        self._ser = ser
        logger.info("Opened Pyxis serial port %s @ %d baud", self._port, self._baud)
        return self

    def close(self) -> None:
        if self._ser is not None:
            try:
                self._ser.close()
            except Exception:  # pragma: no cover - best-effort teardown
                pass
        self._ser = None

    @property
    def is_open(self) -> bool:
        return self._ser is not None and bool(self._ser.is_open)

    def __enter__(self) -> "PyxisGen3":
        return self.open()

    def __exit__(self, *exc) -> None:
        self.close()

    def __repr__(self) -> str:
        state = "open" if self.is_open else "closed"
        return f"PyxisGen3(port={self._port!r}, baud={self._baud}, {state})"

    # ------------------------------------------------------------------ #
    # Low-level framed transaction
    # ------------------------------------------------------------------ #
    def _next_tid(self) -> str:
        tid = self._tid
        self._tid = (self._tid + 1) % 100
        return f"{tid:02d}"

    def transact(
        self,
        command: str,
        payload: str = "",
        *,
        target: Optional[str] = None,
        timeout_s: Optional[float] = None,
    ) -> List[str]:
        """Send one framed command and return its data lines.

        The echoed ``!ii`` and the trailing ``END``/``SET`` terminators are
        stripped; error frames raise :class:`PyxisError`.
        """
        if self._ser is None:
            raise RuntimeError("Pyxis serial port is not open; call open() first")
        tgt = target or self._device
        tid = self._next_tid()
        frame = f"<{tgt}{self._device_id}{tid}{command}{payload}>"

        # Flush any stray bytes (e.g. GETDNN's duplicate END) before writing.
        self._ser.reset_input_buffer()
        logger.debug("Pyxis TX %s", frame)
        self._ser.write(frame.encode("ascii"))
        self._ser.flush()

        lines = self._read_response(
            timeout_s if timeout_s is not None else self._read_timeout_s
        )
        logger.debug("Pyxis RX %s", lines)

        self._raise_if_error(lines, frame)

        echo = f"!{tid}"
        if echo not in lines:
            logger.warning(
                "Pyxis: expected echo %r not in response %r (command %s)",
                echo, lines, frame,
            )
        return [
            ln for ln in lines
            if ln != echo and ln.upper() not in _TERMINATORS
        ]

    def _read_response(self, timeout_s: float) -> List[str]:
        """Read raw bytes until a terminator line (``END``/``SET``) or timeout.

        Newline handling is deliberately tolerant (the manual's "ASCII 0x10" is
        treated as line-based): CR, LF and CRLF are all accepted as separators.
        """
        deadline = time.monotonic() + timeout_s
        raw = bytearray()
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise PyxisTimeout(
                    f"No terminated response within {timeout_s:g}s "
                    f"(got {bytes(raw)!r})"
                )
            self._ser.timeout = min(self._read_timeout_s, max(remaining, 0.05))
            chunk = self._ser.read(128)
            if chunk:
                raw.extend(chunk)
                if self._has_terminator(raw):
                    break
        return self._split_lines(raw)

    @staticmethod
    def _split_lines(raw: bytes) -> List[str]:
        norm = bytes(raw).replace(b"\r\n", b"\n").replace(b"\r", b"\n")
        out = []
        for ln in norm.split(b"\n"):
            text = ln.decode("ascii", "replace").strip()
            if text:
                out.append(text)
        return out

    @classmethod
    def _has_terminator(cls, raw: bytes) -> bool:
        return any(ln.upper() in _TERMINATORS for ln in cls._split_lines(raw))

    @staticmethod
    def _raise_if_error(lines: List[str], frame: str) -> None:
        code: Optional[int] = None
        text = ""
        for ln in lines:
            upper = ln.upper()
            if upper.startswith("ERROR ID"):
                code = PyxisGen3._int(PyxisGen3._kv_value(ln), default=-1)
            elif upper.startswith("ERROR TEXT"):
                text = PyxisGen3._kv_value(ln)
        if code is not None:
            raise PyxisError(code, text, command=frame)

    # ------------------------------------------------------------------ #
    # Parsing helpers
    # ------------------------------------------------------------------ #
    @staticmethod
    def _kv_value(line: str) -> str:
        return line.split("=", 1)[1].strip() if "=" in line else ""

    @staticmethod
    def _int(value: object, default: int = 0) -> int:
        try:
            return int(str(value).strip())
        except (TypeError, ValueError):
            return default

    @staticmethod
    def _flag(status: Dict[str, str], key: str) -> bool:
        return str(status.get(key, "0")).strip() not in ("0", "", "false", "False")

    @staticmethod
    def _pa_to_deg(pa: object) -> float:
        return PyxisGen3._int(pa) / 1000.0

    @staticmethod
    def _deg_to_pa(deg: float) -> int:
        """Instrumental degrees -> integer PA (x1000), wrapped into 0..359999."""
        wrapped = float(deg) % 360.0
        pa = int(round(wrapped * 1000.0))
        if pa >= 360000:  # 359.9996.. rounds up to a full turn
            pa = 0
        if not 0 <= pa <= 359999:
            raise ValueError(f"target PA {pa} out of range 0..359999 (deg={deg})")
        return pa

    # ------------------------------------------------------------------ #
    # High-level commands
    # ------------------------------------------------------------------ #
    def ping(self) -> str:
        """GETDNN -- returns the device nickname; used as a comms check."""
        data = self.transact("GETDNN")
        for ln in data:
            if ln.lower().startswith("nickname") and "=" in ln:
                return self._kv_value(ln)
        return data[0] if data else ""

    def get_status(self) -> Dict[str, str]:
        """GETSTA -- parsed ``key -> value`` status dictionary."""
        status: Dict[str, str] = {}
        for ln in self.transact("GETSTA"):
            if "=" in ln:
                key, val = ln.split("=", 1)
                status[key.strip()] = val.strip()
        return status

    def get_config(self) -> Dict[str, str]:
        """GETCFG -- parsed ``key -> value`` configuration dictionary."""
        config: Dict[str, str] = {}
        for ln in self.transact("GETCFG"):
            if "=" in ln:
                key, val = ln.split("=", 1)
                config[key.strip()] = val.strip()
        return config

    @property
    def position_deg(self) -> float:
        return self._pa_to_deg(self.get_status().get("Current PA", "0"))

    @property
    def target_deg(self) -> float:
        return self._pa_to_deg(self.get_status().get("Target PA", "0"))

    @property
    def is_moving(self) -> bool:
        return self._flag(self.get_status(), "Is Moving")

    @property
    def is_homing(self) -> bool:
        return self._flag(self.get_status(), "Is Homing")

    @property
    def is_homed(self) -> bool:
        return self._flag(self.get_status(), "Is Homed")

    def halt(self) -> None:
        """DOHALT -- immediately stop the current movement."""
        self.transact("DOHALT")

    def home(self, poll_s: float = 0.5, timeout_s: float = 180.0) -> float:
        """DOHOME -- run the homing routine; block until homed.

        Returns the final ``Current PA`` in degrees. A short initial wait lets
        the controller assert ``Is Homing`` before the first status poll so we
        do not mistake the pre-home state for completion.
        """
        self.transact("DOHOME")
        t0 = time.monotonic()
        time.sleep(min(poll_s, 1.0))
        status: Dict[str, str] = {}
        while True:
            status = self.get_status()
            if not self._flag(status, "Is Homing") and self._flag(status, "Is Homed"):
                break
            if time.monotonic() - t0 > timeout_s:
                raise PyxisTimeout(f"Pyxis home timed out after {timeout_s:g}s")
            time.sleep(poll_s)
        return self._pa_to_deg(status.get("Current PA", "0"))

    def move_absolute(
        self,
        deg: float,
        poll_s: float = 0.5,
        timeout_s: float = 120.0,
        pre_poll_s: float = 0.2,
    ) -> float:
        """MOVEPA -- move to an absolute instrumental PA; block until settled.

        Returns the final ``Current PA`` in degrees. Raises :class:`PyxisError`
        (e.g. code 11, "Device is Not Homed") if the controller rejects the move.
        """
        target_pa = self._deg_to_pa(deg)
        self.transact("MOVEPA", str(target_pa))
        t0 = time.monotonic()
        time.sleep(pre_poll_s)  # let the controller assert Is Moving = 1
        status: Dict[str, str] = {}
        while True:
            status = self.get_status()
            if not self._flag(status, "Is Moving"):
                break
            if time.monotonic() - t0 > timeout_s:
                raise PyxisTimeout(
                    f"Pyxis move to {deg:g} deg timed out after {timeout_s:g}s"
                )
            time.sleep(poll_s)
        return self._pa_to_deg(status.get("Current PA", str(target_pa)))

    # ------------------------------------------------------------------ #
    # ASCOM Alpaca IRotatorV3-compatible shim
    #
    # Lets a PyxisGen3 stand in for an alpaca.rotator.Rotator so it slots into
    # obs_utils.alpaca.move_rotator_absolute / imaging.select_hwp_angle unchanged.
    # Alpaca's MoveAbsolute is asynchronous; callers poll IsMoving then read
    # Position -- which is exactly how MOVEPA + GETSTA behave here.
    # ------------------------------------------------------------------ #
    def MoveAbsolute(self, position_deg: float) -> None:  # noqa: N802 (Alpaca name)
        self.transact("MOVEPA", str(self._deg_to_pa(position_deg)))

    @property
    def IsMoving(self) -> bool:  # noqa: N802 (Alpaca name)
        return self.is_moving

    @property
    def Position(self) -> float:  # noqa: N802 (Alpaca name)
        return self.position_deg

    @property
    def Connected(self) -> bool:  # noqa: N802 (Alpaca name)
        return self.is_open

    @Connected.setter
    def Connected(self, value: bool) -> None:  # noqa: N802 (Alpaca name)
        if value:
            self.open()
        else:
            self.close()


def connect_pyxis_gen3(
    port: str,
    baud: int = 19200,
    *,
    autodetect: bool = True,
    baud_candidates: Optional[Sequence[int]] = None,
    read_timeout_s: float = 2.0,
    toggle_dtr: Optional[bool] = None,
) -> PyxisGen3:
    """Open the Pyxis and verify comms with a GETDNN ping.

    With ``autodetect`` the given ``baud`` is tried first, then the remaining
    :data:`DEFAULT_BAUD_CANDIDATES` (or ``baud_candidates``) until a ping
    succeeds. Raises :class:`ConnectionError` if none answer.
    """
    if autodetect:
        pool = list(baud_candidates or DEFAULT_BAUD_CANDIDATES)
        candidates = [baud] + [b for b in pool if b != baud]
    else:
        candidates = [baud]

    last_err: Optional[Exception] = None
    for b in candidates:
        dev = PyxisGen3(port, baud=b, read_timeout_s=read_timeout_s, toggle_dtr=toggle_dtr)
        try:
            dev.open()
            nickname = dev.ping()
            logger.info(
                "Pyxis Gen3 connected on %s @ %d baud (nickname=%r)", port, b, nickname
            )
            return dev
        except (PyxisError, PyxisTimeout, OSError, ImportError) as exc:
            last_err = exc
            dev.close()
            logger.debug("Pyxis probe @ %d baud failed: %s", b, exc)

    raise ConnectionError(
        f"Could not reach Pyxis Gen3 on {port} (tried baud {candidates}): {last_err}"
    )
