from __future__ import annotations

"""Bounded polling for device state changes.

Every PWI4 wait in this project started life as ``while not <ready>: sleep()``.
In a *script* that stalls one run and the operator sees it hang. In a *notebook*
it is worse: the kernel is the session, so one unbounded loop on a dead axis
wedges the camera, the wheel, and the HWP along with the mount, and the only way
out is a kernel restart that drops every live connection. Recovering from that
costs more than the slew it was waiting on.

:func:`wait_until` is the one place that pattern lives. It polls a predicate to a
deadline and raises :class:`DeviceTimeout`, whose message names the device and
the wait, so a failure reads as "focuser did not enable within 30 s" instead of a
silent hang. It generalizes the bounded enable already written for the mount in
:func:`obs_utils.night_safety._enable_axes`, so the runner's gates and the
notebook helpers share one implementation rather than two that drift.

The predicate is called immediately, before any sleep: a device that is already
in the requested state costs nothing.
"""

import logging
import time
from typing import Callable, Optional, TypeVar

logger = logging.getLogger(__name__)

__all__ = ["DeviceTimeout", "wait_until", "wait_for_value"]

T = TypeVar("T")


class DeviceTimeout(TimeoutError):
    """A device did not reach the expected state before the deadline.

    Subclasses :class:`TimeoutError`, so callers that already catch that keep
    working. ``what`` and ``timeout_s`` are kept as attributes for callers that
    want to report the failing device rather than re-parse the message.
    """

    def __init__(self, what: str, timeout_s: float, detail: str = "") -> None:
        self.what = what
        self.timeout_s = timeout_s
        message = f"{what} did not complete within {timeout_s:.0f} s"
        if detail:
            message = f"{message}. {detail}"
        super().__init__(message)


def wait_until(
    predicate: Callable[[], bool],
    *,
    timeout_s: float,
    poll_s: float = 0.5,
    what: str = "device",
    detail: str = "",
    on_error: str = "raise",
) -> float:
    """Poll ``predicate`` until it returns true. Return the elapsed seconds.

    Raises :class:`DeviceTimeout` if ``timeout_s`` passes first. ``detail`` is
    appended to the message and is the right place for the operator fix ("check
    the drive power and rerun").

    ``on_error`` controls what a *raising predicate* means. Status reads over
    HTTP fail transiently, and a dropped packet should not abort a slew, so the
    default ``"retry"`` semantics of the loop are: with ``on_error="raise"`` the
    exception propagates immediately; with ``on_error="retry"`` it is logged at
    debug level and the poll continues to the deadline. Use ``"retry"`` while
    waiting for a device to *appear*, ``"raise"`` once it is known live.
    """
    if on_error not in ("raise", "retry"):
        raise ValueError(f"on_error must be 'raise' or 'retry', got {on_error!r}")

    started = time.monotonic()
    deadline = started + float(timeout_s)
    while True:
        try:
            if predicate():
                return time.monotonic() - started
        except Exception as exc:
            if on_error == "raise":
                raise
            logger.debug("[%s] status read failed, retrying: %r", what, exc)
        if time.monotonic() >= deadline:
            raise DeviceTimeout(what, float(timeout_s), detail)
        time.sleep(poll_s)


def wait_for_value(
    read: Callable[[], T],
    target: T,
    *,
    tol: float,
    timeout_s: float,
    poll_s: float = 0.5,
    what: str = "device",
    detail: str = "",
) -> T:
    """Wait until ``read()`` lands within ``tol`` of ``target``; return the value.

    The motion counterpart of :func:`wait_until`, for axes that report a position
    rather than a done flag (focuser steps, rotator degrees). On timeout the
    message carries the last value read, which is what tells you whether the
    device is stuck or merely slow.
    """
    last: Optional[T] = None

    def landed() -> bool:
        nonlocal last
        last = read()
        return abs(float(last) - float(target)) <= tol  # type: ignore[arg-type]

    try:
        wait_until(
            landed,
            timeout_s=timeout_s,
            poll_s=poll_s,
            what=what,
            detail=detail,
            on_error="retry",
        )
    except DeviceTimeout as exc:
        raise DeviceTimeout(
            what,
            float(timeout_s),
            f"last position {last!r}, target {target!r} (tol {tol}). {detail}".strip(),
        ) from exc
    return last  # type: ignore[return-value]
