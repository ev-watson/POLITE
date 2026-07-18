from __future__ import annotations

"""Lifecycle management for the observatory's two local Alpaca servers.

The QHY268M camera is served directly by ``qhy_alpaca`` on port 11112, while
the ZWO EFW and Optec Pyxis are served by ASCOM Remote on port 11111.  The
servers are deliberately long-lived: this module starts them before a normal
observatory startup and retains any subprocess handles for the life of the
Python kernel.  It does not connect individual hardware devices; the usual
``open_imaging_session`` path still owns those Alpaca connections.
"""

import json
import logging
import os
from pathlib import Path
import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from typing import Dict, Optional, Sequence
from urllib.error import URLError
from urllib.parse import urlsplit
from urllib.request import urlopen


logger = logging.getLogger(__name__)

ASCOM_REMOTE_PORT = 11111
QHY_ALPACA_PORT = 11112
DEFAULT_ASCOM_REMOTE_EXE = Path(r"C:\Program Files\ASCOM\RemoteServer\RemoteServer.exe")
_LOOPBACK_HOSTS = frozenset({"localhost", "127.0.0.1", "::1"})


@dataclass(frozen=True)
class AlpacaServerStatus:
    """Readiness result returned by :func:`start_observatory_alpaca_servers`."""

    ascom_endpoint: str
    qhy_endpoint: str
    ascom_started: bool
    qhy_started: bool


# Keep references to daemons that this Python kernel launched.  Do not call
# wait(): both processes must survive after this helper returns.
_server_processes: Dict[str, subprocess.Popen] = {}
_server_start_lock = threading.Lock()


def _is_windows() -> bool:
    return os.name == "nt"


def _split_endpoint(endpoint: str) -> tuple[str, int]:
    """Return the hostname and port from an Alpaca endpoint string."""
    text = endpoint.strip().rstrip("/")
    parsed = urlsplit(text if "://" in text else f"//{text}", scheme="http")
    if parsed.scheme not in {"http", "https"} or not parsed.hostname or parsed.port is None:
        raise ValueError(
            f"Alpaca endpoint must include a host and port, got {endpoint!r}"
        )
    return parsed.hostname.lower(), parsed.port


def _management_url(endpoint: str) -> str:
    host, port = _split_endpoint(endpoint)
    host_text = f"[{host}]" if ":" in host and not host.startswith("[") else host
    return f"http://{host_text}:{port}/management/apiversions"


def _endpoint_is_local(endpoint: Optional[str], expected_port: int) -> bool:
    if endpoint is None:
        return False
    try:
        host, port = _split_endpoint(endpoint)
    except ValueError:
        return False
    return host in _LOOPBACK_HOSTS and port == expected_port


def uses_local_observatory_alpaca_servers(
    ascom_endpoint: str,
    qhy_endpoint: Optional[str],
) -> bool:
    """Whether this is the observatory Windows two-server configuration.

    The lab configuration uses one INDIGO server and must not launch Windows
    executables.  Likewise, an explicitly remote Alpaca host is left alone.
    """
    return (
        _is_windows()
        and _endpoint_is_local(ascom_endpoint, ASCOM_REMOTE_PORT)
        and _endpoint_is_local(qhy_endpoint, QHY_ALPACA_PORT)
    )


def _server_is_ready(endpoint: str, request_timeout_s: float = 1.0) -> bool:
    """Return whether the server's read-only Alpaca management endpoint answers."""
    try:
        with urlopen(_management_url(endpoint), timeout=request_timeout_s) as response:
            if response.status != 200:
                return False
            payload = json.loads(response.read().decode("utf-8"))
    except (OSError, URLError, ValueError, json.JSONDecodeError):
        return False

    # ASCOM Remote and the QHY implementation use different response field
    # casing, but each successful endpoint supplies an API-version value.
    return isinstance(payload, dict) and any(
        key in payload for key in ("Value", "ApiVersions", "APIVersions")
    )


def _announce(message: str) -> None:
    """Show a commissioning message immediately and retain it in the session log."""
    print(message, flush=True)
    logger.info(message)


def _repo_root() -> Path:
    return Path(__file__).resolve().parent.parent


def _qhy_environment(qhy_root: Path) -> Dict[str, str]:
    environment = os.environ.copy()
    source_dir = qhy_root / "src"
    environment["QHYCCD_ALPACA_CONFIG"] = str(qhy_root / "config.windows.yaml")
    prior_pythonpath = environment.get("PYTHONPATH")
    environment["PYTHONPATH"] = (
        str(source_dir)
        if not prior_pythonpath
        else f"{source_dir}{os.pathsep}{prior_pythonpath}"
    )
    return environment


def _launch_ascom_remote(executable: Path) -> subprocess.Popen:
    if not executable.is_file():
        raise FileNotFoundError(
            "ASCOM Remote Server was not found at "
            f"{executable}. Install/configure ASCOM Remote Server before startup."
        )
    return subprocess.Popen([str(executable)])


def _launch_qhy_server(
    qhy_root: Path,
    python_executable: Path,
) -> subprocess.Popen:
    config_path = qhy_root / "config.windows.yaml"
    main_path = qhy_root / "src" / "main.py"
    if not config_path.is_file() or not main_path.is_file():
        raise FileNotFoundError(
            "QHY Alpaca server files are missing. Expected "
            f"{config_path} and {main_path}."
        )
    return subprocess.Popen(
        [str(python_executable), str(main_path)],
        cwd=str(qhy_root),
        env=_qhy_environment(qhy_root),
    )


def _existing_process_is_running(name: str) -> bool:
    process = _server_processes.get(name)
    return process is not None and process.poll() is None


def _process_failure_details(names: Sequence[str]) -> str:
    details = []
    for name in names:
        process = _server_processes.get(name)
        if process is not None and process.poll() is not None:
            details.append(f"{name} exited with code {process.returncode}")
    return f" ({'; '.join(details)})" if details else ""


def _await_servers(
    ascom_endpoint: str,
    qhy_endpoint: str,
    *,
    timeout_s: float,
    poll_s: float,
) -> None:
    deadline = time.monotonic() + timeout_s
    while True:
        ascom_ready = _server_is_ready(ascom_endpoint)
        qhy_ready = _server_is_ready(qhy_endpoint)
        if ascom_ready and qhy_ready:
            return
        if time.monotonic() >= deadline:
            unavailable = []
            if not ascom_ready:
                unavailable.append(f"ASCOM Remote ({ascom_endpoint})")
            if not qhy_ready:
                unavailable.append(f"QHY Alpaca ({qhy_endpoint})")
            detail = _process_failure_details(("ASCOM Remote", "QHY Alpaca"))
            raise TimeoutError(
                "Timed out waiting for Alpaca server readiness: "
                f"{', '.join(unavailable)}{detail}"
            )
        time.sleep(poll_s)


def start_observatory_alpaca_servers(
    *,
    ascom_endpoint: str = "localhost:11111",
    qhy_endpoint: str = "localhost:11112",
    timeout_s: float = 60.0,
    poll_s: float = 0.25,
    ascom_executable: Path = DEFAULT_ASCOM_REMOTE_EXE,
    qhy_root: Optional[Path] = None,
    python_executable: Optional[Path] = None,
) -> AlpacaServerStatus:
    """Start and verify ASCOM Remote and QHY Alpaca as one idempotent unit.

    Each server is launched only when its own read-only ``management/apiversions``
    endpoint is unavailable.  Both daemons are spawned before readiness polling
    begins, so their startup runs in parallel.  Existing manual or prior-kernel
    servers are reused.  The function validates server availability only; device
    connections remain the responsibility of the normal Alpaca connection calls.

    This launcher is intentionally restricted to a Windows local-host layout.
    Close EZCAP and other QHY applications first because the camera USB device
    is exclusive.
    """
    if timeout_s <= 0 or poll_s <= 0:
        raise ValueError("timeout_s and poll_s must both be positive")
    if not uses_local_observatory_alpaca_servers(ascom_endpoint, qhy_endpoint):
        raise RuntimeError(
            "The consolidated launcher is only valid for local Windows ASCOM "
            "Remote (:11111) and QHY Alpaca (:11112) endpoints."
        )

    qhy_root = (qhy_root or (_repo_root() / "qhy_alpaca")).resolve()
    python_executable = python_executable or Path(sys.executable)
    ascom_started = False
    qhy_started = False

    # Serialise the readiness check and spawn decision so notebook cell re-runs
    # cannot launch a second copy while the first call is still starting up.
    with _server_start_lock:
        ascom_ready = _server_is_ready(ascom_endpoint)
        qhy_ready = _server_is_ready(qhy_endpoint)

        if ascom_ready:
            _announce(f"ASCOM Remote already ready at {ascom_endpoint}")
        elif _existing_process_is_running("ASCOM Remote"):
            _announce("ASCOM Remote is already starting; waiting for readiness")
        else:
            _announce(f"Starting ASCOM Remote Server at {ascom_endpoint}")
            _server_processes["ASCOM Remote"] = _launch_ascom_remote(ascom_executable)
            ascom_started = True

        if qhy_ready:
            _announce(f"QHY Alpaca already ready at {qhy_endpoint}")
        elif _existing_process_is_running("QHY Alpaca"):
            _announce("QHY Alpaca is already starting; waiting for readiness")
        else:
            _announce(f"Starting QHY Alpaca Server at {qhy_endpoint}")
            _server_processes["QHY Alpaca"] = _launch_qhy_server(
                qhy_root, python_executable
            )
            qhy_started = True

        _await_servers(
            ascom_endpoint,
            qhy_endpoint,
            timeout_s=timeout_s,
            poll_s=poll_s,
        )

    _announce(
        "Alpaca servers ready: "
        f"ASCOM Remote ({ascom_endpoint}); QHY Alpaca ({qhy_endpoint})"
    )
    return AlpacaServerStatus(
        ascom_endpoint=ascom_endpoint,
        qhy_endpoint=qhy_endpoint,
        ascom_started=ascom_started,
        qhy_started=qhy_started,
    )
