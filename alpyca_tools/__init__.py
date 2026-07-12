"""ASCOM Alpaca camera helpers with optional hardware dependencies.

The package initializer deliberately avoids importing the Alpaca client.  This
keeps pure helpers such as :mod:`alpyca_tools.fits_writer` usable on reduction
machines where the observatory-control dependency is not installed.
"""

from __future__ import annotations

from importlib import import_module

_EXPORTS = {
    "CameraDevice": (".camera_device", "CameraDevice"),
    "ExposureSettings": (".camera_ops", "ExposureSettings"),
    "capture_image": (".camera_ops", "capture_image"),
    "configure_camera": (".camera_ops", "configure_camera"),
    "download_image": (".camera_ops", "download_image"),
    "wait_ready": (".camera_ops", "wait_ready"),
    "discover": (".discovery", "discover"),
    "FitsHeaderConfig": (".fits_writer", "FitsHeaderConfig"),
    "capture_fits": (".fits_writer", "capture_fits"),
    "write_fits": (".fits_writer", "write_fits"),
    "CameraState": (".schema", "CameraState"),
    "Telemetry": (".telemetry", "Telemetry"),
    "setup_logging": (".telemetry", "setup_logging"),
}

__all__ = list(_EXPORTS)


def __getattr__(name: str):
    """Import a public object only when it is first requested."""
    try:
        module_name, attr_name = _EXPORTS[name]
    except KeyError as exc:  # pragma: no cover - standard module protocol
        raise AttributeError(name) from exc
    value = getattr(import_module(module_name, __name__), attr_name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))
