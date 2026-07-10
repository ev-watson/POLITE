from pathlib import Path
from typing import Dict, List, Optional
import os
import sys

import yaml
from pydantic import BaseModel, Field


def _load_yaml_configs() -> dict:
    """Load config.yaml with optional overrides."""
    base_config = {}
    override_config = {}

    config_env = os.environ.get("QHYCCD_ALPACA_CONFIG")
    if config_env:
        base_path = Path(config_env)
    else:
        win_path = Path(__file__).parent.parent / "config.windows.yaml"
        base_path = win_path if sys.platform == "win32" and win_path.exists() else Path(__file__).parent.parent / "config.yaml"
    if base_path.exists():
        with open(base_path, "r") as f:
            base_config = yaml.safe_load(f) or {}

    docker_path = Path("/alpyca/config.yaml")
    if docker_path.exists():
        with open(docker_path, "r") as f:
            override_config = yaml.safe_load(f) or {}

    def deep_merge(base: dict, override: dict) -> dict:
        result = base.copy()
        for key, value in override.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = deep_merge(result[key], value)
            else:
                result[key] = value
        return result

    return deep_merge(base_config, override_config)


class DeviceDefaults(BaseModel):
    temperature: float = Field(default=-10.0)
    readout_mode: int = Field(default=0)
    binning: int = Field(default=1)
    gain: int = Field(default=0)
    offset: int = Field(default=0)
    usb_traffic: float = Field(default=10.0)
    # GPS timestamping. Only cameras with the QHYCCD GPS receiver module
    # (e.g. QHY174-GPS) produce a valid GPS header. On non-GPS cameras such as
    # the QHY268M the "GPS struct" is just the first image pixels, which can be
    # misparsed as a LOCKED fix and yield a bogus DATE-OBS near JD 2450000.5
    # (1995-10-09). Leave this False unless a real GPS receiver is installed.
    has_gps: bool = Field(default=False)
    # Reject a GPS-derived timestamp that disagrees with the system clock by
    # more than this many seconds (defense-in-depth against a misparsed header).
    gps_max_clock_skew_s: float = Field(default=60.0)


class DeviceConfig(BaseModel):
    entity: str = Field(default="Camera")
    device_number: int = Field(default=0)
    serial_number: str = Field(default="")
    defaults: DeviceDefaults = Field(default_factory=DeviceDefaults)


class ServerConfig(BaseModel):
    host: str = Field(default="0.0.0.0")
    port: int = Field(default=5000)


class Config(BaseModel):
    entity: str = Field(default="qhyccd_camera")
    library: str = Field(default="/usr/local/lib/libqhyccd.so")
    server: ServerConfig = Field(default_factory=ServerConfig)
    log_level: str = Field(default="INFO")
    devices: List[DeviceConfig] = Field(default_factory=list)

    @classmethod
    def load(cls) -> "Config":
        return cls(**_load_yaml_configs())

    def get_device(self, device_number: int) -> Optional[DeviceConfig]:
        for device in self.devices:
            if device.device_number == device_number:
                return device
        return None


config = Config.load()