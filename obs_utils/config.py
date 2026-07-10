from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class Pwi4Config:
    host: str = "localhost"
    port: int = 8220


@dataclass
class PyxisSerialConfig:
    """Direct-serial config for the Optec Pyxis 2" Gen3 HWP rotator.

    The Gen3 controller uses a framed serial protocol that INDIGO's legacy
    ``indigo_rotator_optec`` cannot speak, so POLITE drives it natively over
    serial (bypassing INDIGO/Alpaca for the rotator only; camera + EFW stay on
    INDIGO->Alpaca, mount + field rotator stay on PWI4). See
    ``obs_utils.pyxis_gen3`` and
    ``reference-sheets/hardware/pyxis-command-processing-rev2.md``.

    On macOS use the ``/dev/cu.*`` node (not ``/dev/tty.*``). The Optec FTDI
    cable enumerated as ``/dev/cu.usbserial-OP7XD6WD`` during bring-up; confirm
    with ``ls /dev/cu.usbserial*``. Baud is not stated by the manual -- 19200 is
    the most likely value and ``autodetect_baud`` probes the rest.
    """

    enabled: bool = False
    port: str = "/dev/cu.usbserial-OP7XD6WD"
    baud: int = 19200
    autodetect_baud: bool = True


@dataclass
class AlpacaConfig:
    """Alpaca device mapping.

    On the lab Mac, INDIGO exposes camera + EFW + rotator on one endpoint
    (default ``:11111``). On the observatory Windows PC the QHY268M may use a
    separate SDK-direct Alpaca server (``camera_host``, default ``:11112``)
    while the ZWO EFW and Optec Pyxis stay on ASCOM Remote Server (``host``,
    default ``:11111``).
    """

    host: str = "localhost:11111"
    # When set, the main camera uses this endpoint instead of ``host``.
    camera_host: Optional[str] = None
    camera_index: int = 0
    # QHY268M is a single mono CMOS with no internal guide chip; leave None
    # unless a separate guide camera is wired as its own INDIGO/Alpaca device.
    guide_camera_index: Optional[int] = None
    filterwheel_index: Optional[int] = 0  # Alpaca FilterWheel # for the ZWO EFW (indigo_wheel_asi)
    # Optec Pyxis 2" carrying the half-wave plate (indigo_rotator_optec). None
    # disables HWP control (e.g. plain imaging with no polarimetry stage).
    rotator_index: Optional[int] = None
    # ZWO 5-slot EFW labels; must match the poltools FilterConfig / FITS FILTER
    # keyword (see poltools.default_efw_filters).
    filter_names: List[str] = field(
        # Order matches the physical EFW carousel (0-indexed Position):
        # 0=Clear(slot1) 1=B(slot2) 2=V(slot3) 3=R(slot4) 4=Dark(slot5).
        default_factory=lambda: [
            "Clear",
            "Photometric B",
            "Photometric V",
            "Photometric R",
            "Dark",
        ]
    )


@dataclass
class SkyRegionLimit:
    name: str
    alt_min_deg: float
    alt_max_deg: float
    az_min_deg: float
    az_max_deg: float


@dataclass
class SlewLimits:
    max_slew_rate_deg_s: Optional[float] = 5.0
    enforce_rate: bool = False
    regions: List[SkyRegionLimit] = field(default_factory=list)
    enforce_regions: bool = True


def default_sky_regions() -> List[SkyRegionLimit]:
    return [
        SkyRegionLimit(
            name="placeholder_sky_box",
            alt_min_deg=15.0,
            alt_max_deg=85.0,
            az_min_deg=0.0,
            az_max_deg=360.0,
        )
    ]
