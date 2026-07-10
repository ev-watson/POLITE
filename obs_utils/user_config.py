from __future__ import annotations

"""User-editable observatory configuration.

Observatory Windows (production):
- QHY268M camera        SDK-direct Alpaca server  (camera_host :11112, device 0)
- ZWO EFW filter wheel  ASCOM Remote Server       (host :11111)
- Optec Pyxis HWP       ASCOM Remote Server       (host :11111)

Lab Mac (INDIGO):
- All three on INDIGO ``indigo_agent_alpaca`` at ``localhost:11111``; leave
  ``camera_host=None``.
"""

from .config import AlpacaConfig, Pwi4Config, PyxisSerialConfig

PWI4_CONFIG = Pwi4Config(
    host="localhost",
    port=8220,
)

# Optec Pyxis 2" Gen3 (half-wave plate) on a direct FTDI serial link.
# Set enabled=True and confirm `port` once the cable is plugged in
# (`ls /dev/cu.usbserial*`). Driven by obs_utils.pyxis_gen3.PyxisGen3.
PYXIS_CONFIG = PyxisSerialConfig(
    enabled=False,
    port="/dev/cu.usbserial-OP7XD6WD",
    baud=115200,
    autodetect_baud=True,
)

ALPACA_CONFIG = AlpacaConfig(
    host="localhost:11111",
    camera_host="localhost:11112",  # ryanswindle SDK-direct server; None on lab Mac
    camera_index=0,
    guide_camera_index=None,   # QHY268M has no internal guide chip
    filterwheel_index=0,  # ASCOM Remote: FilterWheel 0 = ASCOM.EFW2.FilterWheel
    rotator_index=0,
    # ORDER MUST MATCH the physical EFW carousel (0-indexed software Position):
    #   pos 0 = slot 1 = Clear, pos 1 = slot 2 = B, pos 2 = slot 3 = V,
    #   pos 3 = slot 4 = R, pos 4 = slot 5 = Dark  (verified installed 2026-07-10).
    filter_names=[
        "Clear",
        "Photometric B",
        "Photometric V",
        "Photometric R",
        "Dark",
    ],
)
