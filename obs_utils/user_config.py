from __future__ import annotations

"""User-editable observatory configuration.

Update device numbers here to match the INDIGO ``indigo_agent_alpaca`` mapping
(check the INDIGO Control Panel ``AGENT_ALPACA_DEVICES`` during bring-up):
- QHY268M main camera        (indigo_ccd_qhy2       -> Alpaca Camera)
- ZWO EFW 5-slot filter wheel (indigo_wheel_asi      -> Alpaca FilterWheel)
- Optec Pyxis 2" HWP stage   (indigo_rotator_optec  -> Alpaca Rotator)

The PlaneWave mount + field rotator stay on PWI4 (not INDIGO). The Optec Pyxis
Gen3 HWP rotator is driven natively over serial (see PYXIS_CONFIG below), not
through INDIGO -- its Gen3 controller protocol is incompatible with INDIGO's
legacy indigo_rotator_optec driver.
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
    camera_index=2,
    guide_camera_index=None,   # QHY268M has no internal guide chip
    filterwheel_index=1,
    rotator_index=0,        # set to the Pyxis Alpaca # to enable HWP control
    filter_names=[
        "Photometric B",
        "Photometric V",
        "Photometric R",
        "Clear",
        "Dark",
    ],
)
