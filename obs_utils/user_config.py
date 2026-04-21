from __future__ import annotations

"""User-editable observatory configuration.

Update device indices here to match ASCOM Remote Server assignments:
- SBIGImagingCamera (main)
- SBIGGuidingCamera (internal guide chip)
- SBIG filter wheel
"""

from .config import AlpacaConfig, Pwi4Config

PWI4_CONFIG = Pwi4Config(
    host="localhost",
    port=8220,
)

ALPACA_CONFIG = AlpacaConfig(
    host="localhost:11111",
    camera_index=0,
    guide_camera_index=1,
    filterwheel_index=0,
    filter_names=["Clear/Luminance", "Red", "Blue", "Green", "Halpha"],
)
