# Upstream

Forked from [ryanswindle/alpaca-qhyccd-camera](https://github.com/ryanswindle/alpaca-qhyccd-camera)
with POLITE observatory patches (Windows DLL discovery, QHY268M config, connect cleanup).

POLITE runs this as a **separate Alpaca server on port 11112** for the QHY268M while
the ZWO EFW and Optec Pyxis stay on ASCOM Remote Server port 11111.

Bring-up: `scripts/qhy268_bringup.ps1` from the POLITE repo root.
