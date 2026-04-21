# Windows Alpaca Bridge Setup

This project talks HTTP Alpaca only. The COM camera driver must live on Windows and be exposed via an Alpaca bridge.

Recommended bridge: **ASCOM Remote Server** (included with the ASCOM Platform).

## Install and enable
1. Install the ASCOM Platform on the Windows machine that has the COM camera driver.
2. Install the camera's ASCOM driver (COM) and verify it works in the vendor tool.
3. Install **ASCOM Remote Server** (Alpaca bridge) and run it.
4. In ASCOM Remote Server, add the camera device and note its device number.
5. Ensure Windows Firewall allows inbound connections to the Alpaca port (default 11111).

## Quick diagnostics
- From another machine: open `http://<windows-host>:11111/management/v1/description` in a browser.
- Or run `python alpyca_tools/scripts/camera_diag.py --host <windows-host>:11111`.

## Notes
- Keep all COM driver calls on Windows. This repo does not call COM directly.
- Use a static IP or a reserved DHCP lease for the Windows host to keep the URL stable.
