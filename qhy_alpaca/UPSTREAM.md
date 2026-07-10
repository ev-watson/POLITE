# Origin

Originally derived from [ryanswindle/alpaca-qhyccd-camera](https://github.com/ryanswindle/alpaca-qhyccd-camera)
(MIT, see LICENSE), now maintained as a POLITE-owned server. It was moved out of
`third_party/` to `qhy_alpaca/` — we no longer track upstream and carry our own fixes.

POLITE runs this as a **separate Alpaca server on port 11112** for the QHY268M while
the ZWO EFW and Optec Pyxis stay on ASCOM Remote Server port 11111.

Bring-up: `scripts/qhy268_bringup.ps1` from the POLITE repo root.

## POLITE changes beyond upstream
- Windows DLL discovery, QHY268M config profile, connect cleanup.
- **DATE-OBS / GPS fix.** Upstream unconditionally enabled QHY GPS timestamping and
  trusted the in-buffer GPS header. On a non-GPS camera (QHY268M) that header is
  just image pixels, which can decode as a `LOCKED` fix and feed garbage seconds
  through the JD-2450000.5 epoch, producing a `DATE-OBS` near 1995-10-09. Fixed by:
  - `defaults.has_gps` (default **False**) gates both enabling GPS and parsing the
    header; non-GPS cameras use the (NTP-synced) system clock. See `config.py`.
  - When GPS *is* enabled, a `gps_max_clock_skew_s` sanity window (default 60 s)
    rejects any GPS timestamp that disagrees with the system clock, as a backstop.
