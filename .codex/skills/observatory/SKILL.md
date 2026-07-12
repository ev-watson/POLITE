---
name: observatory
description: Conventions for writing POLITE observatory automation scripts (mount/camera/EFW/HWP night sessions). Use when creating or editing observatory control scripts, night-session automation, or FITS-producing capture code, especially for file-naming and phase structure.
---

# Observatory Script Skill
When creating observatory automation scripts:
- Use Python 3 with structured logging
- Include phases: startup, calibration, imaging, shutdown
- Use naming convention: :q:t:fr:f:e:b.fit
  - Where:
    - :q = 8-digit zero-padded Sequence number (e.g. 00000001, 00000002, etc.)
    - :t = target name, only if frame type is Light, otherwise leave out (e.g. Jupiter)
    - :fr = frame type (Light, Dark, FlatField)
    - :f = filter name (e.g. L/Clear, Red, Green, Blue, etc.)
    - :e = exposure time formatted like [#].###[secs, mins] (e.g. 12.000secs, 2.500mins, etc.)
    - :b = binning, only if frame type is FlatField, otherwise leave out (e.g. 1x1, 2x1, 2x2, etc.)
- Add error recovery for each phase
- Target macOS/zsh environment
