"""
Alpaca (alpyca) camera -> FITS capture routine with research-grade headers.

Notes:
- FITS required structural keywords (SIMPLE/BITPIX/NAXIS/NAXISn) are written by astropy.
- This adds standard observational, instrument, and (optional) WCS metadata in a FITS-safe way.
- Many keywords in your example (e.g., UCAM*, GEOMCODE, etc.) are instrument/pipeline-specific;
  this supports injecting them via `extra_cards` without hard-coding a single camera’s schema.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union

import numpy as np
from alpyca_tools.camera_device import CameraDevice
from alpaca.camera import ImageArrayElementTypes
from astropy.io import fits


@dataclass
class CaptureConfig:
    host: str = "localhost:32323"      # e.g. "192.168.1.10:11111"
    camera_index: int = 0

    exposure_s: float = 1.0
    is_light: bool = True              # True=light, False=dark/frame/flat depends on your workflow
    imagetyp: str = "LIGHT"            # "LIGHT", "DARK", "BIAS", "FLAT", etc.

    binx: int = 1
    biny: int = 1
    startx: int = 0
    starty: int = 0
    numx: Optional[int] = None         # default full frame after binning
    numy: Optional[int] = None

    poll_s: float = 0.5
    timeout_s: float = 300.0
    verbose: bool = True

    # Common observational metadata (set what you know; omit what you do not).
    object_name: Optional[str] = None
    observer: Optional[str] = None
    telescope: Optional[str] = None
    observatory: Optional[str] = None          # OBSERVAT
    instrument: Optional[str] = None           # INSTRUME override
    filter_name: Optional[str] = None          # FILTER / FILTNAM style
    airmass: Optional[float] = None
    ra: Optional[str] = None                   # sexagesimal string "HH:MM:SS.S"
    dec: Optional[str] = None                  # sexagesimal string "+DD:MM:SS.S"
    ha: Optional[str] = None                   # sexagesimal string
    equinox: Optional[float] = 2000.0          # EQUINOX, if you provide WCS/RA/DEC

    # Optional WCS (Celestial TAN example). Supply if you have a plate solution.
    # Keys should be valid FITS keywords: CRPIX1, CRVAL1, CD1_1, CTYPE1, etc.
    wcs_cards: Dict[str, Any] = field(default_factory=dict)

    # Extra instrument/pipeline cards (your UCAM*/GEOM*/DATASEC/etc live here).
    extra_cards: Dict[str, Any] = field(default_factory=dict)

    # If you want FITS CHECKSUM/DATASUM
    add_checksum: bool = True


def _alpaca_image_to_numpy(c: CameraDevice) -> Tuple[np.ndarray, np.dtype]:
    """
    Convert alpyca ImageArray (+ ImageArrayInfo) to a numpy array with astropy-compatible axis order.
    alpyca returns nested lists with axes that typically need transposition.
    """
    img = c.ImageArray
    info = c.ImageArrayInfo

    if info.ImageElementType == ImageArrayElementTypes.Int32:
        # Common convention: if camera max ADU fits in uint16, write as uint16 with BZERO/BSCALE
        if c.MaxADU <= 65535:
            dtype = np.uint16
        else:
            dtype = np.int32
    elif info.ImageElementType == ImageArrayElementTypes.Double:
        dtype = np.float64
    else:
        # Conservative fallback
        dtype = np.int32

    if info.Rank == 2:
        nda = np.array(img, dtype=dtype).transpose()
    else:
        # Some devices return (plane, y, x) or (x,y,plane) depending on driver; this matches alpyca example.
        nda = np.array(img, dtype=dtype).transpose(2, 1, 0)

    return nda, np.dtype(dtype)


def _set_card(hdr: fits.Header, key: str, value: Any, comment: Optional[str] = None) -> None:
    """
    FITS-safe setter:
    - Keys > 8 chars are written as HIERARCH <key> by astropy if you pass them literally.
    - Many archives prefer explicit 8-char keys when possible; keep your keys short when you can.
    """
    if value is None:
        return
    if comment is None:
        hdr[key] = value
    else:
        hdr[key] = (value, comment)


def _build_header(c: Camera, cfg: CaptureConfig, data_dtype: np.dtype, shape: Tuple[int, ...]) -> fits.Header:
    hdr = fits.Header()

    # Minimal FITS “documentation” comments (optional; keeps headers friendly for humans/archives)
    hdr["COMMENT"] = "FITS format per NASA/IAU definition; header populated by alpyca capture routine."

    # If we store unsigned 16-bit from a signed source, set BZERO/BSCALE in the conventional way.
    # (astropy will write BITPIX etc; this is the scaling convention.)
    if data_dtype == np.dtype(np.uint16):
        _set_card(hdr, "BZERO", 32768.0, "Data zero point")
        _set_card(hdr, "BSCALE", 1.0, "Data scale factor")

    # Exposure / timing (FITS standard practice)
    # DATE-OBS should be start time of exposure; alpyca provides c.LastExposureStartTime.
    _set_card(hdr, "EXPTIME", float(c.LastExposureDuration), "Exposure time [s]")
    _set_card(hdr, "EXPOSURE", float(c.LastExposureDuration), "Exposure time [s]")
    _set_card(hdr, "DATE-OBS", str(c.LastExposureStartTime), "Exposure start time (UTC recommended)")
    _set_card(hdr, "TIMESYS", "UTC", "Time system")

    # Binning / subframe
    _set_card(hdr, "XBINNING", int(c.BinX), "Binning factor in X")
    _set_card(hdr, "YBINNING", int(c.BinY), "Binning factor in Y")

    # Image type / observing metadata (common archive conventions)
    _set_card(hdr, "IMAGETYP", cfg.imagetyp.upper(), "Image type (LIGHT/DARK/BIAS/FLAT)")
    _set_card(hdr, "OBSTYPE", cfg.imagetyp.upper(), "Image type")

    _set_card(hdr, "OBJECT", cfg.object_name, "Target name")
    _set_card(hdr, "OBSERVER", cfg.observer, "Observer")
    _set_card(hdr, "TELESCOP", cfg.telescope, "Telescope")
    _set_card(hdr, "OBSERVAT", cfg.observatory, "Observatory/site")

    # Instrument identity
    instrume = cfg.instrument if cfg.instrument is not None else getattr(c, "SensorName", None)
    _set_card(hdr, "INSTRUME", instrume, "Instrument/sensor name")

    # Camera settings commonly needed for reduction
    # Not all cameras expose these; keep them best-effort.
    try:
        _set_card(hdr, "GAIN", int(c.Gain), "Camera gain setting")
    except Exception:
        pass

    try:
        off = c.Offset
        if isinstance(off, (int, np.integer)):
            _set_card(hdr, "OFFSET", int(off), "Camera offset setting")
            _set_card(hdr, "PEDESTAL", int(off), "Bias pedestal (if applicable)")
        else:
            _set_card(hdr, "OFFSET", off, "Camera offset setting")
    except Exception:
        pass

    try:
        _set_card(hdr, "CCD-TEMP", float(c.CCDTemperature), "Detector temperature [C]")
    except Exception:
        pass

    # Filter wheel keyword conventions vary; include a simple one if supplied.
    _set_card(hdr, "FILTER", cfg.filter_name, "Filter name")

    # Pointing/airmass (only if you provide it)
    _set_card(hdr, "AIRMASS", cfg.airmass, "Airmass at start")
    _set_card(hdr, "RA", cfg.ra, "Right Ascension (sexagesimal)")
    _set_card(hdr, "DEC", cfg.dec, "Declination (sexagesimal)")
    _set_card(hdr, "HA", cfg.ha, "Hour angle (sexagesimal)")
    if cfg.equinox is not None:
        _set_card(hdr, "EQUINOX", float(cfg.equinox), "Equinox of celestial coordinates")

    # Optional WCS block (insert exactly what you know; do not invent)
    # Typical set: CTYPE*, CUNIT*, CRPIX*, CRVAL*, CD*_* (or PC + CDELT), WCSNAME, RADESYS/RADECSYS.
    for k, v in (cfg.wcs_cards or {}).items():
        _set_card(hdr, k, v)

    # Extra instrument/pipeline-specific cards (your long example header lives here)
    for k, v in (cfg.extra_cards or {}).items():
        _set_card(hdr, k, v)

    # Provenance
    _set_card(hdr, "HISTORY", "Created using Python alpyca-client library + astropy.io.fits")

    return hdr


def capture_fits(cfg: CaptureConfig, out_file: Union[str, Path]) -> Path:
    out_path = Path(out_file).expanduser().resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    c = CameraDevice(cfg.host, cfg.camera_index)
    c.Connected = True
    try:
        # Configure camera/ROI
        c.BinX = int(cfg.binx)
        c.BinY = int(cfg.biny)

        c.StartX = int(cfg.startx)
        c.StartY = int(cfg.starty)

        if cfg.numx is None:
            c.NumX = int(c.CameraXSize // c.BinX)
        else:
            c.NumX = int(cfg.numx)

        if cfg.numy is None:
            c.NumY = int(c.CameraYSize // c.BinY)
        else:
            c.NumY = int(cfg.numy)

        # Start exposure
        c.StartExposure(float(cfg.exposure_s), bool(cfg.is_light))

        t0 = time.time()
        last_pct = None
        while not c.ImageReady:
            if (time.time() - t0) > float(cfg.timeout_s):
                raise TimeoutError(f"Exposure timed out after {cfg.timeout_s} s (ImageReady never became True).")

            if cfg.verbose:
                try:
                    pct = int(c.PercentCompleted)
                    if pct != last_pct:
                        print(f"{pct}% complete")
                        last_pct = pct
                except Exception:
                    pass

            time.sleep(float(cfg.poll_s))

        if cfg.verbose:
            print("finished")

        # Retrieve image and convert
        data, data_dtype = _alpaca_image_to_numpy(c)

        # Build header and write FITS
        hdr = _build_header(c, cfg, data_dtype=data_dtype, shape=data.shape)
        hdu = fits.PrimaryHDU(data=data, header=hdr)
        hdul = fits.HDUList([hdu])

        if cfg.add_checksum:
            hdul.add_checksum()

        hdul.writeto(out_path, overwrite=True, output_verify="fix")

        return out_path

    finally:
        try:
            c.Connected = False
        except Exception:
            pass


if __name__ == "__main__":
    # Example usage: minimal capture
    cfg = CaptureConfig(
        host="localhost:32323",
        camera_index=0,
        exposure_s=2.0,
        is_light=True,
        imagetyp="LIGHT",
        binx=1,
        biny=1,
        object_name="Test",
        observer="python",
        instrument=None,  # let it pull SensorName if available
        add_checksum=True,
        # If you have a plate solution, drop it here (example keys only):
        wcs_cards={
            # "WCSNAME": "Celestial coordinates",
            # "CTYPE1": "RA---TAN", "CTYPE2": "DEC--TAN",
            # "CUNIT1": "deg", "CUNIT2": "deg",
            # "CRPIX1": 512.0, "CRPIX2": 512.0,
            # "CRVAL1": 209.1292724609, "CRVAL2": 38.31526947021,
            # "CD1_1": -0.0001027239995892, "CD1_2": -3.946270226152E-06,
            # "CD2_1":  3.946270226152E-06, "CD2_2": -0.0001027239995892,
            # "RADECSYS": "FK5", "EQUINOX": 2000.0,
        },
        # Inject any instrument/pipeline-specific cards you already compute/know:
        extra_cards={
            # "DATASEC": "[1:1024,1:1024]",
            # "PROGRAM": "YOURDAQ",
            # "VERSION": "v1.0",
        },
    )

    out = capture_fits(cfg, "~/Desktop/alpaca_test.fts")
    print(str(out))
