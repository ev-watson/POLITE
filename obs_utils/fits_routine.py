"""
Alpaca (alpyca) camera -> FITS capture routine with observing metadata.

Notes:
- FITS required structural keywords (SIMPLE/BITPIX/NAXIS/NAXISn) are written by astropy.
- This adds observational, instrument, and optional WCS metadata.
- Instrument- and pipeline-specific keywords can be supplied through
  `extra_cards` without hard-coding a single camera schema.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, Optional, Tuple, Union

import numpy as np
from astropy.io import fits

from obs_utils.timing import FilterWheelState, TimingProvenance, stamp_timing_cards

if TYPE_CHECKING:
    from alpyca_tools.camera_device import CameraDevice


@dataclass
class DetectorCards:
    """Detector readout provenance (mirrors alpyca_tools.fits_writer.DetectorCards)."""

    gain_setting: Optional[int] = None
    egain_e_per_adu: Optional[float] = None
    readout_mode: Optional[int] = None
    readout_mode_name: Optional[str] = None
    offset_setting: Optional[int] = None
    cooler_setpoint_c: Optional[float] = None
    pixel_size_um: Optional[float] = None
    ron_e: Optional[float] = None


@dataclass
class PolarimetryCards:
    """Optional polarimetry header block (HWP + α-BBO Savart dual-beam metadata).

    Mirrors ``poltools.io.POL_KEYWORDS`` so simulated and real POLITE frames carry
    identical cards (the keyword/grouping is what the reduction path reads). All
    fields are optional; only the set ones are written. ``INSTROT`` is the **PWI4
    field-rotator** angle. The Savart material/thickness default to the installed
    α-BBO 18 mm plate; the per-band ``BEAMSEP``/``BEAMPA`` are measured from flats.
    """

    hwp_angle_deg: Optional[float] = None             # HWPANG
    retardance_deg: Optional[float] = None            # RETARD (HWP nominal 180)
    instrument_rotator_deg: Optional[float] = None    # INSTROT (PWI4 rotator)
    pol_beam: Optional[str] = "dual"                  # POLBEAM
    pol_seq_id: Optional[str] = None                  # POLSEQ
    pol_seq_index: Optional[int] = None               # POLSEQN
    pol_efficiency: Optional[float] = None            # POLEFF
    beam_sep_px: Optional[float] = None               # BEAMSEP
    beam_pa_deg: Optional[float] = None               # BEAMPA
    savart_material: Optional[str] = "alpha-BBO"      # SAVMAT
    savart_thickness_mm: Optional[float] = 18.0       # SAVTHK
    eff_wavelength_nm: Optional[float] = None          # WAVELEN


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

    # Optional polarimetry block (HWP + α-BBO Savart). None for ordinary imaging.
    polarimetry: Optional["PolarimetryCards"] = None
    detector: Optional[DetectorCards] = None

    # Extra instrument/pipeline cards.
    extra_cards: Dict[str, Any] = field(default_factory=dict)

    timing: Optional[TimingProvenance] = None
    filter_wheel_state: Optional[FilterWheelState] = None

    # If you want FITS CHECKSUM/DATASUM
    add_checksum: bool = True


def _alpaca_image_to_numpy(c: CameraDevice) -> Tuple[np.ndarray, np.dtype]:
    """
    Retrieve the exposed image as a numpy array with astropy-compatible axis order.

    Delegates to ``alpyca_tools.camera_ops.download_image``, which decodes the
    Alpaca ImageBytes stream directly via ``np.frombuffer``. That avoids
    ``alpaca.camera``'s Int32 reconstruction bug on 64-bit macOS/Linux (it uses
    ``array.array('l')``, which is 8 bytes there rather than the assumed 4).
    """
    from alpyca_tools.camera_ops import download_image

    data, _info, dtype = download_image(c)
    return data, dtype


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


def _build_header(c: CameraDevice, cfg: CaptureConfig, data_dtype: np.dtype, shape: Tuple[int, ...]) -> fits.Header:
    hdr = fits.Header()

    # Minimal provenance comment.
    hdr["COMMENT"] = "FITS format per NASA/IAU definition; header populated by alpyca capture routine."

    # If we store unsigned 16-bit from a signed source, set BZERO/BSCALE in the conventional way.
    # (astropy will write BITPIX etc; this is the scaling convention.)
    if data_dtype == np.dtype(np.uint16):
        _set_card(hdr, "BZERO", 32768.0, "Data zero point")
        _set_card(hdr, "BSCALE", 1.0, "Data scale factor")

    # Exposure / timing
    # DATE-OBS should be start time of exposure; alpyca provides c.LastExposureStartTime.
    _set_card(hdr, "EXPTIME", float(c.LastExposureDuration), "Exposure time [s]")
    _set_card(hdr, "EXPOSURE", float(c.LastExposureDuration), "Exposure time [s]")
    _set_card(hdr, "DATE-OBS", str(c.LastExposureStartTime), "Exposure start time (UTC recommended)")
    _set_card(hdr, "TIMESYS", "UTC", "Time system")
    if cfg.timing is not None:
        stamp_timing_cards(hdr, cfg.timing, cfg.filter_wheel_state)

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
        gain_val = int(c.Gain)
        if cfg.detector is not None and cfg.detector.gain_setting is not None:
            gain_val = int(cfg.detector.gain_setting)
        _set_card(hdr, "GAIN", gain_val, "QHY gain index, not e-/ADU")
    except Exception:
        pass

    try:
        off = c.Offset
        if cfg.detector is not None and cfg.detector.offset_setting is not None:
            off = cfg.detector.offset_setting
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

    det = cfg.detector
    if det is not None:
        _set_card(hdr, "EGAIN", det.egain_e_per_adu, "Conversion gain [e-/ADU]")
        _set_card(hdr, "READMODE", det.readout_mode, "Readout mode index")
        _set_card(hdr, "RMODE", det.readout_mode_name, "Readout mode name")
        _set_card(hdr, "SET-TEMP", det.cooler_setpoint_c, "Cooler setpoint [C]")
        if det.pixel_size_um is not None:
            _set_card(hdr, "XPIXSZ", det.pixel_size_um, "Pixel size X [um]")
            _set_card(hdr, "YPIXSZ", det.pixel_size_um, "Pixel size Y [um]")
        _set_card(hdr, "RON", det.ron_e, "Read noise [e-]")

    # Filter wheel keyword conventions vary; include a simple one if supplied.
    _set_card(hdr, "FILTER", cfg.filter_name, "Filter name")

    # Optional polarimetry block (mirrors poltools.io so real frames match sim).
    pol = cfg.polarimetry
    if pol is not None:
        _set_card(hdr, "HWPANG", pol.hwp_angle_deg, "Half-wave-plate angle [deg]")
        _set_card(hdr, "RETARD", pol.retardance_deg, "Retarder retardance delta [deg]")
        _set_card(hdr, "INSTROT", pol.instrument_rotator_deg, "PWI4 field-rotator angle [deg]")
        _set_card(hdr, "POLBEAM", pol.pol_beam, "Beam(s) recorded (o/e/dual)")
        _set_card(hdr, "POLSEQ", pol.pol_seq_id, "Polarimetry sequence identifier")
        _set_card(hdr, "POLSEQN", pol.pol_seq_index, "Index within polarimetry sequence")
        _set_card(hdr, "POLEFF", pol.pol_efficiency, "Polarization (modulation) efficiency")
        _set_card(hdr, "BEAMSEP", pol.beam_sep_px, "o<->e beam separation [px]")
        _set_card(hdr, "BEAMPA", pol.beam_pa_deg, "o->e split position angle [deg]")
        _set_card(hdr, "SAVMAT", pol.savart_material, "Savart-plate material")
        _set_card(hdr, "SAVTHK", pol.savart_thickness_mm, "Savart-plate thickness [mm]")
        _set_card(hdr, "WAVELEN", pol.eff_wavelength_nm, "Filter effective wavelength [nm]")

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
    from alpyca_tools.camera_device import CameraDevice

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

        hdul.writeto(out_path, overwrite=True, output_verify="fix", checksum=cfg.add_checksum)

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
