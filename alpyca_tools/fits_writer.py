from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union

import numpy as np
from astropy.io import fits

from obs_utils.timing import FilterWheelState, TimingProvenance, stamp_timing_cards

from .camera_device import CameraDevice
from .camera_ops import ExposureSettings, capture_image


@dataclass
class DetectorCards:
    """Detector readout provenance (§5.1.1 of first-light implementation plan)."""

    gain_setting: Optional[int] = None          # GAIN — Alpaca slider index
    egain_e_per_adu: Optional[float] = None     # EGAIN — measured e-/ADU
    readout_mode: Optional[int] = None          # READMODE
    readout_mode_name: Optional[str] = None     # RMODE
    offset_setting: Optional[int] = None        # OFFSET
    cooler_setpoint_c: Optional[float] = None    # SET-TEMP
    pixel_size_um: Optional[float] = None       # XPIXSZ / YPIXSZ
    ron_e: Optional[float] = None               # RON (optional)


@dataclass
class PolarimetryCards:
    """Optional polarimetry header block (HWP + α-BBO Savart dual-beam metadata).

    Mirrors ``poltools.io.POL_KEYWORDS`` so simulated and real POLITE frames carry
    identical cards. All fields optional; only the set ones are written.
    ``INSTROT`` is the **PWI4 field-rotator** angle; the Savart material/thickness
    default to the installed α-BBO 18 mm plate; per-band ``BEAMSEP``/``BEAMPA`` are
    measured from flats.
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
class FitsHeaderConfig:
    imagetyp: str = "LIGHT"
    object_name: Optional[str] = None
    observer: Optional[str] = None
    telescope: Optional[str] = None
    observatory: Optional[str] = None
    instrument: Optional[str] = None
    filter_name: Optional[str] = None
    airmass: Optional[float] = None
    ra: Optional[str] = None
    dec: Optional[str] = None
    ha: Optional[str] = None
    equinox: Optional[float] = 2000.0
    polarimetry: Optional[PolarimetryCards] = None
    detector: Optional[DetectorCards] = None
    wcs_cards: Dict[str, Any] = field(default_factory=dict)
    extra_cards: Dict[str, Any] = field(default_factory=dict)
    timing: Optional[TimingProvenance] = None
    filter_wheel_state: Optional[FilterWheelState] = None
    add_checksum: bool = True


def _set_card(hdr: fits.Header, key: str, value: Any, comment: Optional[str] = None) -> None:
    if value is None:
        return
    if comment is None:
        hdr[key] = value
    else:
        hdr[key] = (value, comment)


def build_header(
    camera: CameraDevice,
    cfg: FitsHeaderConfig,
    data_dtype: np.dtype,
    shape: Tuple[int, ...],
) -> fits.Header:
    hdr = fits.Header()
    hdr["COMMENT"] = "FITS header populated by alpyca_tools"

    if data_dtype == np.dtype(np.uint16):
        _set_card(hdr, "BZERO", 32768.0, "Data zero point")
        _set_card(hdr, "BSCALE", 1.0, "Data scale factor")

    _set_card(hdr, "EXPTIME", float(camera.LastExposureDuration), "Exposure time [s]")
    _set_card(hdr, "EXPOSURE", float(camera.LastExposureDuration), "Exposure time [s]")
    _set_card(hdr, "DATE-OBS", str(camera.LastExposureStartTime), "Exposure start time")
    _set_card(hdr, "TIMESYS", "UTC", "Time system")
    if cfg.timing is not None:
        stamp_timing_cards(hdr, cfg.timing, cfg.filter_wheel_state)

    _set_card(hdr, "XBINNING", int(camera.BinX), "Binning factor in X")
    _set_card(hdr, "YBINNING", int(camera.BinY), "Binning factor in Y")

    _set_card(hdr, "IMAGETYP", cfg.imagetyp.upper(), "Image type (LIGHT/DARK/BIAS/FLAT)")
    _set_card(hdr, "OBSTYPE", cfg.imagetyp.upper(), "Image type")

    _set_card(hdr, "OBJECT", cfg.object_name, "Target name")
    _set_card(hdr, "OBSERVER", cfg.observer, "Observer")
    _set_card(hdr, "TELESCOP", cfg.telescope, "Telescope")
    _set_card(hdr, "OBSERVAT", cfg.observatory, "Observatory/site")

    instrume = cfg.instrument if cfg.instrument is not None else getattr(camera, "SensorName", None)
    _set_card(hdr, "INSTRUME", instrume, "Instrument/sensor name")

    try:
        gain_val = int(camera.Gain)
        if cfg.detector is not None and cfg.detector.gain_setting is not None:
            gain_val = int(cfg.detector.gain_setting)
        _set_card(hdr, "GAIN", gain_val, "QHY gain index, not e-/ADU")
    except Exception:
        pass

    try:
        off = camera.Offset
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
        _set_card(hdr, "CCD-TEMP", float(camera.CCDTemperature), "Detector temperature [C]")
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

    _set_card(hdr, "AIRMASS", cfg.airmass, "Airmass at start")
    _set_card(hdr, "RA", cfg.ra, "Right Ascension (sexagesimal)")
    _set_card(hdr, "DEC", cfg.dec, "Declination (sexagesimal)")
    _set_card(hdr, "HA", cfg.ha, "Hour angle (sexagesimal)")
    if cfg.equinox is not None:
        _set_card(hdr, "EQUINOX", float(cfg.equinox), "Equinox of celestial coordinates")

    for k, v in (cfg.wcs_cards or {}).items():
        _set_card(hdr, k, v)

    for k, v in (cfg.extra_cards or {}).items():
        _set_card(hdr, k, v)

    _set_card(hdr, "HISTORY", "Created using alpyca_tools + astropy.io.fits")

    return hdr


def write_fits(
    out_file: Union[str, Path],
    data: np.ndarray,
    header: fits.Header,
    add_checksum: bool = True,
) -> Path:
    out_path = Path(out_file).expanduser().resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    hdu = fits.PrimaryHDU(data=data, header=header)
    hdul = fits.HDUList([hdu])

    hdul.writeto(out_path, overwrite=True, output_verify="fix", checksum=add_checksum)
    return out_path


def capture_fits(
    camera: CameraDevice,
    exposure: ExposureSettings,
    header_cfg: FitsHeaderConfig,
    out_file: Union[str, Path],
    poll_s: float = 0.5,
    timeout_s: float = 300.0,
) -> Path:
    data, _info, dtype = capture_image(
        camera,
        exposure,
        poll_s=poll_s,
        timeout_s=timeout_s,
    )
    header = build_header(camera, header_cfg, data_dtype=dtype, shape=data.shape)
    return write_fits(out_file, data, header, add_checksum=header_cfg.add_checksum)
