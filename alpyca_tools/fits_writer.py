from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, Optional, Tuple, Union

import numpy as np
from astropy.io import fits

if TYPE_CHECKING:
    from .camera_device import CameraDevice
    from .camera_ops import ExposureSettings
    from obs_utils.timing import FilterWheelState, TimingProvenance


@dataclass
class DetectorCards:
    """Detector readout provenance (§5.1.1 of first-light implementation plan)."""

    gain_setting: Optional[int] = None          # GAIN — Alpaca slider index
    readout_mode: Optional[int] = None          # READMODE
    readout_mode_name: Optional[str] = None     # RMODE
    offset_setting: Optional[int] = None        # OFFSET
    cooler_setpoint_c: Optional[float] = None    # SET-TEMP
    pixel_size_um: Optional[float] = None       # XPIXSZ / YPIXSZ
    # NB: conversion gain (e-/ADU) and read noise (e-) are intentionally NOT
    # carried here or written to the header. They are per-night characterization
    # results, not as-acquired state; the analyst supplies them at reduction time.


@dataclass
class PolarimetryCards:
    """Optional polarimetry header block (as-acquired HWP + sequence state).

    Mirrors ``poltools.io.POL_KEYWORDS`` so simulated and real POLITE frames carry
    identical cards. All fields optional; only the set ones are written.
    ``INSTROT`` is the **PWI4 field-rotator** angle.

    NB: ``HWPANG`` is the *commanded* angle. The Optec Pyxis Gen3 is open-loop
    with no encoder, so there is nothing to read back; the rotator moves in
    discrete steps and does not land exactly on a round angle. ``WPUNCERT``
    carries that step quantization, which is why it is kept while other constant
    instrument properties are not: it is the uncertainty on ``HWPANG`` in this
    same frame, and a reduction weighting or fitting the modulation curve needs
    it alongside the angle it qualifies.

    Deliberately NOT carried here, on the same reasoning as conversion gain and
    read noise in :class:`DetectorCards` — a header card should record what was
    true of *this frame* at write time, not a spec constant or an analysis result:

    - Savart material/thickness, and the plate's beam separation and position
      angle. Separation is measured, not specified: the manufacturer-nominal
      0.9 mm gives 239.4 px on 3.76 µm pixels where the 2026-07-09 measurement
      gave 238.4 px (= 0.896 mm), and the α-BBO plate is dispersive so the true
      value is per band, not one scalar. Beam geometry lives in the per-session
      ``pol_config.yaml`` sidecar, which can flag it measured vs nominal
      (``beam_geometry_characterized``); a FITS card cannot.
    - HWP retardance. A real half-wave plate is half-wave only at its design
      wavelength, so a nominal 180 would assert an ideal retarder.
    - Modulation efficiency — a reduction result from polarized standards.
    - Filter effective wavelength — a constant per band, recoverable from
      ``FILTER``.
    """

    hwp_angle_deg: Optional[float] = None             # HWPANG (commanded)
    hwp_uncert_deg: Optional[float] = None            # WPUNCERT (open-loop step quant.)
    instrument_rotator_deg: Optional[float] = None    # INSTROT (PWI4 rotator)
    pol_beam: Optional[str] = "dual"                  # POLBEAM (data layout)
    pol_seq_id: Optional[str] = None                  # POLSEQ
    pol_seq_index: Optional[int] = None               # POLSEQN


@dataclass
class FitsHeaderConfig:
    imagetyp: str = "LIGHT"
    # These fields make the writer usable by both live Alpaca capture and the
    # simulator. Live capture normally obtains them from ``camera``; simulated
    # frames must provide them explicitly.
    exposure_s: Optional[float] = None
    date_obs: Optional[str] = None
    binx: int = 1
    biny: int = 1
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
    radesys: Optional[str] = None               # RADESYS (e.g. 'FK5')
    ra_deg: Optional[float] = None              # CRVAL1 — field-center RA [deg]
    dec_deg: Optional[float] = None             # CRVAL2 — field-center Dec [deg]
    alt_deg: Optional[float] = None             # OBJCTALT
    az_deg: Optional[float] = None              # OBJCTAZ
    lst: Optional[str] = None                   # LST (sexagesimal)
    site_lat_deg: Optional[float] = None        # SITELAT [+N]
    site_lon_deg: Optional[float] = None        # SITELONG [+E]
    site_elev_m: Optional[float] = None         # SITEELEV [m]
    focal_len_mm: Optional[float] = None        # FOCALLEN
    focal_ratio: Optional[float] = None         # FOCRATIO
    aperture_mm: Optional[float] = None         # APTDIA
    pixscale: Optional[float] = None            # PIXSCALE [arcsec/pixel]
    origin: Optional[str] = None                # ORIGIN
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


def _stamp_time_cards(hdr: fits.Header, date_obs: Any, exposure_s: Any) -> None:
    """Add MJD-OBS, DATE-END, DATE-AVG, MJD-AVG derived from the start time.

    Defensive: a malformed DATE-OBS or exposure simply skips the derived cards
    rather than aborting the write.
    """
    try:
        from astropy.time import Time, TimeDelta

        t0 = Time(str(date_obs), format="isot", scale="utc")
        dur = float(exposure_s)
        t_mid = t0 + TimeDelta(dur / 2.0, format="sec")
        t_end = t0 + TimeDelta(dur, format="sec")
        _set_card(hdr, "MJD-OBS", float(t0.mjd), "MJD at exposure start (UTC)")
        _set_card(hdr, "DATE-AVG", t_mid.isot, "UTC at exposure midpoint")
        _set_card(hdr, "MJD-AVG", float(t_mid.mjd), "MJD at exposure midpoint (UTC)")
        _set_card(hdr, "DATE-END", t_end.isot, "UTC at exposure end")
    except Exception:
        pass


def _stamp_obsgeo(hdr: fits.Header, lat_deg: Any, lon_deg: Any, elev_m: Any) -> None:
    """Add geocentric OBSGEO-X/Y/Z [m] from geodetic site coordinates."""
    if lat_deg is None or lon_deg is None or elev_m is None:
        return
    try:
        from obs_utils.obs_math import obsgeo_xyz

        xyz = obsgeo_xyz(lat_deg, lon_deg, elev_m)
        if xyz is not None:
            _set_card(hdr, "OBSGEO-X", xyz[0], "Geocentric site X [m]")
            _set_card(hdr, "OBSGEO-Y", xyz[1], "Geocentric site Y [m]")
            _set_card(hdr, "OBSGEO-Z", xyz[2], "Geocentric site Z [m]")
    except Exception:
        pass


def build_header(
    camera: Optional[CameraDevice],
    cfg: FitsHeaderConfig,
    data_dtype: np.dtype,
    shape: Tuple[int, ...],
) -> fits.Header:
    hdr = fits.Header()
    hdr["COMMENT"] = "FITS header populated by alpyca_tools"

    exposure_s = cfg.exposure_s
    if exposure_s is None and camera is not None:
        exposure_s = getattr(camera, "LastExposureDuration", None)
    if exposure_s is None:
        raise ValueError("FITS header requires exposure_s or camera.LastExposureDuration")

    date_obs = cfg.date_obs
    if date_obs is None and camera is not None:
        date_obs = getattr(camera, "LastExposureStartTime", None)
    if date_obs is None:
        raise ValueError("FITS header requires date_obs or camera.LastExposureStartTime")

    if data_dtype == np.dtype(np.uint16):
        _set_card(hdr, "BZERO", 32768.0, "Data zero point")
        _set_card(hdr, "BSCALE", 1.0, "Data scale factor")

    _set_card(hdr, "EXPTIME", float(exposure_s), "Exposure time [s]")
    _set_card(hdr, "EXPOSURE", float(exposure_s), "Exposure time [s]")
    _set_card(hdr, "DATE-OBS", str(date_obs), "Exposure start time")
    _set_card(hdr, "TIMESYS", "UTC", "Time system")
    _stamp_time_cards(hdr, date_obs, exposure_s)
    if cfg.timing is not None:
        from obs_utils.timing import stamp_timing_cards

        stamp_timing_cards(hdr, cfg.timing, cfg.filter_wheel_state)

    _set_card(hdr, "XBINNING", int(getattr(camera, "BinX", cfg.binx)), "Binning factor in X")
    _set_card(hdr, "YBINNING", int(getattr(camera, "BinY", cfg.biny)), "Binning factor in Y")

    _set_card(hdr, "IMAGETYP", cfg.imagetyp.upper(), "Image type (LIGHT/DARK/BIAS/FLAT)")
    _set_card(hdr, "OBSTYPE", cfg.imagetyp.upper(), "Image type")

    _set_card(hdr, "OBJECT", cfg.object_name, "Target name")
    _set_card(hdr, "OBSERVER", cfg.observer, "Observer")
    _set_card(hdr, "TELESCOP", cfg.telescope, "Telescope")
    _set_card(hdr, "OBSERVAT", cfg.observatory, "Observatory/site")
    _set_card(hdr, "ORIGIN", cfg.origin, "Institution/pipeline that wrote this file")

    instrume = cfg.instrument
    if instrume is None and camera is not None:
        instrume = getattr(camera, "SensorName", None)
    _set_card(hdr, "INSTRUME", instrume, "Instrument/sensor name")

    if camera is not None:
        try:
            gain_val = int(camera.Gain)
            if cfg.detector is not None and cfg.detector.gain_setting is not None:
                gain_val = int(cfg.detector.gain_setting)
            _set_card(hdr, "GAIN", gain_val, "Camera gain index, not e-/ADU")
        except Exception:
            pass

    if camera is not None:
        try:
            off = camera.Offset
            if cfg.detector is not None and cfg.detector.offset_setting is not None:
                off = cfg.detector.offset_setting
            if isinstance(off, (int, np.integer)):
                _set_card(hdr, "OFFSET", int(off), "Camera offset setting")
            else:
                _set_card(hdr, "OFFSET", off, "Camera offset setting")
        except Exception:
            pass

    if camera is not None:
        try:
            _set_card(hdr, "DET-TEMP", float(camera.CCDTemperature), "Detector temperature [C]")
        except Exception:
            pass

    det = cfg.detector
    if det is not None:
        _set_card(hdr, "READMODE", det.readout_mode, "Readout mode index")
        _set_card(hdr, "RMODE", det.readout_mode_name, "Readout mode name")
        _set_card(hdr, "SET-TEMP", det.cooler_setpoint_c, "Cooler setpoint [C]")
        if det.pixel_size_um is not None:
            _set_card(hdr, "XPIXSZ", det.pixel_size_um, "Pixel size X [um]")
            _set_card(hdr, "YPIXSZ", det.pixel_size_um, "Pixel size Y [um]")

    # Camera properties are authoritative when no explicit detector card was
    # supplied. Never substitute a model-specific pixel-size or gain default.
    if camera is not None:
        if "XPIXSZ" not in hdr:
            _set_card(hdr, "XPIXSZ", getattr(camera, "PixelSizeX", None), "Pixel size X [um]")
        if "YPIXSZ" not in hdr:
            _set_card(hdr, "YPIXSZ", getattr(camera, "PixelSizeY", None), "Pixel size Y [um]")
        if "READMODE" not in hdr:
            _set_card(hdr, "READMODE", getattr(camera, "ReadoutMode", None), "Readout mode index")

    _set_card(hdr, "FILTER", cfg.filter_name, "Filter name")

    # Optional polarimetry block (mirrors poltools.io so real frames match sim).
    pol = cfg.polarimetry
    if pol is not None:
        _set_card(hdr, "HWPANG", pol.hwp_angle_deg, "Commanded half-wave-plate angle [deg]")
        _set_card(hdr, "WPUNCERT", pol.hwp_uncert_deg, "HWPANG uncertainty, rotator step quant. [deg]")
        _set_card(hdr, "INSTROT", pol.instrument_rotator_deg, "PWI4 field-rotator angle [deg]")
        _set_card(hdr, "POLBEAM", pol.pol_beam, "Beam(s) recorded (o/e/dual)")
        _set_card(hdr, "POLSEQ", pol.pol_seq_id, "Polarimetry sequence identifier")
        _set_card(hdr, "POLSEQN", pol.pol_seq_index, "Index within polarimetry sequence")

    _set_card(hdr, "AIRMASS", cfg.airmass, "Airmass at start")
    _set_card(hdr, "RA", cfg.ra, "Right Ascension (sexagesimal)")
    _set_card(hdr, "DEC", cfg.dec, "Declination (sexagesimal)")
    _set_card(hdr, "HA", cfg.ha, "Hour angle (sexagesimal)")
    if cfg.equinox is not None:
        _set_card(hdr, "EQUINOX", float(cfg.equinox), "Equinox of celestial coordinates")
    _set_card(hdr, "RADESYS", cfg.radesys, "Celestial reference frame")

    # Numeric field-center seed for downstream plate-solving. Deliberately no
    # CTYPE/CD matrix: the dual-beam WCS solution is derived in reduction.
    _set_card(hdr, "CRVAL1", cfg.ra_deg, "Field-center RA seed [deg] (no CTYPE/CD)")
    _set_card(hdr, "CRVAL2", cfg.dec_deg, "Field-center Dec seed [deg] (no CTYPE/CD)")
    _set_card(hdr, "OBJCTALT", cfg.alt_deg, "Telescope altitude [deg]")
    _set_card(hdr, "OBJCTAZ", cfg.az_deg, "Telescope azimuth [deg]")
    _set_card(hdr, "LST", cfg.lst, "Local mean sidereal time (sexagesimal)")

    # Site geolocation
    _set_card(hdr, "SITELAT", cfg.site_lat_deg, "Site latitude [deg, +N]")
    _set_card(hdr, "SITELONG", cfg.site_lon_deg, "Site longitude [deg, +E]")
    _set_card(hdr, "SITEELEV", cfg.site_elev_m, "Site elevation [m]")
    _stamp_obsgeo(hdr, cfg.site_lat_deg, cfg.site_lon_deg, cfg.site_elev_m)

    # Optics. These are manufacturer-nominal, kept because they are standard
    # keywords external tools read as astrometric seeds — not measurements. The
    # true plate scale follows the focal length, which moves with focus and
    # temperature, and comes from an astrometric solution in reduction.
    _set_card(hdr, "FOCALLEN", cfg.focal_len_mm, "Nominal focal length [mm]")
    _set_card(hdr, "FOCRATIO", cfg.focal_ratio, "Nominal focal ratio (f/#)")
    _set_card(hdr, "APTDIA", cfg.aperture_mm, "Aperture diameter [mm]")
    _set_card(hdr, "PIXSCALE", cfg.pixscale, "Nominal plate scale seed [arcsec/pixel]")

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
    from .camera_ops import capture_image

    data, _info, dtype = capture_image(
        camera,
        exposure,
        poll_s=poll_s,
        timeout_s=timeout_s,
    )
    header = build_header(camera, header_cfg, data_dtype=dtype, shape=data.shape)
    return write_fits(out_file, data, header, add_checksum=header_cfg.add_checksum)
