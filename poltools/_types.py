"""
poltools._types — Core data structures for the polarimetry pipeline.

Frozen configuration dataclasses and a uniform result container
(:class:`StokesResult`, compatible with :class:`caltools.AnalysisResult`).

Conventions
-----------
* Detector arrays use ``[row, col] == [y, x]`` with origin upper-left.
  A source at ``(x, y)`` is stored at ``array[y, x]``.
* Stokes vectors are 4-vectors ``(I, Q, U, V)`` throughout. This pipeline
  measures linear polarization only; a returned *V* of 0 is not a measurement.
* The Savart plate splits each source into an **ordinary** and **extraordinary**
  beam on the detector. The extraordinary beam is defined to carry the analyzer
  +Q′ axis so that the intensity ratio
  ``R = (F_extraordinary − F_ordinary)/(F_extraordinary + F_ordinary)``
  modulates as ``q cos4θ + u sin4θ`` when the half-wave plate (HWP) is ideal.
  The absolute position-angle zero-point is set by standard-star calibration.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Dict, Optional, Tuple

import numpy as np

from caltools import SensorConfig

# Extraordinary-beam position = ordinary-beam position + separation along the
# split direction (position angle), in detector (x, y) pixels.


@dataclass(frozen=True)
class BeamGeometry:
    """Dual-beam geometry produced by the α-BBO Savart plate on the detector.

    The plate is dispersive: beam separation depends on filter bandpass, so
    each filter needs its own :class:`BeamGeometry` (see :class:`FilterConfig`
    and :meth:`PolConfig.for_filter`).

    Parameters
    ----------
    separation_px : float
        Centre-to-centre distance between ordinary and extraordinary beams [px].
    position_angle_deg : float
        Direction from the ordinary to the extraordinary beam in the detector
        frame [degrees].
    """

    separation_px: float
    position_angle_deg: float = 0.0

    def offset_xy(self) -> Tuple[float, float]:
        """Return the (dx, dy) pixel offset from ordinary to extraordinary beam."""
        pa = np.deg2rad(self.position_angle_deg)
        return (self.separation_px * np.sin(pa), self.separation_px * np.cos(pa))


@dataclass(frozen=True)
class FilterConfig:
    """Per-filter calibration for one electronic filter wheel slot.

    Because the Savart split shifts with wavelength, beam separation must be
    measured separately in each band (from flat fields or standard-star beam
    pairs). Using one separation for all filters will mis-pair ordinary and
    extraordinary images when dispersion moves the split by more than a few
    pixels.

    Parameters
    ----------
    name : str
        Filter-wheel slot label (matches the FITS ``FILTER`` keyword).
    beam : BeamGeometry
        Ordinary/extraordinary geometry in this band.
    eff_wavelength_nm : float, optional
        Nominal effective wavelength [nm]; metadata only.
    efficiency : float
        Modulation efficiency in this band (from polarized standard stars).
    is_dark : bool
        True for the blocking (dark) slot.
    characterized : bool
        True once ``beam`` and ``efficiency`` come from calibration data.
    """

    name: str
    beam: BeamGeometry
    eff_wavelength_nm: Optional[float] = None
    efficiency: float = 1.0
    is_dark: bool = False
    characterized: bool = False


def default_efw_filters(separation_px: float = 60.0,
                        position_angle_deg: float = 0.0) -> Tuple[FilterConfig, ...]:
    """Default POLITE ZWO 5-slot filter wheel with placeholder geometry.

    ``beam`` is identical in every slot until measured per band from flat
    fields or standard stars. Nominal Johnson–Cousins wavelengths are labels
    only.
    """
    bg = BeamGeometry(separation_px=separation_px,
                      position_angle_deg=position_angle_deg)
    return (
        FilterConfig("Photometric B", bg, eff_wavelength_nm=440.0),
        FilterConfig("Photometric V", bg, eff_wavelength_nm=551.0),
        FilterConfig("Photometric R", bg, eff_wavelength_nm=640.0),
        FilterConfig("Clear", bg, eff_wavelength_nm=None),
        FilterConfig("Dark", bg, eff_wavelength_nm=None, is_dark=True),
    )


@dataclass(frozen=True)
class PolConfig:
    """Instrument configuration shared by the simulator and reducer.

    Parameters
    ----------
    sensor : SensorConfig
        Detector parameters (gain, size, pixel pitch).
    beam : BeamGeometry
        Active dual-beam split geometry.
    plate_scale_arcsec : float
        Arcsec per pixel.
    hwp_angles_deg : tuple of float
        Half-wave plate angles in the modulation sequence [degrees].
    retardance_deg : float
        Retarder retardance δ [degrees]; 180° for a half-wave plate.
    instrument_rotator_deg : float
        Field-rotator angle α [degrees], upstream of the half-wave plate.
    filter_name : str
        Active filter-wheel slot (matches FITS ``FILTER``).
    filters : tuple of FilterConfig
        Registry of per-filter Savart beam geometry.
    read_noise_e : float
        Gaussian read-noise core [e⁻]. Must match ``readout_mode`` and
        ``gain_setting``. A per-pixel map may be passed to photometry instead.
    dark_rate_e_per_s : float
        Dark current [e⁻ pixel⁻¹ s⁻¹].
    full_well_e : float
        Full-well capacity [e⁻].
    linearity_limit_e : float, optional
        Non-linearity onset [e⁻]; defaults to ``full_well_e``.
    readout_mode : str
        Detector readout mode for noise/gain provenance.
    gain_setting : int
        Gain slider value for noise/gain provenance.
    """

    sensor: SensorConfig
    beam: BeamGeometry
    plate_scale_arcsec: float = 0.224
    hwp_angles_deg: Tuple[float, ...] = (0.0, 22.5, 45.0, 67.5)
    retardance_deg: float = 180.0
    instrument_rotator_deg: float = 0.0
    filter_name: str = "Clear"
    filters: Tuple[FilterConfig, ...] = ()
    read_noise_e: float = 3.5
    dark_rate_e_per_s: float = 0.005
    full_well_e: float = 51000.0
    linearity_limit_e: Optional[float] = None
    readout_mode: str = "Mode0"
    gain_setting: int = 0

    def with_hwp_angles(self, angles) -> "PolConfig":
        """Return a copy with a new half-wave plate angle sequence."""
        return replace(self, hwp_angles_deg=tuple(float(a) for a in angles))

    def active_filter(self) -> Optional[FilterConfig]:
        """Registered :class:`FilterConfig` for ``filter_name``, or ``None``."""
        for f in self.filters:
            if f.name == self.filter_name:
                return f
        return None

    def for_filter(self, name: str) -> "PolConfig":
        """Return a copy configured for filter-wheel slot ``name``.

        Raises ``KeyError`` if ``name`` is not in ``filters``.
        """
        for f in self.filters:
            if f.name == name:
                return replace(self, beam=f.beam, filter_name=f.name)
        raise KeyError(
            f"filter {name!r} not in registry "
            f"{[f.name for f in self.filters]!r}"
        )

    def fwhm_px(self, seeing_arcsec: float) -> float:
        """Convert seeing FWHM [arcsec] to pixels."""
        return seeing_arcsec / self.plate_scale_arcsec

    def sat_limit_e(self) -> float:
        """Saturation/linearity limit [e⁻]."""
        return (self.linearity_limit_e if self.linearity_limit_e is not None
                else self.full_well_e)


@dataclass(frozen=True)
class PointSource:
    """Point source with astrophysical Stokes state.

    Parameters
    ----------
    x, y : float
        Ordinary-beam position [pixels], origin upper-left.
    flux_e : float
        Total intensity *I* [e⁻] over the exposure (partitioned between beams).
    stokes : tuple
        Normalized Stokes ``(1, q, u, v)``.
    name : str
        Source label.
    """

    x: float
    y: float
    flux_e: float
    stokes: Tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0)
    name: str = ""

    @property
    def q(self) -> float:
        return self.stokes[1]

    @property
    def u(self) -> float:
        return self.stokes[2]

    @property
    def p(self) -> float:
        return float(np.hypot(self.stokes[1], self.stokes[2]))

    @property
    def theta_deg(self) -> float:
        return 0.5 * np.rad2deg(np.arctan2(self.stokes[2], self.stokes[1])) % 180.0


@dataclass
class BeamFlux:
    """Ordinary and extraordinary aperture fluxes at one half-wave plate angle.

    Fluxes are in electrons. ``saturated`` is True when an aperture peak reached
    the detector saturation or linearity limit; the beam ratio may then be
    biased.
    """

    hwp_deg: float
    f_o: float
    f_e: float
    sig_o: float = 0.0
    sig_e: float = 0.0
    saturated: bool = False


@dataclass
class StokesResult:
    """Polarimetry result (compatible with :class:`caltools.AnalysisResult`).

    Parameters
    ----------
    name : str
        Source label.
    scalar_summary : dict
        Key scalars: intensity, normalized Stokes ``q, u``, polarization
        fraction ``p``, debiased ``p_mas``, position angle, uncertainties, …
    maps : dict
        Optional 2-D products.
    metadata : dict
        Reduction method, filter, angle count, estimator choice.
    """

    name: str
    scalar_summary: Dict[str, float] = field(default_factory=dict)
    maps: Dict[str, np.ndarray] = field(default_factory=dict)
    metadata: Dict[str, object] = field(default_factory=dict)

    def __repr__(self) -> str:
        s = self.scalar_summary
        p = s.get("p_mas", s.get("p", float("nan")))
        th = s.get("theta_deg", float("nan"))
        sp = s.get("sigma_p", float("nan"))
        return (
            f"StokesResult('{self.name}', p={p:.4%}+/-{sp:.4%}, "
            f"theta={th:.2f} deg, method={self.metadata.get('method', '?')})"
        )
