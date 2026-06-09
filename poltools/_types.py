"""
poltools._types — Core data structures for the POLITE polarimetry pipeline.

Mirrors the caltools data-model style (frozen config dataclasses + a uniform
result container compatible with ``caltools.AnalysisResult``).

Conventions (fixed once, used everywhere)
-----------------------------------------
* Detector arrays are indexed ``[row, col] == [y, x]`` with the **origin in the
  upper-left** (CLAUDE.md): row 0 is the top, col 0 is the left. A source at
  detector coordinate ``(x, y)`` lives at ``array[y, x]``.
* Stokes vectors are full 4-vectors ``(I, Q, U, V)`` everywhere so a future
  quarter-wave-plate / Stokes-V mode drops in without rework. The *linear*
  pipeline (single HWP) never solves for V — a returned V of 0 is **not** a
  measurement (a single HWP cannot constrain V).
* The two analyzed beams of the Savart plate are labelled ``o`` (ordinary) and
  ``e`` (extraordinary). By convention here the **e-beam carries the analyzer
  +Q' axis** so that the double-difference ratio
  ``R = (F_e - F_o)/(F_e + F_o) = q cos4θ + u sin4θ`` reproduces the ideal-HWP
  modulation formula literally. The absolute sign/zero-point of the position
  angle is fixed
  by standard-star calibration, so this label choice is immaterial to science.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Dict, Optional, Tuple

import numpy as np

from caltools import SensorConfig

# Beam offset convention: e_position = o_position + separation_px * (sin PA, cos PA)
# in (x, y) detector coordinates (origin upper-left). PA=0 places e directly
# "below" o (larger row index). Used identically by simulate and photometry.


@dataclass(frozen=True)
class BeamGeometry:
    """α-BBO Savart-plate dual-beam split geometry on the detector.

    The split is **dispersive** (α-BBO birefringence), so a separate
    :class:`BeamGeometry` belongs to each EFW band — see :class:`FilterConfig`
    and :meth:`PolConfig.for_filter`.

    Parameters
    ----------
    separation_px : float
        Centre-to-centre o<->e beam separation in pixels.
    position_angle_deg : float
        Direction of the o->e split vector in the detector frame (deg).
    """

    separation_px: float
    position_angle_deg: float = 0.0

    def offset_xy(self) -> Tuple[float, float]:
        """Return the (dx, dy) o->e offset in detector (x, y) pixels."""
        pa = np.deg2rad(self.position_angle_deg)
        return (self.separation_px * np.sin(pa), self.separation_px * np.cos(pa))


@dataclass(frozen=True)
class FilterConfig:
    """Per-filter optical calibration for one ZWO EFW slot.

    The α-BBO Savart plate is birefringent and **dispersive**, so the ordinary/
    extraordinary beam separation (and, at second order, its position angle)
    differs from band to band. The split geometry is therefore stored **per
    filter** and is **measured from flats / standard-star o<->e pairs (Source
    A)** — it is **not** computed from a dispersion model here (a single fixed
    separation across all bands would mis-pair o/e in :func:`pair_oe`, since a
    few-percent dispersion of a ~60-px split exceeds the pairing tolerance).
    Until a slot is characterized it carries a placeholder ``beam`` and
    ``characterized=False``.

    Parameters
    ----------
    name : str
        EFW slot label, matching the FITS ``FILTER`` card (e.g. "Photometric V").
    beam : BeamGeometry
        Savart o<->e split geometry **in this band**.
    eff_wavelength_nm : float, optional
        Effective wavelength of the band (label / provenance only; not used to
        compute any optical quantity).
    efficiency : float
        Polarization (modulation) efficiency in this band, calibrated from a
        polarized standard. Default 1.0.
    is_dark : bool
        True for the blocking / "Dark" slot (no light reaches the detector).
    characterized : bool
        True once ``beam`` / ``efficiency`` come from real calibration data;
        ``False`` marks them as placeholders.
    """

    name: str
    beam: BeamGeometry
    eff_wavelength_nm: Optional[float] = None
    efficiency: float = 1.0
    is_dark: bool = False
    characterized: bool = False


def default_efw_filters(separation_px: float = 60.0,
                        position_angle_deg: float = 0.0) -> Tuple[FilterConfig, ...]:
    """The POLITE ZWO 5-slot EFW as placeholder (un-characterized) slots.

    Every slot shares the same placeholder :class:`BeamGeometry` until the
    per-band α-BBO beam separation is measured from flats / standards (Source A);
    each is flagged ``characterized=False``. Effective wavelengths are nominal
    Johnson–Cousins band centres (label / provenance only).
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
    """Instrument configuration for the POLITE polarimeter (sim + reduction).

    Single source of truth shared by the forward model and the reducer.

    Parameters
    ----------
    sensor : caltools.SensorConfig
        Detector parameters (gain, size, pixel pitch, ...).
    beam : BeamGeometry
        Dual-beam split geometry.
    plate_scale_arcsec : float
        Arcsec per pixel (QHY268M on CDK20 ~ 0.224).
    hwp_angles_deg : tuple of float
        HWP positions in the modulation sequence (deg). Default is the minimal
        4-angle set spanning q and u (double_ratio / double_difference); N>=4
        supported (lsq_modulation).
    retardance_deg : float
        Retarder retardance delta (HWP = 180; 90 reserved for a future QWP).
    instrument_rotator_deg : float
        **PWI4 Focuser/Rotator** field angle alpha (deg). The PWI4 rotator sits
        in the chain *after* the CDK20 and *upstream* of the L3 cut filter and
        the HWP, so it rotates the whole instrument (HWP/EFW/Savart) relative to
        the sky+telescope frame. Modelled as a frame rotation applied to the
        incident Stokes vector (see :func:`poltools.mueller.oe_intensities`).
    filter_name : str
        Active EFW slot identifier; matches a ``name`` in ``filters`` and the
        FITS ``FILTER`` card.
    filters : tuple of FilterConfig
        Registry of the ZWO EFW slots with their **per-band** Savart geometry
        (the α-BBO split is dispersive). Empty by default (single-filter use via
        ``beam``); populate with :func:`default_efw_filters` and select a band
        with :meth:`for_filter`.
    read_noise_e : float
        Detector read noise (e-). For the QHY268M this is **gain-mode-specific**
        (HCG/LCG and the gain setting change it strongly); it must match
        ``readout_mode``/``gain_setting`` below. The default 3.5 e- is the
        characterized Mode 0 / Gain 0 value (project A/B; ``reduction.md``). The
        scalar describes the Gaussian read-noise *core*; a per-pixel read-noise
        map (RTN/hot-pixel tail) can be supplied directly to ``measure_fluxes``.
    dark_rate_e_per_s : float
        Dark current (e-/pixel/s).
    full_well_e : float
        Full-well capacity (e-) used for saturation clipping.
    linearity_limit_e : float, optional
        Onset of the non-linearity rolloff (e-). If set, photometry flags any
        aperture whose peak exceeds it. Defaults to ``full_well_e`` when unset.
    readout_mode : str
        QHY268M readout mode the noise/gain values correspond to (e.g.
        "Mode0"). TheSkyX writes no GAIN keyword, so the acquisition mode is
        tracked here as provenance — ``read_noise_e`` and ``sensor.gain_e_per_adu``
        must be sourced from the PTC characterization at *this* mode
        (``reduction.md`` §3.1).
    gain_setting : int
        QHY268M gain slider value the noise/gain values correspond to.
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
        """Return a copy with a new HWP-angle sequence (frozen — no mutation).

        Uses ``dataclasses.replace`` so every other field (including the
        gain-mode provenance) is carried over verbatim.
        """
        return replace(self, hwp_angles_deg=tuple(float(a) for a in angles))

    def active_filter(self) -> Optional[FilterConfig]:
        """Return the registered :class:`FilterConfig` matching ``filter_name``.

        ``None`` when the filter is not in the ``filters`` registry (single-
        filter configs that only use ``beam`` directly).
        """
        for f in self.filters:
            if f.name == self.filter_name:
                return f
        return None

    def for_filter(self, name: str) -> "PolConfig":
        """Return a copy configured for EFW slot ``name``.

        Applies that band's **per-filter** Savart geometry (``beam``) and sets
        ``filter_name`` via :func:`dataclasses.replace` (frozen — no mutation),
        mirroring :meth:`with_hwp_angles`. Every other field (gain-mode
        provenance, the full ``filters`` registry, ...) carries over verbatim, so
        the returned config drives ``simulate``/``photometry``/``pipeline`` in
        that band with no further changes. Raises ``KeyError`` if ``name`` is not
        registered.
        """
        for f in self.filters:
            if f.name == name:
                return replace(self, beam=f.beam, filter_name=f.name)
        raise KeyError(
            f"filter {name!r} not in registry "
            f"{[f.name for f in self.filters]!r}"
        )

    def fwhm_px(self, seeing_arcsec: float) -> float:
        """Convert a seeing FWHM in arcsec to pixels via the plate scale."""
        return seeing_arcsec / self.plate_scale_arcsec

    def sat_limit_e(self) -> float:
        """Saturation/linearity limit (e-): ``linearity_limit_e`` or full well."""
        return (self.linearity_limit_e if self.linearity_limit_e is not None
                else self.full_well_e)


@dataclass(frozen=True)
class PointSource:
    """A point source with an astrophysical Stokes state.

    Parameters
    ----------
    x, y : float
        Ordinary-beam detector position (px), origin upper-left.
    flux_e : float
        Total source intensity flux I in electrons over the exposure
        (summed over both o and e beams; the beams partition this flux).
    stokes : tuple
        Normalized Stokes ``(1, q, u, v)`` with q,u,v in [-1, 1].
    name : str
        Optional label.
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
    """Measured o/e fluxes for one source at one HWP angle.

    Fluxes and uncertainties are in **electrons** (a consistent linear unit;
    the double-difference ratio is unit-independent).

    ``saturated`` flags that the o- or e-aperture peak reached the detector
    saturation/linearity limit at this angle, so the beam ratio (hence q,u) may
    be biased — surfaced by the pipeline rather than silently reduced.
    """

    hwp_deg: float
    f_o: float
    f_e: float
    sig_o: float = 0.0
    sig_e: float = 0.0
    saturated: bool = False


@dataclass
class StokesResult:
    """Uniform polarimetry result (compatible with caltools.AnalysisResult).

    Parameters
    ----------
    name : str
        Source / analysis label.
    scalar_summary : dict
        Key scalars: ``I, q, u, p, p_mas, theta_deg, sigma_p, sigma_theta_deg,
        snr, chi2`` ...
    maps : dict
        Optional 2-D products (e.g. vector overlays).
    metadata : dict
        Method, n_angles, estimator, and Source A/B provenance.
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
