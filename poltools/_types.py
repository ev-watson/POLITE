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
  ``R = (F_e - F_o)/(F_e + F_o) = q cos4θ + u sin4θ`` reproduces the research-map
  formula literally. The absolute sign/zero-point of the position angle is fixed
  by standard-star calibration, so this label choice is immaterial to science.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple

import numpy as np

from caltools import SensorConfig

# Beam offset convention: e_position = o_position + separation_px * (sin PA, cos PA)
# in (x, y) detector coordinates (origin upper-left). PA=0 places e directly
# "below" o (larger row index). Used identically by simulate and photometry.


@dataclass(frozen=True)
class BeamGeometry:
    """Savart / Wollaston dual-beam split geometry on the detector.

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
        4-angle set spanning q and u (Method A); N>=4 supported (Method B).
    retardance_deg : float
        Retarder retardance delta (HWP = 180; 90 reserved for a future QWP).
    instrument_rotator_deg : float
        Instrument/field-rotator angle alpha (deg).
    filter_name : str
        Filter identifier.
    read_noise_e : float
        Detector read noise (e-).
    dark_rate_e_per_s : float
        Dark current (e-/pixel/s).
    full_well_e : float
        Full-well capacity (e-) used for saturation clipping.
    """

    sensor: SensorConfig
    beam: BeamGeometry
    plate_scale_arcsec: float = 0.224
    hwp_angles_deg: Tuple[float, ...] = (0.0, 22.5, 45.0, 67.5)
    retardance_deg: float = 180.0
    instrument_rotator_deg: float = 0.0
    filter_name: str = "Clear"
    read_noise_e: float = 3.5
    dark_rate_e_per_s: float = 0.005
    full_well_e: float = 51000.0

    def with_hwp_angles(self, angles) -> "PolConfig":
        """Return a copy with a new HWP-angle sequence (frozen — no mutation)."""
        return PolConfig(
            sensor=self.sensor,
            beam=self.beam,
            plate_scale_arcsec=self.plate_scale_arcsec,
            hwp_angles_deg=tuple(float(a) for a in angles),
            retardance_deg=self.retardance_deg,
            instrument_rotator_deg=self.instrument_rotator_deg,
            filter_name=self.filter_name,
            read_noise_e=self.read_noise_e,
            dark_rate_e_per_s=self.dark_rate_e_per_s,
            full_well_e=self.full_well_e,
        )

    def fwhm_px(self, seeing_arcsec: float) -> float:
        """Convert a seeing FWHM in arcsec to pixels via the plate scale."""
        return seeing_arcsec / self.plate_scale_arcsec


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
    """

    hwp_deg: float
    f_o: float
    f_e: float
    sig_o: float = 0.0
    sig_e: float = 0.0


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
