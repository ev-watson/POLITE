"""
poltools — Imaging polarimetry for dual-beam half-wave plate polarimeters.

Provides a forward model (Mueller calculus plus detector simulation) and a
reduction pipeline that extracts linear Stokes parameters with published error
estimators. Developed for the POLITE observatory (PlaneWave CDK20, QHY268M,
α-BBO Savart analyzer) but applicable to any single-detector polarimeter that
records ordinary and extraordinary beams while stepping a half-wave plate.

Optical chain modelled::

    Sky → telescope → field rotator → half-wave plate → filter wheel
        → α-BBO Savart plate → detector

The Savart plate splits each source into two images whose separation depends on
filter bandpass. Register per-filter geometry with :func:`default_efw_filters`
and select a band with :meth:`PolConfig.for_filter`.

Depends on ``caltools`` for FITS I/O and :class:`~caltools.SensorConfig`.
"""

__version__ = "0.1.0"

from ._types import (
    BeamFlux,
    BeamGeometry,
    FilterConfig,
    PointSource,
    PolConfig,
    StokesResult,
    default_efw_filters,
)
from .mueller import (
    M_hwp,
    M_linear_polarizer,
    M_retarder,
    M_rotator,
    oe_intensities,
    stokes_vector,
    system_mueller,
)
from .simulate import make_scene, render_frame, simulate_sequence
from .io import (
    group_by_filter,
    group_by_hwp_angle,
    group_pol_sequence,
    load_pol_config_sidecar,
    read_pol_frame,
    write_pol_fits,
)
from .pol_config import (
    SessionDetectorConfig,
    polconfig_from_detector,
    polconfig_from_fits_headers,
    polconfig_snapshot,
    write_pol_config_sidecar,
)
from .photometry import (
    aperture_peaks,
    detect_sources,
    measure_fluxes,
    measure_pair,
    pair_oe,
    photometer_sequence,
)
from .modulation import (
    double_difference,
    double_ratio,
    lsq_modulation,
    ratio_r,
)
from .stokes import assemble_stokes, polarization_fraction_angle
from .errors import (
    debias_mas,
    debias_naive,
    debias_wardle_kronberg,
    residual_sigma_p,
    sigma_theta_highsnr,
    sigma_theta_nkc,
)
from .calibration import (
    PolCalibration,
    fit_efficiency,
    fit_instrumental_polarization,
    fit_pa_zeropoint,
)
from .pipeline import reduce_to_stokes

__all__ = [
    "__version__",
    "PolConfig", "BeamGeometry", "FilterConfig", "default_efw_filters",
    "PointSource", "BeamFlux", "StokesResult",
    "stokes_vector", "M_rotator", "M_retarder", "M_hwp", "M_linear_polarizer",
    "system_mueller", "oe_intensities",
    "render_frame", "simulate_sequence", "make_scene",
    "write_pol_fits", "read_pol_frame", "group_by_hwp_angle", "group_by_filter",
    "group_pol_sequence", "load_pol_config_sidecar", "write_pol_config_sidecar",
    "SessionDetectorConfig", "polconfig_from_detector", "polconfig_from_fits_headers",
    "polconfig_snapshot",
    "detect_sources", "pair_oe", "measure_fluxes", "measure_pair",
    "photometer_sequence", "aperture_peaks",
    "ratio_r", "double_ratio", "double_difference", "lsq_modulation",
    "assemble_stokes", "polarization_fraction_angle",
    "residual_sigma_p", "debias_naive", "debias_wardle_kronberg", "debias_mas",
    "sigma_theta_highsnr", "sigma_theta_nkc",
    "PolCalibration", "fit_instrumental_polarization", "fit_pa_zeropoint",
    "fit_efficiency",
    "reduce_to_stokes",
]
