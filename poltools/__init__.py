"""
poltools — Imaging polarimetry pipeline for the POLITE observatory.

A telescope-chain **forward model** (simulator) + **Stokes-extraction pipeline**
with publication-grade error metrics, for the dual-beam (HWP + Savart) imaging
polarimeter. Linear (I, Q, U) now; full-Stokes-ready (4-vector / general
retarder) architecture for a future quarter-wave-plate Stokes-V mode.

Design: a sibling of ``caltools`` (import graph ``poltools → caltools`` only),
reusing ``caltools`` I/O + detector noise model and the ``AnalysisResult``
contract (here :class:`StokesResult`).

All reduction/error methods are from peer-reviewed / user-provided sources
(CLAUDE.md Sources A/B); each function's docstring cites its specific source.

Quick start
-----------
>>> import poltools as pt
>>> cfg = pt.PolConfig(sensor=sensor, beam=pt.BeamGeometry(separation_px=20))
>>> paths = pt.simulate_sequence(scene, cfg, out_dir="FITSDATA/SIM", rng=rng,
...                              shape=(512, 512))
>>> results = pt.reduce_to_stokes([str(p) for p in paths], cfg,
...                               o_positions=positions, method="double_ratio")
"""

__version__ = "0.1.0"

# --- Types ---
from ._types import (
    BeamFlux,
    BeamGeometry,
    PointSource,
    PolConfig,
    StokesResult,
)

# --- Mueller forward model ---
from .mueller import (
    M_hwp,
    M_linear_polarizer,
    M_retarder,
    M_rotator,
    oe_intensities,
    stokes_vector,
    system_mueller,
)

# --- Simulator ---
from .simulate import make_scene, render_frame, simulate_sequence

# --- I/O ---
from .io import (
    group_by_hwp_angle,
    group_pol_sequence,
    read_pol_frame,
    write_pol_fits,
)

# --- Photometry ---
from .photometry import (
    detect_sources,
    measure_fluxes,
    measure_pair,
    pair_oe,
    photometer_sequence,
)

# --- Modulation → q,u ---
from .modulation import (
    double_difference,
    double_ratio,
    lsq_modulation,
    ratio_r,
)

# --- Stokes assembly ---
from .stokes import assemble_stokes, polarization_fraction_angle

# --- Errors / debiasing ---
from .errors import (
    debias_mas,
    debias_naive,
    debias_wardle_kronberg,
    residual_sigma_p,
    sigma_p_mas,
    sigma_theta_highsnr,
    sigma_theta_nkc,
)

# --- Calibration ---
from .calibration import (
    PolCalibration,
    fit_efficiency,
    fit_instrumental_polarization,
    fit_pa_zeropoint,
)

# --- Pipeline ---
from .pipeline import reduce_to_stokes

__all__ = [
    "__version__",
    # types
    "PolConfig", "BeamGeometry", "PointSource", "BeamFlux", "StokesResult",
    # mueller
    "stokes_vector", "M_rotator", "M_retarder", "M_hwp", "M_linear_polarizer",
    "system_mueller", "oe_intensities",
    # simulate
    "render_frame", "simulate_sequence", "make_scene",
    # io
    "write_pol_fits", "read_pol_frame", "group_by_hwp_angle", "group_pol_sequence",
    # photometry
    "detect_sources", "pair_oe", "measure_fluxes", "measure_pair",
    "photometer_sequence",
    # modulation
    "ratio_r", "double_ratio", "double_difference", "lsq_modulation",
    # stokes
    "assemble_stokes", "polarization_fraction_angle",
    # errors
    "residual_sigma_p", "debias_naive", "debias_wardle_kronberg", "debias_mas",
    "sigma_p_mas", "sigma_theta_highsnr", "sigma_theta_nkc",
    # calibration
    "PolCalibration", "fit_instrumental_polarization", "fit_pa_zeropoint",
    "fit_efficiency",
    # pipeline
    "reduce_to_stokes",
]
