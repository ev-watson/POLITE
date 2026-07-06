"""
caltools - Detector characterization tools for astronomical imaging.

Calibration and characterization routines for FITS-based detector data.
Most functions operate on image arrays and basic acquisition metadata;
detector-specific checks are explicit in their function names.

Usage
-----
>>> import caltools as ct
>>> config = ct.sensor_config_from_header("frame.fit", gain=0.5)
>>> bias = ct.master_bias(bias_paths)
>>> rn_map, ts_map = ct.read_noise_map(bias_cube)
"""

__version__ = "0.1.0"

# --- Types ---
from ._types import AnalysisResult, Frame, FrameCube, ROI, SensorConfig

# --- I/O ---
from .io import (
    get_file_index,
    get_timestamps,
    group_by_type_and_exposure,
    load_cube,
    load_cube_chunked,
    load_frame,
    sensor_config_from_header,
)

# --- Stacking ---
from .stacking import master_bias, master_dark, master_flat

# --- Statistics ---
from .stats import (
    WelfordAccumulator,
    gaussianity_tests,
    mad_sigma,
    outlier_mask,
    sigma_vs_mean_2d,
)

# --- Noise ---
from .noise import (
    detect_rtn_pixels,
    dsnu,
    fpn,
    read_noise_map,
    read_noise_spatial,
    row_column_noise,
)

# --- Dark ---
from .dark import (
    dark_current_vs_exposure,
    dark_current_vs_temperature,
    dark_spatial_structure,
    warm_pixel_map,
)

# --- Gain ---
from .gain import (
    full_well_capacity,
    noise_decomposition,
    photon_transfer_curve,
    photon_transfer_curve_with_ron,
)

# --- Linearity ---
from .linearity import linearity_error, linearity_test

# --- PRNU ---
from .prnu import prnu_map

# --- Plotting ---
from .plotting import (
    histogram_gaussian_overlay,
    image_with_colorbar,
    noise_map_with_histogram,
    ptc_plot,
    summary_table,
)

__all__ = [
    # Types
    "SensorConfig",
    "AnalysisResult",
    "Frame",
    "FrameCube",
    "ROI",
    # I/O
    "load_frame",
    "load_cube",
    "load_cube_chunked",
    "sensor_config_from_header",
    "group_by_type_and_exposure",
    "get_timestamps",
    "get_file_index",
    # Stacking
    "master_bias",
    "master_dark",
    "master_flat",
    # Stats
    "WelfordAccumulator",
    "gaussianity_tests",
    "mad_sigma",
    "outlier_mask",
    "sigma_vs_mean_2d",
    # Noise
    "read_noise_map",
    "read_noise_spatial",
    "row_column_noise",
    "dsnu",
    "fpn",
    "detect_rtn_pixels",
    # Dark
    "dark_current_vs_exposure",
    "dark_current_vs_temperature",
    "dark_spatial_structure",
    "warm_pixel_map",
    # Gain
    "photon_transfer_curve",
    "photon_transfer_curve_with_ron",
    "full_well_capacity",
    "noise_decomposition",
    # Linearity
    "linearity_test",
    "linearity_error",
    # PRNU
    "prnu_map",
    # Plotting
    "image_with_colorbar",
    "ptc_plot",
    "histogram_gaussian_overlay",
    "noise_map_with_histogram",
    "summary_table",
]
