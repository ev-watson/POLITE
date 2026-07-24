"""
caltools — Detector characterization for astronomical imaging sensors.

Routines to build master calibration frames, measure read noise and dark
current, determine conversion gain from the photon transfer curve, test
linearity, and map pixel-to-pixel sensitivity variation. Developed for the
POLITE QHY268M (Sony IMX571) but usable for any FITS-based imaging detector.
"""

__version__ = "0.1.0"

from ._types import AnalysisResult, Frame, FrameCube, ROI, SensorConfig

from .io import (
    get_timestamps,
    group_by_type_and_exposure,
    load_cube,
    load_cube_chunked,
    load_frame,
    sensor_config_from_header,
)

from .stacking import master_bias, master_dark, master_flat

from .stats import (
    WelfordVariance,
    gaussianity_tests,
    mad_sigma,
    outlier_mask,
    sigma_vs_mean_2d,
)

from .noise import (
    detect_rtn_pixels,
    dsnu,
    fpn,
    read_noise_map,
    read_noise_map_from_paths,
    read_noise_spatial,
    row_column_noise,
)

from .dark import (
    arrhenius,
    dark_current_vs_exposure,
    dark_current_vs_temperature,
    dark_spatial_structure,
    warm_pixel_map,
)

from .flat import (
    full_well_capacity,
    noise_decomposition,
    momsdom,
    photon_transfer_curve,
    photon_transfer_curve_with_ron,
)

from .linearity import linearity_error, linearity_test

from .prnu import prnu_map

from .plotting import (
    histogram_gaussian_overlay,
    image_with_colorbar,
    quick_view,
    noise_map_with_histogram,
    momsdom_twilight_plot,
    ptc_plot,
    dark_current_vs_exposure_plot,
    dark_current_vs_temperature_plot,
    summary_table,
)

__all__ = [
    "SensorConfig",
    "AnalysisResult",
    "Frame",
    "FrameCube",
    "ROI",
    "load_frame",
    "load_cube",
    "load_cube_chunked",
    "sensor_config_from_header",
    "group_by_type_and_exposure",
    "get_timestamps",
    "master_bias",
    "master_dark",
    "master_flat",
    "WelfordVariance",
    "gaussianity_tests",
    "mad_sigma",
    "outlier_mask",
    "sigma_vs_mean_2d",
    "read_noise_map",
    "read_noise_map_from_paths",
    "read_noise_spatial",
    "row_column_noise",
    "dsnu",
    "fpn",
    "detect_rtn_pixels",
    "arrhenius",
    "dark_current_vs_exposure",
    "dark_current_vs_temperature",
    "dark_spatial_structure",
    "warm_pixel_map",
    "momsdom",
    "photon_transfer_curve",
    "photon_transfer_curve_with_ron",
    "full_well_capacity",
    "noise_decomposition",
    "linearity_test",
    "linearity_error",
    "prnu_map",
    "image_with_colorbar",
    "quick_view",
    "momsdom_twilight_plot",
    "ptc_plot",
    "dark_current_vs_exposure_plot",
    "dark_current_vs_temperature_plot",
    "histogram_gaussian_overlay",
    "noise_map_with_histogram",
    "summary_table",
]
