"""Observatory automation package with lazy public exports.

Importing a pure module (for example ``obs_utils.qa_lib`` or
``obs_utils.timing``) must not require camera, mount, or Alpaca drivers.  Public
objects remain available from ``obs_utils`` but their owning module is loaded
only on first access.
"""

from __future__ import annotations

from importlib import import_module

_EXPORTS = {
    "AlpacaConfig": (".config", "AlpacaConfig"),
    "Pwi4Config": (".config", "Pwi4Config"),
    "PyxisSerialConfig": (".config", "PyxisSerialConfig"),
    "SkyRegionLimit": (".config", "SkyRegionLimit"),
    "SlewLimits": (".config", "SlewLimits"),
    "default_sky_regions": (".config", "default_sky_regions"),
    "PyxisGen3": (".pyxis_gen3", "PyxisGen3"),
    "PyxisError": (".pyxis_gen3", "PyxisError"),
    "PyxisTimeout": (".pyxis_gen3", "PyxisTimeout"),
    "connect_pyxis_gen3": (".pyxis_gen3", "connect_pyxis_gen3"),
    "CaptureRequest": (".imaging", "CaptureRequest"),
    "capture_fits_file": (".imaging", "capture_fits_file"),
    "capture_image_array": (".imaging", "capture_image_array"),
    "close_session": (".imaging", "close_session"),
    "open_session": (".imaging", "open_session"),
    "select_filter": (".imaging", "select_filter"),
    "select_hwp_angle": (".imaging", "select_hwp_angle"),
    "CameraROI": (".roi", "CameraROI"),
    "EffectiveArea": (".roi", "EffectiveArea"),
    "apply_roi": (".roi", "apply_roi"),
    "camera_effective_area": (".roi", "camera_effective_area"),
    "DeviceTimeout": (".waits", "DeviceTimeout"),
    "wait_until": (".waits", "wait_until"),
    "wait_for_value": (".waits", "wait_for_value"),
    "StarProfile": (".focus", "StarProfile"),
    "brightest_star": (".focus", "brightest_star"),
    "measure_fwhm": (".focus", "measure_fwhm"),
    "measure_hfd": (".focus", "measure_hfd"),
    "star_profile": (".focus", "star_profile"),
    "connect_mount": (".mount", "connect_mount"),
    "enable_motors": (".mount", "enable_motors"),
    "home_mount": (".mount", "home_mount"),
    "load_pointing_model": (".mount", "load_pointing_model"),
    "set_slew_time_constant": (".mount", "set_slew_time_constant"),
    "slew_altaz": (".mount", "slew_altaz"),
    "slew_radec_j2000": (".mount", "slew_radec_j2000"),
    "wait_for_slew": (".mount", "wait_for_slew"),
    "StartupConfig": (".startup", "StartupConfig"),
    "StartupState": (".startup", "StartupState"),
    "startup_observatory": (".startup", "startup_observatory"),
    "AlpacaServerStatus": (".alpaca_servers", "AlpacaServerStatus"),
    "start_observatory_alpaca_servers": (
        ".alpaca_servers",
        "start_observatory_alpaca_servers",
    ),
    "uses_local_observatory_alpaca_servers": (
        ".alpaca_servers",
        "uses_local_observatory_alpaca_servers",
    ),
    "FilterWheelState": (".timing", "FilterWheelState"),
    "NtpStatus": (".timing", "NtpStatus"),
    "TimingConfig": (".timing", "TimingConfig"),
    "TimingProvenance": (".timing", "TimingProvenance"),
    "acquire_session_timing_snapshot": (".timing", "acquire_session_timing_snapshot"),
    "LogPaths": (".logging", "LogPaths"),
    "LoggingConfig": (".logging", "LoggingConfig"),
    "build_log_paths": (".logging", "build_log_paths"),
    "setup_logging": (".logging", "setup_logging"),
    "FramePlan": (".night_session", "FramePlan"),
    "NightSessionConfig": (".night_session", "NightSessionConfig"),
    "TargetPlan": (".night_session", "TargetPlan"),
    "run_night_session": (".night_session", "run_night_session"),
    "NightPlanError": (".night_plan", "NightPlanError"),
    "describe": (".night_plan", "describe"),
    "load_night_plan": (".night_plan", "load_night_plan"),
    "INSTALLED_EFW_NAMES": (".night_safety", "INSTALLED_EFW_NAMES"),
    "plan_hwp_angles": (".night_safety", "plan_hwp_angles"),
    "plan_requires_hwp": (".night_safety", "plan_requires_hwp"),
    "report_settings": (".night_safety", "report_settings"),
    "verify_filter_wheel": (".night_safety", "verify_filter_wheel"),
    "verify_hwp": (".night_safety", "verify_hwp"),
    "cooler_gate": (".night_safety", "cooler_gate"),
    "GuidePulse": (".autoguide", "GuidePulse"),
    "autoguide_from_offsets": (".autoguide", "autoguide_from_offsets"),
    "dither_mount_offset_arcsec": (".autoguide", "dither_mount_offset_arcsec"),
    "offsets_to_pulses": (".autoguide", "offsets_to_pulses"),
    "pulse_guide": (".autoguide", "pulse_guide"),
    "random_dither_mount_offset_arcsec": (".autoguide", "random_dither_mount_offset_arcsec"),
    "ALPACA_CONFIG": (".user_config", "ALPACA_CONFIG"),
    "PWI4_CONFIG": (".user_config", "PWI4_CONFIG"),
    "PYXIS_CONFIG": (".user_config", "PYXIS_CONFIG"),
}

__all__ = list(_EXPORTS)


def __getattr__(name: str):
    """Import a public object only when it is first requested."""
    try:
        module_name, attr_name = _EXPORTS[name]
    except KeyError as exc:  # pragma: no cover - standard module protocol
        raise AttributeError(name) from exc
    value = getattr(import_module(module_name, __name__), attr_name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))
