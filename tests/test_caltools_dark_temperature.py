"""Contracts for dark-current temperature summaries and their plot."""

import matplotlib
import numpy as np

matplotlib.use("Agg")

import matplotlib.pyplot as plt

from caltools._types import AnalysisResult
from caltools.dark import dark_current_vs_temperature
from caltools.plotting import dark_current_vs_temperature_plot


def test_temperature_result_uses_one_exposure_fit_rate_per_temperature(monkeypatch):
    """Temperature points must be the exposure-fit slopes, not rate averages."""
    rates_by_marker = {"cold": (0.5, 0.05), "warm": (2.0, 0.20)}
    fit_calls = []

    monkeypatch.setattr(
        "caltools.dark.master_bias",
        lambda paths, roi=None: np.zeros((2, 2), dtype=np.float32),
    )

    def fake_exposure_fit(dark_files, bias, gain, roi=None):
        fit_calls.append(dark_files)
        marker = next(iter(dark_files.values()))[0]
        rate, error = rates_by_marker[marker]
        return AnalysisResult(
            name="dark_current_vs_exposure",
            scalar_summary={
                "dark_rate_e_per_s": rate,
                "dark_rate_e_err": error,
            },
        )

    monkeypatch.setattr("caltools.dark.dark_current_vs_exposure", fake_exposure_fit)
    groups = {
        -10.0: {
            ("BIAS", 0.0): ["cold-bias"],
            ("DARK", 5.0): ["cold"],
            ("DARK", 60.0): ["cold"],
        },
        0.0: {
            ("BIAS", 0.0): ["warm-bias"],
            ("DARK", 5.0): ["warm"],
            ("DARK", 60.0): ["warm"],
        },
    }

    result, per_temperature = dark_current_vs_temperature(
        groups, gain=1.0, current_vs_exposure=True
    )

    assert np.array_equal(result.metadata["temperatures_c"], [-10.0, 0.0])
    assert np.array_equal(result.metadata["dark_rates_e_per_s"], [0.5, 2.0])
    assert np.array_equal(result.metadata["dark_rate_errors_e_per_s"], [0.05, 0.20])
    assert len(per_temperature) == 2
    assert all(0.0 not in call for call in fit_calls)


def test_temperature_plot_uses_one_error_bar_per_temperature():
    result = AnalysisResult(
        name="dark_current_vs_temperature",
        metadata={
            "temperatures_c": np.array([-10.0, 0.0]),
            "dark_rates_e_per_s": np.array([0.5, 2.0]),
            "dark_rate_errors_e_per_s": np.array([0.05, 0.20]),
        },
    )
    fig, ax = plt.subplots()
    dark_current_vs_temperature_plot(ax, result)

    assert ax.get_xlabel() == "Temperature (°C)"
    assert ax.get_ylabel() == "Dark current (e-/pixel/s)"
    assert ax.get_yscale() == "linear"
    plt.close(fig)
