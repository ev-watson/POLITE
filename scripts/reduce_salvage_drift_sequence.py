#!/usr/bin/env python3
"""Frame-by-frame dual-beam diagnostic for the salvage drift sequence.

This reducer is intentionally separate from calibrated ``poltools`` Stokes
reduction.  It measures the pair geometry from detections, follows the source
before it leaves the detector, performs local masked-aperture photometry, and
fits the fourth-harmonic beam ratio with a free throughput offset.  No sky P/PA
or calibrated polarization is reported.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from astropy.io import fits

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import caltools as ct  # noqa: E402
from poltools.errors import sigma_r  # noqa: E402
from scripts.analyze_salvage_first_light import _candidate_peaks  # noqa: E402


def _match_pair(peaks, dx: float, dy: float, tolerance_px: float):
    matches = []
    for ordinary in peaks:
        for extraordinary in peaks:
            if ordinary is extraordinary:
                continue
            rx = extraordinary["x"] - ordinary["x"] - dx
            ry = extraordinary["y"] - ordinary["y"] - dy
            residual = math.hypot(rx, ry)
            if residual <= tolerance_px:
                score = min(ordinary["smooth_snr"], extraordinary["smooth_snr"])
                matches.append((score - 0.1 * residual, ordinary, extraordinary))
    if not matches:
        return None
    _, ordinary, extraordinary = max(matches, key=lambda item: item[0])
    return ordinary, extraordinary


def _local_flux(
    image: np.ndarray,
    bad_mask: np.ndarray,
    x: float,
    y: float,
    *,
    r_ap: float,
    r_in: float,
    r_out: float,
) -> tuple[float, float, float]:
    margin = int(math.ceil(r_out)) + 2
    xi, yi = int(round(x)), int(round(y))
    x0, x1 = max(0, xi - margin), min(image.shape[1], xi + margin + 1)
    y0, y1 = max(0, yi - margin), min(image.shape[0], yi + margin + 1)
    yy, xx = np.mgrid[y0:y1, x0:x1]
    rr = np.hypot(xx - x, yy - y)
    good = ~bad_mask[y0:y1, x0:x1] & np.isfinite(image[y0:y1, x0:x1])
    aperture = (rr <= r_ap) & good
    annulus = (rr >= r_in) & (rr <= r_out) & good
    if np.count_nonzero(aperture) < 0.8 * math.pi * r_ap ** 2:
        return float("nan"), float("nan"), float("nan")
    local = image[y0:y1, x0:x1]
    background = float(np.median(local[annulus]))
    bg_sigma = float(1.4826 * np.median(np.abs(local[annulus] - background)))
    flux = float(np.sum(local[aperture] - background))
    n_ap = int(np.count_nonzero(aperture))
    n_sky = max(int(np.count_nonzero(annulus)), 1)
    # Empirical detector+sky term only. Source shot noise needs a measured
    # e-/ADU conversion gain, which this session does not have.
    sigma = bg_sigma * math.sqrt(n_ap + n_ap ** 2 / n_sky)
    return flux, sigma, background


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _parse_good_time(value: Any) -> datetime | None:
    try:
        parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except (TypeError, ValueError):
        return None
    return parsed if parsed.year == 2026 else None


def _fit_drift(rows: list[dict[str, Any]], plate_scale: float) -> dict[str, float]:
    valid = [row for row in rows if row["tracked"] and row["valid_time"]]
    if len(valid) < 3:
        return {}
    times = np.array([
        _parse_good_time(row["date_obs"]).timestamp() for row in valid
    ])
    times -= times[0]
    x = np.array([row["ordinary_x"] for row in valid], dtype=float)
    y = np.array([row["ordinary_y"] for row in valid], dtype=float)
    vx, x0 = np.polyfit(times, x, 1)
    vy, y0 = np.polyfit(times, y, 1)
    residual = np.hypot(x - (x0 + vx * times), y - (y0 + vy * times))
    speed = float(math.hypot(vx, vy))
    return {
        "vx_px_per_s": float(vx),
        "vy_px_per_s": float(vy),
        "speed_px_per_s": speed,
        "speed_arcsec_per_s": speed * plate_scale,
        "predicted_trail_px_at_0.05s": speed * 0.05,
        "fit_residual_rms_px": float(np.sqrt(np.mean(residual ** 2))),
        "n_valid_timestamp_positions": len(valid),
    }


def _fit_modulation(rows: list[dict[str, Any]]) -> dict[str, Any]:
    used = [
        row for row in rows
        if row["tracked"] and row["flux_o_adu"] > 0 and row["flux_e_adu"] > 0
    ]
    if len(used) < 4:
        return {"n_used": len(used)}
    theta = np.deg2rad([row["hwp_angle_deg"] for row in used])
    z = np.array([row["beam_ratio"] for row in used], dtype=float)
    design = np.column_stack([
        np.ones(len(theta)), np.cos(4.0 * theta), np.sin(4.0 * theta)
    ])
    coef, *_ = np.linalg.lstsq(design, z, rcond=None)
    model = design @ coef
    residual = z - model
    dof = max(len(z) - design.shape[1], 1)
    residual_variance = float(residual @ residual) / dof
    covariance = residual_variance * np.linalg.inv(design.T @ design)
    a0, q, u = (float(value) for value in coef)
    sigma_q = float(math.sqrt(max(covariance[1, 1], 0.0)))
    sigma_u = float(math.sqrt(max(covariance[2, 2], 0.0)))
    amplitude = float(math.hypot(q, u))
    sigma_amp = float(math.sqrt(sigma_q ** 2 + sigma_u ** 2))
    return {
        "n_used": len(used),
        "throughput_offset_a0": a0,
        "q_detector_uncalibrated": q,
        "u_detector_uncalibrated": u,
        "sigma_q_empirical": sigma_q,
        "sigma_u_empirical": sigma_u,
        "fourth_harmonic_amplitude": amplitude,
        "amplitude_conservative_snr": amplitude / sigma_amp if sigma_amp else None,
        "residual_rms": float(np.sqrt(np.mean(residual ** 2))),
        "angles_used_deg": [float(np.rad2deg(value)) for value in theta],
    }


def _plot_results(output: Path, rows: list[dict[str, Any]], modulation, shape) -> None:
    tracked = [row for row in rows if row["tracked"]]
    valid_flux = [
        row for row in tracked if row["flux_o_adu"] > 0 and row["flux_e_adu"] > 0
    ]
    fig, axes = plt.subplots(1, 3, figsize=(16, 5), constrained_layout=True)

    ax = axes[0]
    ax.set_xlim(0, shape[1])
    ax.set_ylim(shape[0], 0)
    ax.plot(
        [row["ordinary_x"] for row in tracked],
        [row["ordinary_y"] for row in tracked], "o-", label="beam 1",
    )
    ax.plot(
        [row["extraordinary_x"] for row in tracked],
        [row["extraordinary_y"] for row in tracked], "o-", label="beam 2",
    )
    ax.set_xlabel("detector X [px]")
    ax.set_ylabel("detector Y [px]")
    ax.set_title("Measured detector track")
    ax.legend()

    ax = axes[1]
    angles = np.array([row["hwp_angle_deg"] for row in valid_flux])
    ratios = np.array([row["beam_ratio"] for row in valid_flux])
    errors = np.array([row["sigma_beam_ratio_lower_bound"] for row in valid_flux])
    ax.errorbar(angles, ratios, yerr=errors, fmt="o", capsize=3,
                label="frame photometry")
    if modulation.get("n_used", 0) >= 4:
        grid = np.linspace(0, max(angles), 300)
        model = (
            modulation["throughput_offset_a0"]
            + modulation["q_detector_uncalibrated"] * np.cos(4 * np.deg2rad(grid))
            + modulation["u_detector_uncalibrated"] * np.sin(4 * np.deg2rad(grid))
        )
        ax.plot(grid, model, label="a0 + q cos4θ + u sin4θ")
    ax.axhline(0, color="0.5", lw=1)
    ax.set_xlabel("HWP angle [deg]")
    ax.set_ylabel("(F₂ − F₁)/(F₂ + F₁)")
    ax.set_title("Uncalibrated modulation diagnostic")
    ax.legend()

    ax = axes[2]
    ax.plot(
        range(len(valid_flux)),
        [row["flux_o_adu"] + row["flux_e_adu"] for row in valid_flux], "o-",
    )
    ax.set_xlabel("tracked frame index")
    ax.set_ylabel("F₁ + F₂ [ADU]")
    ax.set_title("Pair total flux (not conserved)")
    fig.savefig(output / "tracked_pair_diagnostics.png", dpi=170)
    plt.close(fig)


def reduce_sequence(
    session_dir: Path,
    calibration_dir: Path,
    output: Path,
    sequence: str,
    beam_dx: float,
    beam_dy: float,
    pair_tolerance: float,
    r_ap: float,
) -> dict[str, Any]:
    output.mkdir(parents=True, exist_ok=True)
    master_bias = fits.getdata(calibration_dir / "master_bias.fits")
    master_dark = fits.getdata(calibration_dir / "master_dark_0.05s.fits")
    bad_mask = fits.getdata(calibration_dir / "bad_pixel_mask.fits").astype(bool)

    paths = []
    for path in sorted(session_dir.glob("*.fits")):
        hdr = fits.getheader(path, ignore_missing_end=True)
        if hdr.get("POLSEQ") == sequence:
            paths.append(path)
    if not paths:
        raise FileNotFoundError(f"No frames with POLSEQ={sequence!r}")

    rows: list[dict[str, Any]] = []
    vectors = []
    shape = master_bias.shape
    for path in paths:
        hdr = fits.getheader(path, ignore_missing_end=True)
        corrected = ct.load_frame(str(path)) - master_bias - master_dark
        peaks, _, _ = _candidate_peaks(corrected, bad_mask)
        match = _match_pair(peaks, beam_dx, beam_dy, pair_tolerance)
        row: dict[str, Any] = {
            "filename": path.name,
            "hwp_angle_deg": float(hdr["HWPANG"]),
            "polseq_index": int(hdr.get("POLSEQN", -1)),
            "date_obs": hdr.get("DATE-OBS"),
            "valid_time": _parse_good_time(hdr.get("DATE-OBS")) is not None,
            "tracked": match is not None,
            "ordinary_x": None,
            "ordinary_y": None,
            "extraordinary_x": None,
            "extraordinary_y": None,
            "separation_dx_px": None,
            "separation_dy_px": None,
            "separation_px": None,
            "flux_o_adu": None,
            "flux_e_adu": None,
            "sigma_o_empirical_adu": None,
            "sigma_e_empirical_adu": None,
            "beam_ratio": None,
            "sigma_beam_ratio_lower_bound": None,
        }
        if match is not None:
            ordinary, extraordinary = match
            ox, oy = ordinary["x"], ordinary["y"]
            ex, ey = extraordinary["x"], extraordinary["y"]
            fo, so, _ = _local_flux(
                corrected, bad_mask, ox, oy,
                r_ap=r_ap, r_in=r_ap + 10, r_out=r_ap + 30,
            )
            fe, se, _ = _local_flux(
                corrected, bad_mask, ex, ey,
                r_ap=r_ap, r_in=r_ap + 10, r_out=r_ap + 30,
            )
            ratio = (fe - fo) / (fe + fo) if fe + fo != 0 else float("nan")
            ratio_sigma = sigma_r(fo, fe, so, se)
            vx, vy = ex - ox, ey - oy
            vectors.append((vx, vy))
            row.update({
                "ordinary_x": ox, "ordinary_y": oy,
                "extraordinary_x": ex, "extraordinary_y": ey,
                "separation_dx_px": vx, "separation_dy_px": vy,
                "separation_px": math.hypot(vx, vy),
                "flux_o_adu": fo, "flux_e_adu": fe,
                "sigma_o_empirical_adu": so, "sigma_e_empirical_adu": se,
                "beam_ratio": ratio,
                "sigma_beam_ratio_lower_bound": ratio_sigma,
            })
        rows.append(row)
    _write_csv(output / "tracked_pair_photometry.csv", rows)

    vector_array = np.asarray(vectors, dtype=float)
    plate_scale = float(fits.getheader(paths[0]).get("PIXSCALE", 0.224))
    geometry = {
        "n_pair_detections": len(vectors),
        "dx_mean_px": float(np.mean(vector_array[:, 0])),
        "dx_std_px": float(np.std(vector_array[:, 0], ddof=1)),
        "dy_mean_px": float(np.mean(vector_array[:, 1])),
        "dy_std_px": float(np.std(vector_array[:, 1], ddof=1)),
        "separation_mean_px": float(np.mean(np.hypot(vector_array[:, 0], vector_array[:, 1]))),
        "position_angle_detector_deg": float(
            np.rad2deg(np.arctan2(np.mean(vector_array[:, 0]),
                                  np.mean(vector_array[:, 1]))) % 360.0
        ),
    }
    drift = _fit_drift(rows, plate_scale)
    modulation = _fit_modulation(rows)
    result = {
        "sequence": sequence,
        "n_frames": len(rows),
        "n_frames_with_pair": sum(row["tracked"] for row in rows),
        "aperture_radius_px": r_ap,
        "geometry": geometry,
        "drift": drift,
        "modulation": modulation,
        "caveats": [
            "Detector-frame diagnostic only; no instrumental-polarization, efficiency, or PA calibration.",
            "Flux uncertainty is an empirical detector/background lower bound because EGAIN is unmeasured.",
            "Frames without two >8-sigma smoothed peaks are excluded; the source leaves/fades before sequence completion.",
        ],
    }
    with (output / "tracked_pair_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2)
        handle.write("\n")
    _plot_results(output, rows, modulation, shape)
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("session_dir", nargs="?", type=Path,
                        default=Path("FITSDATA/20260709"))
    parser.add_argument("--calibration-dir", type=Path,
                        default=Path("generated/salvage_first_light_20260709"))
    parser.add_argument("--output", type=Path,
                        default=Path("generated/salvage_first_light_20260709"))
    parser.add_argument("--sequence", default="driftA_polV8_rep2")
    parser.add_argument("--beam-dx", type=float, default=-127.0)
    parser.add_argument("--beam-dy", type=float, default=203.0)
    parser.add_argument("--pair-tolerance", type=float, default=20.0)
    parser.add_argument("--aperture-radius", type=float, default=20.0)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    result = reduce_sequence(
        args.session_dir, args.calibration_dir, args.output, args.sequence,
        args.beam_dx, args.beam_dy, args.pair_tolerance, args.aperture_radius,
    )
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
