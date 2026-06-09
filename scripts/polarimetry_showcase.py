#!/usr/bin/env python
"""
POLITE polarimetry end-to-end showcase.

Generates simulated 2-D imaging FITS frames from the POLITE telescope chain
(CDK20 → PWI4 rotator → L3 cut → HWP → ZWO EFW → α-BBO Savart 18 mm → QHY268M),
runs them through the full poltools reduction pipeline, calibrates against
simulated standard stars, and extracts the Stokes parameters with
publication-grade error metrics — the deliverable the original request asked for.

The α-BBO split is per-filter (dispersive); this showcase runs in one EFW band
(Photometric V) registered via ``default_efw_filters`` / ``for_filter``.

It exercises all four science regimes (stellar few-%, ISM 1–5%, solar-system
sub-% high-SNR, supernova sub-%), demonstrates double-ratio vs LSQ-modulation
agreement, MAS debiasing of the low-SNR case, instrumental-
polarization + efficiency calibration from standards, and a Monte-Carlo pull
test, writing figures and a results table under docs/polarimetry/.

Run:
    /Users/blu3/miniforge3/envs/POLITE/bin/python scripts/polarimetry_showcase.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import poltools as pt
from poltools import calibration as cal
from poltools import plotting as pplt
from caltools import SensorConfig

# ---- reproducibility -------------------------------------------------------
SEED = 20260607
rng = np.random.default_rng(SEED)

OUT_FITS = ROOT / "FITSDATA" / "SIM_20260607"
FIGDIR = ROOT / "docs" / "polarimetry" / "figures"
FIGDIR.mkdir(parents=True, exist_ok=True)
RESULTS_MD = ROOT / "docs" / "polarimetry" / "showcase_results.md"

# ---- instrument configuration (QHY268M on CDK20, α-BBO Savart dual beam) ----
# The α-BBO Savart split is dispersive -> per-filter geometry. We register the
# ZWO 5-slot EFW and run in one band (Photometric V); the placeholder 60-px split
# is the same across bands here (real per-band BEAMSEP comes from flats, Source A).
SEP_PX = 60.0
SHOW_FILTER = "Photometric V"
sensor = SensorConfig(nx=512, ny=512, pixel_size_um=3.76, gain_e_per_adu=1.0,
                      temperature_c=-10.0, sensor_name="QHY268M")
ANGLES8 = tuple(i * 22.5 for i in range(8))
cfg = pt.PolConfig(
    sensor=sensor,
    beam=pt.BeamGeometry(separation_px=SEP_PX, position_angle_deg=0.0),
    plate_scale_arcsec=0.224,
    hwp_angles_deg=ANGLES8,
    retardance_deg=180.0,
    filters=pt.default_efw_filters(SEP_PX),
    read_noise_e=3.5,
    full_well_e=51000.0,
).for_filter(SHOW_FILTER)

# ---- injected systematics (to be calibrated out) --------------------------
IP_TRUE = (0.004, -0.003)     # instrumental polarization q0,u0
EFF_TRUE = 0.97               # polarization (modulation) efficiency
SKY_E = 80.0
EXPTIME = 20.0
# Seeing/aperture/separation chosen so the brightest beam stays below the
# QHY268M full well (51 ke-): FWHM ~ 9.8 px (sigma 4.17), peak of an 8e6-e-
# source ~ 37 ke- < FWC.  Beam sep 60 px = 6.1 FWHM -> clean o/e separation
# (per-band α-BBO separation in real data is measured from flats).
SEEING = 2.2                  # arcsec
BIAS = 1000.0
R_AP, R_IN, R_OUT = 13.0, 18.0, 29.0

# ---- the science field + standards (truth) --------------------------------
# (name, x, y, p, theta_deg, total_flux_e, role, p_lit). Positions place the
# e-beam (y+60) and sky annulus (r_out=29) inside the 512x512 frame, with all
# sources separated by > 2*r_out so apertures/annuli never overlap.
FIELD = [
    ("stellar_eps_aur", 110, 90,  0.030, 30.0,  5.0e6, "science", None),
    ("ism_dust",        300, 90,  0.015, 120.0, 4.0e6, "science", None),
    ("asteroid",        110, 250, 0.005, 75.0,  7.0e6, "science", None),
    ("supernova",       300, 250, 0.003, 160.0, 8.0e6, "science", None),
    ("sne_faint",       205, 170, 0.005, 45.0,  3.0e5, "science", None),  # low SNR
    ("unpol_std_1",     420, 90,  0.000, 0.0,   6.0e6, "unpol", 0.0),
    ("unpol_std_2",     420, 250, 0.000, 0.0,   6.0e6, "unpol", 0.0),
    ("highpol_std",     205, 360, 0.050, 90.0,  6.0e6, "polstd", 0.050),
]


def truth_qu(p, theta_deg):
    th = np.deg2rad(theta_deg)
    return p * np.cos(2 * th), p * np.sin(2 * th)


def build_scene():
    positions, stokes, fluxes, names = [], [], [], []
    for nm, x, y, p, th, fl, role, plit in FIELD:
        q, u = truth_qu(p, th)
        positions.append((float(x), float(y)))
        stokes.append((1.0, q, u, 0.0))
        fluxes.append(fl)
        names.append(nm)
    return pt.make_scene(positions, stokes, fluxes, names), positions, names


def banner(msg):
    print("\n" + "=" * 72 + f"\n{msg}\n" + "=" * 72)


def main():
    banner("1. SIMULATE telescope-chain frames (HWP sequence -> FITS)")
    scene, positions, names = build_scene()
    paths = pt.simulate_sequence(
        scene, cfg, out_dir=OUT_FITS, exptime_s=EXPTIME, seeing_arcsec=SEEING,
        sky_e_per_px=SKY_E, rng=rng, shape=(512, 512), object_name="SHOWCASE",
        ip=IP_TRUE, efficiency=EFF_TRUE, bias_adu=BIAS,
    )
    print(f"wrote {len(paths)} FITS frames -> {OUT_FITS}")
    for p in paths:
        print("   ", p.name)

    banner("2. READ frames + DETECT sources (photutils) + pair o/e")
    frames_by_angle = {}
    for p in paths:
        data, hwp, _ = pt.read_pol_frame(str(p))
        frames_by_angle[round(hwp, 3)] = data
    ref = frames_by_angle[0.0]
    det = pt.detect_sources(ref, cfg.fwhm_px(SEEING), threshold_sigma=8.0)
    pairs = pt.pair_oe(det, cfg.beam)
    print(f"detected {0 if det is None else len(det)} sources -> {len(pairs)} "
          f"o/e pairs (truth: {len(positions)} sources)")

    banner("3. APERTURE PHOTOMETRY across HWP angles")
    flux_by_src = pt.photometer_sequence(
        frames_by_angle, cfg, positions, names,
        r_ap=R_AP, r_in=R_IN, r_out=R_OUT, bias_adu=BIAS,
    )

    banner("4. MODULATION -> q,u  (double-ratio + LSQ modulation)")
    raw_dratio, raw_lsq = {}, {}
    for nm in names:
        bfs = flux_by_src[nm]
        raw_dratio[nm] = pt.double_ratio(bfs)
        raw_lsq[nm] = pt.lsq_modulation(bfs)
    for nm in names:
        a, b = raw_dratio[nm], raw_lsq[nm]
        print(f"  {nm:16s} double-ratio: q={a['q']:+.4f} u={a['u']:+.4f}   "
              f"LSQ: q={b['q']:+.4f} u={b['u']:+.4f}  chi2={b['chi2']:.1f}/{b['dof']}")

    banner("5. CALIBRATION from standard stars (IP, efficiency)")
    unpol = [nm for (nm, *_rest, role, _p) in FIELD if role == "unpol"]
    q_ip = [raw_dratio[nm]["q"] for nm in unpol]
    u_ip = [raw_dratio[nm]["u"] for nm in unpol]
    q0, u0, _ = cal.fit_instrumental_polarization(q_ip, u_ip)
    print(f"  fitted IP:  q0={q0:+.4f} u0={u0:+.4f}   (injected eff*IP="
          f"{EFF_TRUE*IP_TRUE[0]:+.4f},{EFF_TRUE*IP_TRUE[1]:+.4f})")

    # efficiency from the high-pol standard (IP-corrected)
    nm_std = "highpol_std"
    p_lit_std = dict((f[0], f[7]) for f in FIELD)[nm_std]
    qc, uc = cal.apply_ip(raw_dratio[nm_std]["q"], raw_dratio[nm_std]["u"], q0, u0)
    eff = cal.fit_efficiency([np.hypot(qc, uc)], [p_lit_std])
    print(f"  fitted efficiency: {eff:.4f}   (injected {EFF_TRUE})")

    calib = pt.PolCalibration(q0=q0, u0=u0, dtheta_deg=0.0, efficiency=eff)

    banner("6. CALIBRATED STOKES + ERROR METRICS (science targets)")
    science = [(nm, p, th) for (nm, x, y, p, th, fl, role, plit) in FIELD
               if role == "science"]
    results, table_rows = [], []
    for nm, p_true, th_true in science:
        qu = dict(raw_dratio[nm])
        qc, uc = calib.apply(qu["q"], qu["u"])
        qu["q"], qu["u"] = qc, uc
        if eff not in (0.0, 1.0):
            qu["sigma_q"] /= eff
            qu["sigma_u"] /= eff
        I0 = float(np.mean([b.f_o + b.f_e for b in flux_by_src[nm]]))
        res = pt.assemble_stokes(qu, I0=I0, name=nm, method="double_ratio", estimator="mas")
        results.append(res)
        s = res.scalar_summary
        table_rows.append((nm, p_true, th_true, s))
        print(f"  {nm:16s} p={s['p']:.4%} (inj {p_true:.3%}) "
              f"pMAS={s['p_mas']:.4%} ±{s['sigma_p']:.4%}  "
              f"θ={s['theta_deg']:6.2f}° (inj {th_true:.1f}°) "
              f"±{s['sigma_theta_deg']:.2f}°  SNR={s['snr']:.1f}")

    banner("7. MONTE-CARLO pull test (q,u error-bar calibration)")
    q_t, u_t = truth_qu(0.03, 30.0)
    pulls_q, pulls_u = [], []
    for _ in range(3000):
        S = pt.stokes_vector(1.0, q_t, u_t)
        from poltools._types import BeamFlux
        bfs = []
        for th in ANGLES8:
            Io, Ie = pt.oe_intensities(S, th)
            fo = rng.poisson(Io * 3.0e5)
            fe = rng.poisson(Ie * 3.0e5)
            bfs.append(BeamFlux(th, fo, fe, np.sqrt(max(fo, 1)), np.sqrt(max(fe, 1))))
        B = pt.lsq_modulation(bfs)
        pulls_q.append((B["q"] - q_t) / B["sigma_q"])
        pulls_u.append((B["u"] - u_t) / B["sigma_u"])
    pulls = np.array(pulls_q + pulls_u)
    print(f"  pull mean={pulls.mean():+.3f}  std={pulls.std(ddof=1):.3f}  "
          f"(target 0, 1)")

    banner("8. FIGURES + RESULTS TABLE")
    # modulation curves for three representative targets
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.4))
    for ax, nm in zip(axes, ["stellar_eps_aur", "ism_dust", "supernova"]):
        pplt.plot_modulation_curve(flux_by_src[nm], raw_lsq[nm]["q"], raw_lsq[nm]["u"],
                                   ax=ax, title=nm)
    fig.tight_layout()
    fig.savefig(FIGDIR / "fig_modulation.png", dpi=130)
    plt.close(fig)

    # q-u plane with injected truth
    inj = [truth_qu(p, th) for (nm, p, th) in science]
    ax = pplt.plot_qu_plane(results, injected=inj,
                            title="q–u plane (calibrated, injected = ×)")
    ax.figure.savefig(FIGDIR / "fig_qu_plane.png", dpi=130, bbox_inches="tight")
    plt.close(ax.figure)

    # recovered vs injected p
    inj_p = [p for (nm, p, th) in science]
    rec_p = [r.scalar_summary["p_mas"] for r in results]
    sig_p = [r.scalar_summary["sigma_p"] for r in results]
    ax = pplt.plot_recovered_vs_injected(inj_p, rec_p, sig_p)
    ax.figure.savefig(FIGDIR / "fig_recovered_vs_injected.png", dpi=130,
                      bbox_inches="tight")
    plt.close(ax.figure)

    # pull histogram
    ax = pplt.plot_pull_histogram(pulls)
    ax.figure.savefig(FIGDIR / "fig_pull.png", dpi=130, bbox_inches="tight")
    plt.close(ax.figure)

    # polarization-vector overlay on a frame
    science_positions = [(x, y) for (nm, x, y, p, th, fl, role, plit) in FIELD
                         if role == "science"]
    ax = pplt.plot_pol_vectors(ref, results, science_positions, scale=400.0)
    ax.figure.savefig(FIGDIR / "fig_pol_vectors.png", dpi=130, bbox_inches="tight")
    plt.close(ax.figure)
    print(f"  figures -> {FIGDIR}")

    # results table (markdown)
    lines = ["# POLITE polarimetry showcase — results\n",
             f"Seed {SEED}; QHY268M {sensor.nx}×{sensor.ny}; {len(ANGLES8)} HWP angles; "
             f"double-ratio reduction, MAS debiasing.\n",
             f"Injected IP=({IP_TRUE[0]:+.4f},{IP_TRUE[1]:+.4f}), efficiency={EFF_TRUE}; "
             f"fitted IP=({q0:+.4f},{u0:+.4f}), efficiency={eff:.4f}.\n",
             f"MC pull: mean={pulls.mean():+.3f}, std={pulls.std(ddof=1):.3f} (target 0,1).\n",
             "",
             "| source | p_inj | p_rec | p_MAS | σ_p | θ_inj | θ_rec | σ_θ | SNR |",
             "|---|---|---|---|---|---|---|---|---|"]
    for nm, p_true, th_true, s in table_rows:
        lines.append(
            f"| {nm} | {p_true:.3%} | {s['p']:.3%} | {s['p_mas']:.3%} | "
            f"{s['sigma_p']:.3%} | {th_true:.1f}° | {s['theta_deg']:.2f}° | "
            f"{s['sigma_theta_deg']:.2f}° | {s['snr']:.1f} |")
    RESULTS_MD.write_text("\n".join(lines) + "\n")
    print(f"  results table -> {RESULTS_MD}")
    banner("DONE")


if __name__ == "__main__":
    main()
