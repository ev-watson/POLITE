"""
poltools.simulate — Synthetic polarimetry frames for the POLITE telescope.

Maps astrophysical Stokes vectors through the Mueller model to ordinary and
extraordinary point-spread functions on the detector, then applies QHY268M
detector effects: photon shot noise, dark current, read noise, full-well
clipping, gain, and bias pedestal.

Detector parameters come from ``caltools.SensorConfig``. Filters are treated
as ideal; per-filter Savart geometry is selected with :meth:`PolConfig.for_filter`.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, Union

import numpy as np

from ._types import PointSource, PolConfig
from .mueller import oe_intensities
from . import io as pol_io

_FWHM_TO_SIGMA = 1.0 / (2.0 * np.sqrt(2.0 * np.log(2.0)))


def _add_gaussian_psf(image: np.ndarray, x0: float, y0: float,
                      sigma: float, flux_e: float, stamp_sigma: float = 5.0) -> None:
    """Add a normalized 2-D Gaussian (integral = flux_e) at (x0, y0)."""
    ny, nx = image.shape
    r = int(np.ceil(stamp_sigma * sigma)) + 1
    x_lo, x_hi = max(0, int(np.floor(x0)) - r), min(nx, int(np.ceil(x0)) + r + 1)
    y_lo, y_hi = max(0, int(np.floor(y0)) - r), min(ny, int(np.ceil(y0)) + r + 1)
    if x_lo >= x_hi or y_lo >= y_hi:
        return
    xs = np.arange(x_lo, x_hi)
    ys = np.arange(y_lo, y_hi)
    gx = np.exp(-0.5 * ((xs - x0) / sigma) ** 2)
    gy = np.exp(-0.5 * ((ys - y0) / sigma) ** 2)
    kernel = np.outer(gy, gx)
    norm = 2.0 * np.pi * sigma ** 2
    image[y_lo:y_hi, x_lo:x_hi] += flux_e * kernel / norm


def render_frame(
    sources: Sequence[PointSource],
    cfg: PolConfig,
    hwp_deg: float,
    *,
    exptime_s: float = 1.0,
    seeing_arcsec: float = 2.0,
    sky_e_per_px: float = 50.0,
    rng: Optional[np.random.Generator] = None,
    shape: Optional[Tuple[int, int]] = None,
    retardance_deg: Optional[float] = None,
    efficiency: float = 1.0,
    ip: Tuple[float, float] = (0.0, 0.0),
    beam_throughput: Tuple[float, float] = (1.0, 1.0),
    prnu: Optional[np.ndarray] = None,
    bias_adu: float = 1000.0,
    rotator_deg: Optional[float] = None,
    add_dark: bool = True,
    dark_frame: Optional[np.ndarray] = None,
    return_truth: bool = False,
):
    """Render one detector frame [physical ADU] for a single HWP angle.

    Parameters
    ----------
    efficiency, ip, beam_throughput
        Non-ideal effects: modulation efficiency, instrumental polarization,
        per-beam throughput ratio (for flat-field tests).
    dark_frame : ndarray, optional
        Per-pixel dark accumulation [e⁻]; overrides scalar ``dark_rate·t``.
    """
    rng = np.random.default_rng() if rng is None else rng
    ny, nx = shape if shape is not None else (cfg.sensor.ny, cfg.sensor.nx)
    delta = cfg.retardance_deg if retardance_deg is None else retardance_deg
    rot = cfg.instrument_rotator_deg if rotator_deg is None else rotator_deg

    sigma = cfg.fwhm_px(seeing_arcsec) * _FWHM_TO_SIGMA
    dx, dy = cfg.beam.offset_xy()
    t_o, t_e = beam_throughput

    photo = np.zeros((ny, nx), dtype=np.float64)
    truth = []
    for src in sources:
        S = np.array(src.stokes, dtype=float) * src.flux_e
        I_o_frac, I_e_frac = oe_intensities(
            S, hwp_deg, retardance_deg=delta, efficiency=efficiency,
            ip=ip, rotator_deg=rot,
        )
        f_o = I_o_frac * t_o
        f_e = I_e_frac * t_e
        _add_gaussian_psf(photo, src.x, src.y, sigma, f_o)
        _add_gaussian_psf(photo, src.x + dx, src.y + dy, sigma, f_e)
        truth.append({"name": src.name, "x_o": src.x, "y_o": src.y,
                      "x_e": src.x + dx, "y_e": src.y + dy,
                      "f_o": f_o, "f_e": f_e})

    photo += sky_e_per_px
    if prnu is not None:
        photo *= prnu

    electrons = photo.copy()
    if dark_frame is not None:
        df = np.asarray(dark_frame, dtype=float)
        if df.shape != electrons.shape:
            raise ValueError(
                f"dark_frame shape {df.shape} != frame shape {electrons.shape}"
            )
        electrons += df
    elif add_dark:
        electrons += cfg.dark_rate_e_per_s * exptime_s

    electrons = np.clip(electrons, 0.0, cfg.full_well_e)
    noisy_e = rng.poisson(electrons).astype(np.float64)
    noisy_e += rng.normal(0.0, cfg.read_noise_e, size=noisy_e.shape)

    adu = noisy_e / cfg.sensor.gain_e_per_adu + bias_adu
    out = np.clip(np.rint(adu), 0, (1 << cfg.sensor.bitdepth) - 1).astype(np.uint16)
    if return_truth:
        return out, {"hwp_deg": hwp_deg, "sigma_px": sigma, "offset_xy": (dx, dy),
                     "bias_adu": bias_adu, "sky_e_per_px": sky_e_per_px,
                     "sources": truth}
    return out


def simulate_sequence(
    sources: Sequence[PointSource],
    cfg: PolConfig,
    *,
    out_dir: Union[str, Path],
    exptime_s: float = 1.0,
    seeing_arcsec: float = 2.0,
    sky_e_per_px: float = 50.0,
    rng: Optional[np.random.Generator] = None,
    shape: Optional[Tuple[int, int]] = None,
    object_name: str = "SIM",
    seq_id: str = "SIM",
    **render_kwargs,
) -> List[Path]:
    """Render the full HWP sequence and write one FITS per angle."""
    rng = np.random.default_rng() if rng is None else rng
    out_dir = Path(out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    eff = render_kwargs.get("efficiency", 1.0)

    paths: List[Path] = []
    for i, ang in enumerate(cfg.hwp_angles_deg):
        frame = render_frame(
            sources, cfg, ang, exptime_s=exptime_s, seeing_arcsec=seeing_arcsec,
            sky_e_per_px=sky_e_per_px, rng=rng, shape=shape, **render_kwargs,
        )
        fname = f"{seq_id}_{i:02d}_hwp{ang:05.1f}.fit"
        p = pol_io.write_pol_fits(
            out_dir / fname, frame, ang, cfg, object_name=object_name,
            exptime_s=exptime_s, imagetyp="LIGHT", seq_id=seq_id, seq_index=i,
            efficiency=eff,
        )
        paths.append(p)
    return paths


def make_scene(positions, stokes_list, fluxes, names=None) -> List[PointSource]:
    """Build a list of :class:`PointSource` from parallel input lists."""
    names = names or [f"src{i}" for i in range(len(positions))]
    n = len(positions)
    if not (len(stokes_list) == n and len(fluxes) == n and len(names) == n):
        raise ValueError(
            "make_scene requires positions, stokes_list, fluxes and names of "
            f"equal length; got {len(positions)}, {len(stokes_list)}, "
            f"{len(fluxes)}, {len(names)}"
        )
    out = []
    for (x, y), st, fl, nm in zip(positions, stokes_list, fluxes, names):
        if len(st) == 2:
            st = (1.0, st[0], st[1], 0.0)
        elif len(st) == 3:
            st = (1.0, st[0], st[1], st[2])
        out.append(PointSource(x=x, y=y, flux_e=fl, stokes=tuple(st), name=nm))
    return out
