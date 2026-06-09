"""
poltools.simulate — Forward model of the POLITE telescope chain.

Optical chain modelled (real light path):
``Sky → CDK20 (D=0.508 m, f/6.8) → PWI4 Focuser/Rotator (α) → Astronomik L3
UV/IR-cut → rotating HWP (θ, δ) → ZWO 5-slot EFW (B/V/R/Clear/Dark) → α-BBO
Savart plate (18 mm; dispersive o/e split) → QHY268M (IMX571, 0.224″/px)``.
The L3 and EFW filters are treated as ideal (scalar × identity) and so do not
alter q,u; the EFW selects the band whose **per-filter** Savart separation
(``cfg.beam`` via :meth:`PolConfig.for_filter`) sets the o/e offset.

Renders 2-D imaging FITS frames byte-compatible with the real reduction path:
for each HWP angle it maps every source's astrophysical Stokes vector through the
Mueller chain (PWI4 rotator → HWP → analyzer) to ordinary/extraordinary fluxes,
places the two PSFs on the detector grid (origin upper-left), injects the QHY268M
detector physics (shot noise, dark, read noise, full-well clipping, gain,
bias pedestal, BZERO=32768), and writes a FITS file with the polarimetry keywords.

The detector parameters come from ``caltools.SensorConfig`` so the simulator is
fully detector-parameterized (QHY268M default; legacy CCD selectable).
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
    """Add a normalized 2-D Gaussian (sum=flux_e) to ``image`` at (x0, y0).

    Origin upper-left: pixel (row, col) = (y, x). Renders only a local stamp for
    efficiency; the analytic normalization uses the full Gaussian integral.
    """
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
    norm = 2.0 * np.pi * sigma ** 2  # analytic integral of the un-normalized Gaussian
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
    """Render one detector frame (physical ADU) for a single HWP angle.

    Parameters
    ----------
    sources : sequence of PointSource
        Scene; each carries total intensity flux (e-) and normalized Stokes.
    cfg : PolConfig
        Instrument + detector configuration.
    hwp_deg : float
        HWP angle θ.
    seeing_arcsec : float
        PSF FWHM in arcsec (converted via the plate scale).
    sky_e_per_px : float
        Sky background (e-/pixel) for this exposure.
    efficiency, ip, beam_throughput :
        Non-ideal effects to inject (modulation efficiency, instrumental
        polarization q0/u0, per-beam throughput ratio for flat-field tests).
    prnu : ndarray, optional
        Per-pixel response (PRNU) map multiplying the photo-electron image.
    add_dark : bool
        Add the scalar dark accumulation ``dark_rate·t`` before the Poisson draw.
    dark_frame : ndarray, optional
        Per-pixel dark accumulation (**electrons**) added in place of the scalar
        — a master-dark with its FPN / hot-pixel structure (finding C-4), the
        CMOS-faithful alternative to the flat ``dark_rate·t``. Overrides
        ``add_dark`` when supplied.
    bias_adu : float
        Bias pedestal (ADU).

    Returns
    -------
    ndarray (uint16 physical ADU), or (ndarray, truth_dict) if return_truth.
    """
    rng = np.random.default_rng() if rng is None else rng
    ny, nx = shape if shape is not None else (cfg.sensor.ny, cfg.sensor.nx)
    delta = cfg.retardance_deg if retardance_deg is None else retardance_deg
    rot = cfg.instrument_rotator_deg if rotator_deg is None else rotator_deg

    sigma = cfg.fwhm_px(seeing_arcsec) * _FWHM_TO_SIGMA
    dx, dy = cfg.beam.offset_xy()
    t_o, t_e = beam_throughput

    photo = np.zeros((ny, nx), dtype=np.float64)  # photo-electrons (source + sky)
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
        electrons += df  # master-dark with FPN / hot-pixel structure (C-4)
    elif add_dark:
        electrons += cfg.dark_rate_e_per_s * exptime_s

    # full-well saturation, then shot noise, then read noise
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
    """Render the full HWP sequence and write one FITS per angle.

    Returns the list of written FITS paths (one per ``cfg.hwp_angles_deg``).
    """
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
    """Convenience builder for a list of :class:`PointSource`.

    Parameters
    ----------
    positions : list of (x, y)
    stokes_list : list of (1, q, u, v) or (q, u)
    fluxes : list of float (total e-)
    """
    names = names or [f"src{i}" for i in range(len(positions))]
    n = len(positions)
    if not (len(stokes_list) == n and len(fluxes) == n and len(names) == n):
        raise ValueError(
            "make_scene requires positions, stokes_list, fluxes and names of "
            f"equal length; got {len(positions)}, {len(stokes_list)}, "
            f"{len(fluxes)}, {len(names)} (zip would silently truncate)"
        )
    out = []
    for (x, y), st, fl, nm in zip(positions, stokes_list, fluxes, names):
        if len(st) == 2:
            st = (1.0, st[0], st[1], 0.0)
        elif len(st) == 3:
            st = (1.0, st[0], st[1], st[2])
        out.append(PointSource(x=x, y=y, flux_e=fl, stokes=tuple(st), name=nm))
    return out
