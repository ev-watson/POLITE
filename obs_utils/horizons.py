"""Thin JPL Horizons wrapper for asteroid ephemerides at acquisition."""
from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional, Union

from astropy.time import Time


logger = logging.getLogger(__name__)

HORIZONS_IDS: Dict[str, str] = {
    "Melpomene": "18",
    "Juno": "3",
}


@dataclass(frozen=True)
class EphemerisRow:
    target_id: str
    target_name: str
    ra_hours: float
    dec_deg: float
    phase_angle_deg: Optional[float] = None
    scat_plane_pa_deg: Optional[float] = None
    daily_motion_arcsec_per_h: Optional[float] = None
    query_utc: str = ""


def fetch_ephemeris(
    target_id: str,
    epoch: Time,
    lon_deg: float,
    lat_deg: float,
    elev_m: float,
    *,
    target_name: str = "",
) -> EphemerisRow:
    """Query JPL Horizons for RA, Dec, phase angle, and scattering-plane PA."""
    try:
        from astroquery.jplhorizons import Horizons
    except ImportError as exc:
        raise RuntimeError(
            "astroquery is required for automated Horizons ephemerides; "
            "install astroquery or log ephemeris manually."
        ) from exc

    loc = f"@{lon_deg:.6f},{lat_deg:.6f},{elev_m:.1f}"
    epoch_str = epoch.utc.strftime("%Y-%m-%d %H:%M")
    q = Horizons(id=target_id, location=loc, epochs=epoch_str)
    eph = q.ephemerides()

    ra_deg = float(eph["RA"][0])
    dec_d = float(eph["DEC"][0])
    ra_h = ra_deg / 15.0
    phase = float(eph["alpha"][0]) if "alpha" in eph.colnames else None
    scat_pa = float(eph["PsAng"][0]) if "PsAng" in eph.colnames else None
    motion = None
    if "RA_rate" in eph.colnames and "DEC_rate" in eph.colnames:
        import numpy as np
        ra_r = float(eph["RA_rate"][0])
        dec_r = float(eph["DEC_rate"][0])
        motion = float(np.hypot(ra_r, dec_r))

    return EphemerisRow(
        target_id=str(target_id),
        target_name=target_name or str(target_id),
        ra_hours=ra_h,
        dec_deg=dec_d,
        phase_angle_deg=phase,
        scat_plane_pa_deg=scat_pa,
        daily_motion_arcsec_per_h=motion,
        query_utc=datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    )


def resolve_target_id(name: str, catalog_entry: Optional[Dict[str, Any]] = None) -> Optional[str]:
    if catalog_entry and catalog_entry.get("horizons_id"):
        return str(catalog_entry["horizons_id"])
    return HORIZONS_IDS.get(name)


def cache_ephemeris(path: Union[str, Path], row: EphemerisRow) -> Path:
    out = Path(path).expanduser().resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    records: list = []
    if out.exists():
        with open(out, "r", encoding="utf-8") as fh:
            records = json.load(fh)
    records.append(asdict(row))
    with open(out, "w", encoding="utf-8") as fh:
        json.dump(records, fh, indent=2)
    return out
