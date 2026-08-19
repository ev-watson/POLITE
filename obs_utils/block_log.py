"""Structured per-block observing log (JSONL manifest)."""
from __future__ import annotations

import json
import logging
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterator, Optional, Union

from astropy import units as u
from astropy.coordinates import EarthLocation, SkyCoord, get_body
from astropy.time import Time


logger = logging.getLogger(__name__)

JULIAN_LAT_DEG = 33.0701
JULIAN_LON_DEG = -116.6451
JULIAN_ELEV_M = 1294.0


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def moon_separation_deg(
    ra_hours: float,
    dec_deg: float,
    when: Optional[Time] = None,
    *,
    lat_deg: float = JULIAN_LAT_DEG,
    lon_deg: float = JULIAN_LON_DEG,
    elev_m: float = JULIAN_ELEV_M,
) -> Optional[float]:
    """Target-Moon separation [deg] at ``when`` (default: now UTC)."""
    try:
        t = when or Time.now()
        site = EarthLocation(lat=lat_deg * u.deg, lon=lon_deg * u.deg, height=elev_m * u.m)
        target = SkyCoord(ra=ra_hours * u.hourangle, dec=dec_deg * u.deg, frame="icrs")
        moon = get_body("moon", t, site)
        return float(target.separation(moon).deg)
    except Exception:
        logger.debug("Moon separation computation failed", exc_info=True)
        return None


class BlockLogger:
    """Append one JSON object per observing block to a JSONL manifest."""

    def __init__(self, path: Union[str, Path]) -> None:
        self.path = Path(path).expanduser().resolve()
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def append(self, record: Dict[str, Any]) -> None:
        with open(self.path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(record, sort_keys=False) + "\n")

    @contextmanager
    def block(
        self,
        block_id: str,
        *,
        target: Optional[str] = None,
        brick: Optional[str] = None,
        filter_name: Optional[str] = None,
        extra: Optional[Dict[str, Any]] = None,
    ) -> Iterator[Dict[str, Any]]:
        """Context manager that records ut_start/ut_end around a block."""
        rec: Dict[str, Any] = {
            "block_id": block_id,
            "target": target,
            "brick": brick,
            "filter": filter_name,
            "ut_start": _utc_now_iso(),
        }
        if extra:
            rec.update(extra)
        try:
            yield rec
        finally:
            rec["ut_end"] = _utc_now_iso()
            self.append(rec)


def pwi4_snapshot(pwi4) -> Dict[str, Any]:
    """Capture mount/rotator fields from PWI4 status."""
    out: Dict[str, Any] = {}
    if pwi4 is None:
        return out
    try:
        st = pwi4.status()
        if hasattr(st, "rotator") and st.rotator.exists:
            out["instrot_deg"] = float(st.rotator.field_angle_degs)
        if hasattr(st, "mount"):
            from .obs_math import airmass_kasten_young, zenith_distance_to_altitude

            z = getattr(st.mount, "altitude_degs", None)
            out["pwi4_zenith_distance_deg"] = z
            out["airmass"] = airmass_kasten_young(zenith_distance_to_altitude(z))
    except Exception:
        logger.debug("PWI4 snapshot failed", exc_info=True)
    return out
