"""
AIS (Automatic Identification System) message adapter.

Converts AIS-derived dicts into internal marine autonomy contracts.
No external library required — all parsing is pure Python.

Supported input formats
-----------------------
AIS contact dict (from radar / AIS decoder):
  {
    "id":         str   — MMSI or callsign
    "range_m":    float — slant range (m)
    "bearing_deg": float — relative bearing from own ship (°, clockwise)
    "cog_deg":    float — contact course over ground (°True, clockwise from N)
    "sog_kn":     float — contact speed over ground (knots)
  }

Own-ship NMEA-style dict (from integrated bridge system / GNSS + IMU):
  {
    "lat":        float — latitude  (°, not used for internal x/y)
    "lon":        float — longitude (°, not used for internal x/y)
    "x_m":        float — East position in local frame (m)
    "y_m":        float — North position in local frame (m)
    "hdg_deg":    float — True heading (°)
    "sog_kn":     float — Speed over ground (knots)
    "sog_ms":     float — Speed over ground (m/s)   [alternative]
    "yaw_rate_degs": float — Yaw rate (°/s)          [optional]
    "t_s":        float — timestamp (s)              [optional]
  }

Note: this adapter does NOT verify message checksums or ITU type codes —
those responsibilities belong to the calling application layer.
"""
from __future__ import annotations

import math
from typing import List, Dict, Any, Optional

from ..contracts.schemas import (
    ContactVessel,
    MarinePerception,
    VesselState,
)

# Unit conversion
_KN_TO_MS: float = 0.5144  # 1 knot = 0.5144 m/s
_DEG_TO_RAD: float = math.pi / 180.0


# ---------------------------------------------------------------------------
# Contact parsing
# ---------------------------------------------------------------------------

def parse_ais_contact(msg: Dict[str, Any]) -> ContactVessel:
    """Convert an AIS contact dict to a ContactVessel.

    Required keys: id, range_m, bearing_deg, cog_deg, sog_kn
    All values are validated and clamped to physical limits.

    Returns a ContactVessel with bearing/cog in radians and SOG in m/s.
    """
    contact_id: str = str(msg.get("id", "UNKNOWN"))
    range_m: float = max(0.0, float(msg.get("range_m", 1e6)))
    bearing_deg: float = float(msg.get("bearing_deg", 0.0))
    cog_deg: float = float(msg.get("cog_deg", 0.0))
    sog_kn: float = max(0.0, float(msg.get("sog_kn", 0.0)))

    # Normalise bearing to [0, 2π)
    bearing_rad = (bearing_deg % 360.0) * _DEG_TO_RAD
    # Normalise COG to [0, 2π)
    cog_rad = (cog_deg % 360.0) * _DEG_TO_RAD
    sog_ms = sog_kn * _KN_TO_MS

    return ContactVessel(
        id=contact_id,
        range_m=range_m,
        bearing_rad=bearing_rad,
        cog_rad=cog_rad,
        sog_ms=sog_ms,
    )


# ---------------------------------------------------------------------------
# Perception assembly
# ---------------------------------------------------------------------------

def contacts_to_perception(
    contacts: List[Dict[str, Any]],
    *,
    wind_speed_ms: float = 0.0,
    wind_dir_rad: float = 0.0,
    current_u_ms: float = 0.0,
    current_v_ms: float = 0.0,
    visibility_m: float = 1e6,
    depth_m: float = 100.0,
) -> MarinePerception:
    """Build a MarinePerception from a list of raw AIS contact dicts.

    Each element of contacts must be a dict acceptable to parse_ais_contact().
    Environmental parameters are keyword arguments with sensible defaults.
    """
    parsed = tuple(parse_ais_contact(c) for c in contacts)
    return MarinePerception(
        contacts=parsed,
        wind_speed_ms=wind_speed_ms,
        wind_dir_rad=wind_dir_rad,
        current_u_ms=current_u_ms,
        current_v_ms=current_v_ms,
        visibility_m=visibility_m,
        depth_m=depth_m,
    )


# ---------------------------------------------------------------------------
# Own-ship state from NMEA-style dict
# ---------------------------------------------------------------------------

def ego_state_from_nmea(nmea: Dict[str, Any]) -> VesselState:
    """Create a VesselState from an NMEA/bridge-system dict.

    Accepted keys (all optional with safe defaults):
      x_m, y_m       — local East/North position (m)
      hdg_deg         — true heading (°)
      sog_kn / sog_ms — speed over ground (knots or m/s; sog_ms takes priority)
      yaw_rate_degs   — yaw rate (°/s); defaults to 0
      t_s             — timestamp (s)

    Sway velocity (v_ms) is not directly observable from standard NMEA;
    defaults to 0.0.
    """
    x_m: float = float(nmea.get("x_m", 0.0))
    y_m: float = float(nmea.get("y_m", 0.0))
    hdg_deg: float = float(nmea.get("hdg_deg", 0.0))
    psi_rad: float = hdg_deg * _DEG_TO_RAD

    # Speed: prefer sog_ms, fall back to sog_kn
    if "sog_ms" in nmea:
        u_ms: float = max(0.0, float(nmea["sog_ms"]))
    else:
        sog_kn: float = max(0.0, float(nmea.get("sog_kn", 0.0)))
        u_ms = sog_kn * _KN_TO_MS

    yaw_rate_degs: float = float(nmea.get("yaw_rate_degs", 0.0))
    r_rads: float = yaw_rate_degs * _DEG_TO_RAD

    t_s: float = float(nmea.get("t_s", 0.0))

    return VesselState(
        x_m=x_m,
        y_m=y_m,
        psi_rad=psi_rad,
        u_ms=u_ms,
        v_ms=0.0,
        r_rads=r_rads,
        t_s=t_s,
    )
