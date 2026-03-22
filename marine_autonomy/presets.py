"""
Vessel operating presets — analogous to drive-mode presets in the
Autonomy_Runtime_Stack.

Each preset encodes the performance envelope and safety thresholds
appropriate for a specific operating environment.

Unit notes:
  Speeds in m/s internally; helper methods expose knots / km/h.
  1 knot = 0.5144 m/s
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict


@dataclass
class MarinePreset:
    """Operating profile for a specific marine environment.

    name                — identifier string
    max_speed_ms        — maximum allowed speed (m/s)
    cruise_speed_ms     — target cruise speed (m/s)
    lookahead_m         — LOS lookahead distance Δ (m)
    acceptance_radius_m — waypoint acceptance radius (m)
    safe_range_m        — COLREGs: contacts beyond this are safe (m)
    action_range_m      — COLREGs: trigger assessment distance (m)
    emergency_range_m   — COLREGs: emergency stop distance (m)
    draft_m             — vessel design draught (m); used for depth check
    description         — human-readable summary
    """

    name: str
    max_speed_ms: float
    cruise_speed_ms: float
    lookahead_m: float
    acceptance_radius_m: float
    safe_range_m: float
    action_range_m: float
    emergency_range_m: float
    draft_m: float
    description: str = ""

    def speed_kmh(self) -> float:
        """Cruise speed in km/h."""
        return self.cruise_speed_ms * 3.6

    def speed_kn(self) -> float:
        """Cruise speed in knots."""
        return self.cruise_speed_ms / 0.5144

    def max_speed_kn(self) -> float:
        """Maximum speed in knots."""
        return self.max_speed_ms / 0.5144


# ---------------------------------------------------------------------------
# Preset registry
# ---------------------------------------------------------------------------

PRESET_REGISTRY: Dict[str, MarinePreset] = {
    "harbor": MarinePreset(
        name="harbor",
        max_speed_ms=2.572,   # ≈ 5 kn
        cruise_speed_ms=1.543,  # ≈ 3 kn
        lookahead_m=20.0,
        acceptance_radius_m=10.0,
        safe_range_m=100.0,
        action_range_m=200.0,
        emergency_range_m=20.0,
        draft_m=1.5,
        description="Harbor maneuvering — 5 kn max",
    ),
    "coastal": MarinePreset(
        name="coastal",
        max_speed_ms=7.716,   # ≈ 15 kn
        cruise_speed_ms=5.144,  # ≈ 10 kn
        lookahead_m=80.0,
        acceptance_radius_m=30.0,
        safe_range_m=500.0,
        action_range_m=1000.0,
        emergency_range_m=50.0,
        draft_m=2.0,
        description="Coastal patrol — 15 kn max",
    ),
    "ocean": MarinePreset(
        name="ocean",
        max_speed_ms=12.860,  # ≈ 25 kn
        cruise_speed_ms=10.288,  # ≈ 20 kn
        lookahead_m=200.0,
        acceptance_radius_m=100.0,
        safe_range_m=2000.0,
        action_range_m=5000.0,
        emergency_range_m=200.0,
        draft_m=4.0,
        description="Ocean transit — 25 kn max",
    ),
    "river": MarinePreset(
        name="river",
        max_speed_ms=4.115,   # ≈ 8 kn
        cruise_speed_ms=2.572,  # ≈ 5 kn
        lookahead_m=40.0,
        acceptance_radius_m=15.0,
        safe_range_m=150.0,
        action_range_m=300.0,
        emergency_range_m=30.0,
        draft_m=1.0,
        description="River / inland waterway — 8 kn max",
    ),
}


def get_preset(name: str) -> MarinePreset:
    """Return the named preset, falling back to 'coastal' if not found."""
    return PRESET_REGISTRY.get(name, PRESET_REGISTRY["coastal"])
