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


# ---------------------------------------------------------------------------
# Hull-class preset registry (universal maritime engine)
# ---------------------------------------------------------------------------

HULL_PRESETS: Dict[str, Dict[str, MarinePreset]] = {
    "surface_vessel": {
        "coastal": PRESET_REGISTRY["coastal"],
        "harbor":  PRESET_REGISTRY["harbor"],
    },
    "submarine": {
        "shallow": MarinePreset(
            name="sub_shallow",
            max_speed_ms=5.144,
            cruise_speed_ms=3.086,
            lookahead_m=100.0,
            acceptance_radius_m=30.0,
            safe_range_m=500.0,
            action_range_m=1000.0,
            emergency_range_m=100.0,
            draft_m=5.0,
            description="잠수함 천해 작전 — 10kn",
        ),
        "deep": MarinePreset(
            name="sub_deep",
            max_speed_ms=10.288,
            cruise_speed_ms=7.716,
            lookahead_m=300.0,
            acceptance_radius_m=100.0,
            safe_range_m=2000.0,
            action_range_m=5000.0,
            emergency_range_m=200.0,
            draft_m=8.0,
            description="잠수함 심해 순항 — 20kn",
        ),
    },
    "yacht": {
        "racing": MarinePreset(
            name="yacht_racing",
            max_speed_ms=8.230,
            cruise_speed_ms=6.173,
            lookahead_m=60.0,
            acceptance_radius_m=20.0,
            safe_range_m=300.0,
            action_range_m=600.0,
            emergency_range_m=40.0,
            draft_m=1.8,
            description="레이싱 요트 — 16kn",
        ),
        "cruising": MarinePreset(
            name="yacht_cruising",
            max_speed_ms=5.144,
            cruise_speed_ms=3.601,
            lookahead_m=50.0,
            acceptance_radius_m=15.0,
            safe_range_m=200.0,
            action_range_m=400.0,
            emergency_range_m=30.0,
            draft_m=1.5,
            description="크루징 요트 — 10kn",
        ),
    },
    "boat": {
        "patrol": MarinePreset(
            name="boat_patrol",
            max_speed_ms=15.432,
            cruise_speed_ms=10.288,
            lookahead_m=40.0,
            acceptance_radius_m=10.0,
            safe_range_m=150.0,
            action_range_m=300.0,
            emergency_range_m=20.0,
            draft_m=0.8,
            description="고속 순찰정 — 30kn",
        ),
        "harbor": MarinePreset(
            name="boat_harbor",
            max_speed_ms=2.572,
            cruise_speed_ms=1.543,
            lookahead_m=15.0,
            acceptance_radius_m=5.0,
            safe_range_m=50.0,
            action_range_m=100.0,
            emergency_range_m=10.0,
            draft_m=0.6,
            description="항만 보트 — 5kn",
        ),
    },
    "autonomous_usv": {
        "survey": MarinePreset(
            name="usv_survey",
            max_speed_ms=3.086,
            cruise_speed_ms=2.057,
            lookahead_m=20.0,
            acceptance_radius_m=5.0,
            safe_range_m=100.0,
            action_range_m=200.0,
            emergency_range_m=15.0,
            draft_m=0.4,
            description="수중 탐사 USV — 6kn",
        ),
    },
}


def get_hull_preset(hull_class: str, scenario: str) -> MarinePreset:
    """Return a hull-class specific operating preset.

    Parameters
    ----------
    hull_class  — one of "surface_vessel", "submarine", "yacht", "boat",
                  "autonomous_usv"
    scenario    — scenario key within the hull class (e.g. "deep", "racing")

    Falls back to "surface_vessel" / "coastal" if not found.

    Example::

        get_hull_preset("submarine", "deep")  # → sub_deep preset
        get_hull_preset("yacht", "racing")    # → yacht_racing preset
    """
    return HULL_PRESETS.get(hull_class, {}).get(
        scenario,
        HULL_PRESETS["surface_vessel"]["coastal"],
    )
