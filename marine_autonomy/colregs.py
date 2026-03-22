"""
COLREGs Rule 8/13/14/15/16 simplified FSM — multi-vessel simultaneous assessment.

Situation classification (relative to own ship):
  HEAD_ON           — contact bearing < 15° AND courses opposing (diff > 150°)
                      Rule 14: both vessels alter to starboard
  CROSSING_GIVE_WAY — contact bearing 10°–112.5° on own starboard bow
                      Rule 15: give-way vessel alters course/speed
  CROSSING_STAND_ON — contact bearing 10°–112.5° and own vessel is stand-on
                      Rule 16: stand-on vessel maintains course/speed
  OVERTAKING        — contact bearing 112.5°–247.5° (from astern)
                      Rule 13: overtaking vessel keeps clear
  SAFE              — contact beyond safe_range_m OR no hazard geometry

FSM states:
  STANDBY         — initialising / no movement
  CRUISE          — normal transit, no conflicts
  GIVE_WAY        — active collision avoidance (starboard alteration)
  STAND_ON        — maintain course, monitor situation
  EMERGENCY_STOP  — contact inside emergency_range_m

Multi-vessel algorithm:
  1. All contacts are classified individually (HEAD_ON/CROSSING/OVERTAKING/SAFE)
  2. EMERGENCY_STOP checked first (any contact d < emergency_range_m)
  3. GIVE_WAY contacts collected — dominant (closest) determines avoidance direction
  4. Multiple GIVE_WAY: largest avoidance angle applied
  5. STAND_ON overridden by GIVE_WAY if any give-way contact also present

Disclaimer:
  This implementation provides heuristic guidance only and is NOT a
  certified COLREGs compliance system.  All manoeuvres require confirmation
  by a qualified officer of the watch.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Optional

from .contracts.schemas import VesselState, ContactVessel, MarinePerception


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class COLREGsConfig:
    """Thresholds for the COLREGs classifier and FSM.

    safe_range_m      — Contacts beyond this are deemed safe (m)
    action_range_m    — Contacts within this trigger give-way assessment (m)
    emergency_range_m — Contacts within this trigger EMERGENCY_STOP (m)
    give_way_offset_rad — Minimum starboard heading offset when giving way (rad)
    """

    safe_range_m: float = 300.0
    action_range_m: float = 500.0
    emergency_range_m: float = 50.0
    give_way_offset_rad: float = 0.2618  # ≈ 15 °


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_DEFAULT_CONFIG = COLREGsConfig()


def _normalize_angle(a: float) -> float:
    """Wrap angle to (−π, π]."""
    while a > math.pi:
        a -= 2.0 * math.pi
    while a <= -math.pi:
        a += 2.0 * math.pi
    return a


def _angle_diff(a: float, b: float) -> float:
    """Absolute angular difference between two angles (rad)."""
    return abs(_normalize_angle(a - b))


# ---------------------------------------------------------------------------
# Contact classification
# ---------------------------------------------------------------------------

def classify_contact(
    ego_state: VesselState,
    contact: ContactVessel,
    config: Optional[COLREGsConfig] = None,
) -> str:
    """Classify the encounter geometry between own ship and a contact.

    Returns one of:
      "HEAD_ON", "CROSSING_GIVE_WAY", "CROSSING_STAND_ON",
      "OVERTAKING", "SAFE"

    Bearing sectors (relative to own ship's head, measured clockwise):
      HEAD_ON:    |rel_bearing| < 15° (0.2618 rad) and courses opposing
      CROSSING:   bearing 10°–112.5° on own starboard  → GIVE_WAY
                  bearing 247.5°–350° on own port       → STAND_ON
      OVERTAKING: bearing 112.5°–247.5° (from astern)

    config is optional; uses default COLREGsConfig if not supplied.
    """
    cfg = config if config is not None else _DEFAULT_CONFIG

    if contact.range_m > cfg.safe_range_m:
        return "SAFE"

    bearing = contact.bearing_rad  # relative bearing (rad)
    bearing_deg = math.degrees(bearing) % 360.0

    # Absolute relative bearing (0–180)
    abs_bearing_deg = min(bearing_deg, 360.0 - bearing_deg)

    # Compute course difference (opposing if > 150°)
    own_cog = ego_state.psi_rad
    contact_cog = contact.cog_rad
    course_diff_deg = math.degrees(_angle_diff(own_cog, contact_cog))

    # --- HEAD_ON (Rule 14) ---
    if abs_bearing_deg < 15.0 and course_diff_deg > 150.0:
        return "HEAD_ON"

    # --- OVERTAKING (Rule 13) --- contact is astern
    if 112.5 <= bearing_deg <= 247.5:
        return "OVERTAKING"

    # --- CROSSING (Rule 15) ---
    # Own starboard sector: bearing 10°–112.5°  → we are give-way
    if 10.0 <= bearing_deg <= 112.5:
        return "CROSSING_GIVE_WAY"

    # Own port sector: bearing 247.5°–350° → we are stand-on
    if 247.5 <= bearing_deg <= 350.0:
        return "CROSSING_STAND_ON"

    return "SAFE"


# ---------------------------------------------------------------------------
# COLREGs FSM — multi-vessel
# ---------------------------------------------------------------------------

class COLREGsBehavior:
    """Finite state machine that translates COLREGs encounters into
    avoidance directives, supporting simultaneous multi-vessel assessment.

    Output dict from tick():
      state                    — FSM state string
      avoid_heading_offset_rad — additional heading bias to add to LOS output
      stop                     — True when EMERGENCY_STOP is active
      situations               — list[dict] per-contact situation info
      dominant_contact         — id of most dangerous contact (or None)
    """

    def __init__(self) -> None:
        self._state: str = "STANDBY"

    @property
    def state(self) -> str:
        return self._state

    def tick(
        self,
        ego_state: VesselState,
        perception: MarinePerception,
        config: Optional[COLREGsConfig] = None,
    ) -> Dict:
        """Evaluate all contacts simultaneously and return COLREGs directives.

        Multi-vessel algorithm:
          1. Classify every contact individually.
          2. EMERGENCY_STOP first — any contact d < emergency_range_m.
          3. Collect GIVE_WAY situations; pick dominant (closest) for direction.
          4. Multiple GIVE_WAY: use maximum avoidance angle.
          5. STAND_ON only when no GIVE_WAY present.
        """
        cfg = config if config is not None else _DEFAULT_CONFIG
        contacts = perception.contacts if hasattr(perception, "contacts") else ()

        if not contacts:
            self._state = "CRUISE"
            return {
                "state": "CRUISE",
                "avoid_heading_offset_rad": 0.0,
                "stop": False,
                "situations": [],
                "dominant_contact": None,
            }

        # Step 1: classify all contacts
        situations = []
        for c in contacts:
            sit = classify_contact(ego_state, c, cfg)
            situations.append({
                "id": c.id,
                "range_m": c.range_m,
                "situation": sit,
                "contact": c,
            })

        # Step 2: emergency stop — any contact inside emergency range
        emergency_contacts = [s for s in situations if s["range_m"] < cfg.emergency_range_m]
        if emergency_contacts:
            self._state = "EMERGENCY_STOP"
            return {
                "state": "EMERGENCY_STOP",
                "avoid_heading_offset_rad": 0.0,
                "stop": True,
                "situations": situations,
                "dominant_contact": emergency_contacts[0]["id"],
            }

        # Step 3: filter to action range
        active = [s for s in situations if s["range_m"] < cfg.action_range_m]

        # Step 4: separate give-way and stand-on
        give_way = [
            s for s in active
            if s["situation"] in ("HEAD_ON", "OVERTAKING", "CROSSING_GIVE_WAY")
        ]
        stand_on = [
            s for s in active
            if s["situation"] == "CROSSING_STAND_ON"
        ]

        if give_way:
            # Dominant = closest
            dominant = min(give_way, key=lambda x: x["range_m"])
            sit_type = dominant["situation"]

            # Primary avoidance offset
            if sit_type == "HEAD_ON":
                offset = math.radians(20.0)
            elif sit_type == "CROSSING_GIVE_WAY":
                offset = math.radians(15.0)
            else:  # OVERTAKING
                offset = math.radians(10.0)

            # Multiple give-way: take the maximum
            for s in give_way[1:]:
                if s["situation"] == "HEAD_ON":
                    offset = max(offset, math.radians(20.0))
                elif s["situation"] == "CROSSING_GIVE_WAY":
                    offset = max(offset, math.radians(15.0))
                else:
                    offset = max(offset, math.radians(10.0))

            self._state = "GIVE_WAY"
            return {
                "state": "GIVE_WAY",
                "avoid_heading_offset_rad": offset,
                "stop": False,
                "situations": situations,
                "dominant_contact": dominant["id"],
            }

        if stand_on:
            self._state = "STAND_ON"
            return {
                "state": "STAND_ON",
                "avoid_heading_offset_rad": 0.0,
                "stop": False,
                "situations": situations,
                "dominant_contact": stand_on[0]["id"],
            }

        self._state = "CRUISE"
        return {
            "state": "CRUISE",
            "avoid_heading_offset_rad": 0.0,
            "stop": False,
            "situations": situations,
            "dominant_contact": None,
        }
