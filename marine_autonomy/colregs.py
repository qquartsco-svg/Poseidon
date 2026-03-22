"""
COLREGs Rule 8/13/14/15/16 simplified FSM.

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

Disclaimer:
  This implementation provides heuristic guidance only and is NOT a
  certified COLREGs compliance system.  All manoeuvres require confirmation
  by a qualified officer of the watch.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict

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
    config: COLREGsConfig,
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
    """
    if contact.range_m > config.safe_range_m:
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
# COLREGs FSM
# ---------------------------------------------------------------------------

class COLREGsBehavior:
    """Finite state machine that translates COLREGs encounters into
    avoidance directives.

    Output dict from tick():
      state               — FSM state string
      avoid_heading_offset_rad — additional heading bias to add to LOS output
      stop                — True when EMERGENCY_STOP is active
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
        config: COLREGsConfig,
    ) -> Dict:
        """Evaluate all contacts and return COLREGs directives.

        Priority order (highest first):
          1. EMERGENCY_STOP — any contact inside emergency_range_m
          2. GIVE_WAY       — HEAD_ON or CROSSING_GIVE_WAY or OVERTAKING
          3. STAND_ON       — CROSSING_STAND_ON
          4. CRUISE         — all contacts safe / beyond action_range_m
          5. STANDBY        — no contacts at all
        """
        emergency = False
        give_way = False
        stand_on = False
        has_contacts = len(perception.contacts) > 0
        avoid_offset = 0.0

        for contact in perception.contacts:
            # Emergency stop overrides everything
            if contact.range_m < config.emergency_range_m:
                emergency = True
                break

            if contact.range_m > config.action_range_m:
                continue

            situation = classify_contact(ego_state, contact, config)

            if situation == "HEAD_ON":
                give_way = True
                # Rule 14: alter to starboard — positive heading offset
                avoid_offset = max(avoid_offset, config.give_way_offset_rad)

            elif situation == "CROSSING_GIVE_WAY":
                give_way = True
                avoid_offset = max(avoid_offset, config.give_way_offset_rad)

            elif situation == "OVERTAKING":
                give_way = True
                # Keep clear — alter to starboard
                avoid_offset = max(avoid_offset, config.give_way_offset_rad)

            elif situation == "CROSSING_STAND_ON":
                stand_on = True

        # --- FSM transition ---
        if emergency:
            self._state = "EMERGENCY_STOP"
        elif give_way:
            self._state = "GIVE_WAY"
        elif stand_on:
            self._state = "STAND_ON"
        elif has_contacts:
            self._state = "CRUISE"
        else:
            self._state = "CRUISE"

        return {
            "state": self._state,
            "avoid_heading_offset_rad": avoid_offset if not emergency else 0.0,
            "stop": emergency,
        }
