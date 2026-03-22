"""
LOS (Line-of-Sight) guidance law with waypoint sequencing.

Governing equation:
  α_k   = atan2(y_{k+1} - y_k,  x_{k+1} - x_k)   — path angle
  e     = -(x - x_k)·sin(α_k) + (y - y_k)·cos(α_k) — cross-track error
             (positive e → vessel is to port of desired track)
  ψ_d   = α_k + atan2(-e, Δ)                         — desired heading
  ψ_err = normalize(ψ_d - ψ)
  δ     = clamp(Kp·ψ_err - Kd·r,  -δ_max, δ_max)   — PD rudder command

where:
  Δ   — lookahead distance (m)
  r   — yaw rate from vessel state (rad/s)
  Kp  — proportional gain (1.2)
  Kd  — derivative gain  (0.4)

Reference:
  Fossen, T.I. (2011), Ch. 12 — Guidance Laws for Path Following
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional, Tuple

from .contracts.schemas import VesselState


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _clamp(v: float, lo: float, hi: float) -> float:
    """Clamp v to [lo, hi]."""
    return max(lo, min(hi, v))


def _normalize_angle(a: float) -> float:
    """Wrap angle to (−π, π]."""
    while a > math.pi:
        a -= 2.0 * math.pi
    while a <= -math.pi:
        a += 2.0 * math.pi
    return a


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class LOSConfig:
    """Configuration for the LOS guidance law.

    lookahead_m         — Lookahead distance Δ (m); larger → smoother but less responsive
    acceptance_radius_m — Switch to next waypoint when range < this (m)
    max_rudder_rad      — Maximum rudder command magnitude (rad)
    Kp                  — Heading PD proportional gain
    Kd                  — Heading PD derivative gain
    """

    lookahead_m: float = 30.0
    acceptance_radius_m: float = 15.0
    max_rudder_rad: float = 0.6109  # ≈ 35 °
    Kp: float = 1.2
    Kd: float = 0.4


# ---------------------------------------------------------------------------
# LOS Guidance
# ---------------------------------------------------------------------------

class LOSGuidance:
    """Line-of-Sight guidance controller with waypoint sequencing.

    Usage::

        los = LOSGuidance(LOSConfig(lookahead_m=50.0))
        for step in range(N):
            rudder_rad = los.update(state, waypoints)
    """

    def __init__(self, config: Optional[LOSConfig] = None) -> None:
        self._config: LOSConfig = config or LOSConfig()
        self._wp_idx: int = 0
        self._psi_err_prev: float = 0.0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def reset(self) -> None:
        """Reset waypoint index and derivative state."""
        self._wp_idx = 0
        self._psi_err_prev = 0.0

    def is_complete(self, waypoints: Tuple[Tuple[float, float], ...]) -> bool:
        """Return True when all waypoints have been reached."""
        return len(waypoints) == 0 or self._wp_idx >= len(waypoints)

    def active_waypoint(
        self, waypoints: Tuple[Tuple[float, float], ...]
    ) -> Optional[Tuple[float, float]]:
        """Return the currently active waypoint, or None if finished."""
        if self.is_complete(waypoints):
            return None
        return waypoints[self._wp_idx]

    def update(
        self,
        state: VesselState,
        waypoints: Tuple[Tuple[float, float], ...],
    ) -> float:
        """Compute rudder command (rad) via LOS law.

        Steps:
          1. Advance waypoint index if within acceptance radius.
          2. Compute cross-track error and path angle.
          3. Compute desired heading via LOS formula.
          4. Apply PD heading controller to obtain rudder deflection.

        Returns 0.0 if no waypoints remain.
        """
        if self.is_complete(waypoints):
            return 0.0

        # --- Waypoint switching ---
        self._advance_if_reached(state, waypoints)
        if self.is_complete(waypoints):
            return 0.0

        wp_to = waypoints[self._wp_idx]

        # Previous waypoint (or own initial position if first segment)
        if self._wp_idx > 0:
            wp_from = waypoints[self._wp_idx - 1]
        else:
            # Use a virtual "from" point behind the vessel
            wp_from = (state.x_m, state.y_m)

        # --- LOS desired heading ---
        alpha = self._path_angle(wp_from, wp_to)
        e = self._cross_track_error(state, wp_from, wp_to)
        psi_d = alpha + math.atan2(-e, self._config.lookahead_m)

        # --- PD heading controller ---
        psi_err = _normalize_angle(psi_d - state.psi_rad)
        rudder = _clamp(
            self._config.Kp * psi_err - self._config.Kd * state.r_rads,
            -self._config.max_rudder_rad,
            self._config.max_rudder_rad,
        )
        self._psi_err_prev = psi_err
        return rudder

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _advance_if_reached(
        self,
        state: VesselState,
        waypoints: Tuple[Tuple[float, float], ...],
    ) -> None:
        """Advance _wp_idx while the current waypoint is within acceptance radius."""
        while self._wp_idx < len(waypoints):
            wp = waypoints[self._wp_idx]
            dx = wp[0] - state.x_m
            dy = wp[1] - state.y_m
            dist = math.hypot(dx, dy)
            if dist < self._config.acceptance_radius_m:
                self._wp_idx += 1
            else:
                break

    def _path_angle(
        self,
        wp_from: Tuple[float, float],
        wp_to: Tuple[float, float],
    ) -> float:
        """Compute path angle α_k = atan2(Δy, Δx) between two waypoints.

        α_k is measured from the East axis (x) towards North (y).
        """
        dx = wp_to[0] - wp_from[0]
        dy = wp_to[1] - wp_from[1]
        return math.atan2(dy, dx)

    def _cross_track_error(
        self,
        state: VesselState,
        wp_from: Tuple[float, float],
        wp_to: Tuple[float, float],
    ) -> float:
        """Signed cross-track error e (m).

        Positive e → vessel is to port of the intended track.

        e = -(x - x_k)·sin(α_k) + (y - y_k)·cos(α_k)
        """
        alpha = self._path_angle(wp_from, wp_to)
        dx = state.x_m - wp_from[0]
        dy = state.y_m - wp_from[1]
        return -dx * math.sin(alpha) + dy * math.cos(alpha)
