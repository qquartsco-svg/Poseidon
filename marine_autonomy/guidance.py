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
from typing import List, Optional, Tuple

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


# ---------------------------------------------------------------------------
# Maritime A* path planner
# ---------------------------------------------------------------------------

@dataclass
class DepthChart:
    """수심도 — A* 경로 계획용 (depth chart for path planning).

    grid          — 2D list[list[float]], depth in meters at each cell
    origin_x      — world x-coordinate of grid cell (0, 0)
    origin_y      — world y-coordinate of grid cell (0, 0)
    resolution_m  — meters per grid cell
    min_depth_m   — minimum additional clearance required above draft
    """

    grid: list           # 2D list[list[float]]
    origin_x: float = 0.0
    origin_y: float = 0.0
    resolution_m: float = 10.0   # meters per cell
    min_depth_m: float = 3.0     # minimum safe water depth margin

    def world_to_cell(self, x: float, y: float) -> Tuple[int, int]:
        """Convert world (x, y) to (row, col) grid indices."""
        col = int((x - self.origin_x) / self.resolution_m)
        row = int((y - self.origin_y) / self.resolution_m)
        return row, col

    def cell_to_world(self, row: int, col: int) -> Tuple[float, float]:
        """Convert (row, col) grid indices to world (x, y) cell centre."""
        x = self.origin_x + col * self.resolution_m + self.resolution_m / 2
        y = self.origin_y + row * self.resolution_m + self.resolution_m / 2
        return x, y

    def is_passable(self, row: int, col: int, draft_m: float = 1.5) -> bool:
        """Return True if the cell has sufficient depth for the given draft."""
        rows = len(self.grid)
        if rows == 0:
            return False
        cols = len(self.grid[0])
        if not (0 <= row < rows and 0 <= col < cols):
            return False
        return self.grid[row][col] >= (draft_m + self.min_depth_m)


def _nearest_passable(
    depth_chart: DepthChart,
    cell: Tuple[int, int],
    draft_m: float,
    search_radius: int = 10,
) -> Optional[Tuple[int, int]]:
    """Find nearest passable cell when start/goal cell has insufficient depth."""
    r0, c0 = cell
    for radius in range(1, search_radius + 1):
        for dr in range(-radius, radius + 1):
            for dc in range(-radius, radius + 1):
                if abs(dr) == radius or abs(dc) == radius:
                    nb = (r0 + dr, c0 + dc)
                    if depth_chart.is_passable(nb[0], nb[1], draft_m):
                        return nb
    return None


def maritime_astar(
    depth_chart: DepthChart,
    start_xy: Tuple[float, float],
    goal_xy: Tuple[float, float],
    draft_m: float = 1.5,
    allow_diagonal: bool = True,
) -> list:
    """Maritime A* path planner — obstacle avoidance via depth chart.

    Finds the lowest-cost path from start to goal on a depth grid.
    Cells with insufficient depth (< draft + min_depth_m) are impassable.
    Deep cells are preferred via a depth bonus in the cost function.

    Heuristic: Euclidean distance (diagonal allowed) or Manhattan distance.

    Parameters
    ----------
    depth_chart   — DepthChart with grid depth data
    start_xy      — (x, y) world-coordinate start position
    goal_xy       — (x, y) world-coordinate goal position
    draft_m       — vessel draft (m); used for passability check
    allow_diagonal — permit 8-directional movement (True) or 4-directional

    Returns
    -------
    list of (x, y) world-coordinate waypoints along the path.
    Returns empty list if no path exists.
    """
    import heapq

    start_cell = depth_chart.world_to_cell(*start_xy)
    goal_cell  = depth_chart.world_to_cell(*goal_xy)

    rows = len(depth_chart.grid)
    cols = len(depth_chart.grid[0]) if rows > 0 else 0

    # Handle impassable start / goal cells
    if not depth_chart.is_passable(start_cell[0], start_cell[1], draft_m):
        start_cell = _nearest_passable(depth_chart, start_cell, draft_m)
        if start_cell is None:
            return []

    if not depth_chart.is_passable(goal_cell[0], goal_cell[1], draft_m):
        goal_cell = _nearest_passable(depth_chart, goal_cell, draft_m)
        if goal_cell is None:
            return []

    # Same cell — trivial path
    if start_cell == goal_cell:
        return [depth_chart.cell_to_world(*start_cell)]

    # Movement directions
    if allow_diagonal:
        neighbors  = [(-1, -1), (-1, 0), (-1, 1), (0, -1),
                      (0,  1), ( 1, -1), ( 1, 0), ( 1,  1)]
        step_costs = [1.4142, 1.0, 1.4142, 1.0,
                      1.0,    1.4142, 1.0, 1.4142]
    else:
        neighbors  = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        step_costs = [1.0, 1.0, 1.0, 1.0]

    def heuristic(a: Tuple[int, int], b: Tuple[int, int]) -> float:
        return math.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2)

    open_set: list = [(0.0, start_cell)]
    came_from: dict = {}
    g_score: dict = {start_cell: 0.0}

    while open_set:
        _, current = heapq.heappop(open_set)

        if current == goal_cell:
            # Reconstruct path
            path_cells = []
            while current in came_from:
                path_cells.append(current)
                current = came_from[current]
            path_cells.append(start_cell)
            path_cells.reverse()
            return [depth_chart.cell_to_world(r, c) for r, c in path_cells]

        r, c = current
        for (dr, dc), cost in zip(neighbors, step_costs):
            nb = (r + dr, c + dc)
            if not depth_chart.is_passable(nb[0], nb[1], draft_m):
                continue

            # Depth bonus: deeper water = lower cost
            depth = depth_chart.grid[nb[0]][nb[1]]
            depth_penalty = max(0.0, 1.0 - (depth - draft_m) / 10.0) * 0.2

            new_g = g_score[current] + cost + depth_penalty
            if new_g < g_score.get(nb, float("inf")):
                came_from[nb] = current
                g_score[nb] = new_g
                f = new_g + heuristic(nb, goal_cell)
                heapq.heappush(open_set, (f, nb))

    return []  # No path found
