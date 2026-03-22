"""
Marine autonomy data contracts.

All types are frozen dataclasses (immutable value objects) so ticks are
deterministic and state cannot be mutated in-place — the same principle as
the Autonomy_Runtime_Stack sibling package.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Tuple, Optional, Any


@dataclass(frozen=True)
class VesselState:
    """3-DOF vessel position/velocity state.

    Coordinate convention (Fossen / NED-like):
      x_m   — East  position (m)
      y_m   — North position (m)
      psi_rad — Heading (rad), 0 = North, positive clockwise
      u_ms  — Surge velocity (m/s, forward along vessel x-axis)
      v_ms  — Sway  velocity (m/s, positive to starboard)
      r_rads — Yaw rate (rad/s, positive clockwise)
      t_s   — Simulation clock (s)
    """

    x_m: float = 0.0
    y_m: float = 0.0
    psi_rad: float = 0.0
    u_ms: float = 0.0
    v_ms: float = 0.0
    r_rads: float = 0.0
    t_s: float = 0.0


@dataclass(frozen=True)
class VesselCommand:
    """Thruster + rudder command issued by the controller.

    thrust_n    — Net surge thrust (N, positive = forward)
    rudder_rad  — Rudder angle (rad, positive = starboard turn)
    """

    thrust_n: float = 0.0
    rudder_rad: float = 0.0


@dataclass(frozen=True)
class ContactVessel:
    """Another vessel observed by radar or AIS.

    id          — Unique identifier string (MMSI, callsign, etc.)
    range_m     — Slant range to contact (m)
    bearing_rad — Relative bearing from own ship (rad, 0 = ahead)
    cog_rad     — Contact's course over ground (rad, 0 = North)
    sog_ms      — Contact's speed over ground (m/s)
    """

    id: str = ""
    range_m: float = 1e6
    bearing_rad: float = 0.0
    cog_rad: float = 0.0
    sog_ms: float = 0.0


@dataclass(frozen=True)
class MarinePerception:
    """Sensor frame for one marine control tick.

    contacts       — Tuple of detected ContactVessel objects
    wind_speed_ms  — Wind speed (m/s)
    wind_dir_rad   — Wind direction (rad, meteorological: from which the wind blows)
    current_u_ms   — Ocean current surge component (m/s, in vessel frame)
    current_v_ms   — Ocean current sway component  (m/s, in vessel frame)
    visibility_m   — Horizontal visibility (m)
    depth_m        — Water depth beneath keel (m)
    """

    contacts: Tuple[ContactVessel, ...] = ()
    wind_speed_ms: float = 0.0
    wind_dir_rad: float = 0.0
    current_u_ms: float = 0.0
    current_v_ms: float = 0.0
    visibility_m: float = 1e6
    depth_m: float = 100.0


@dataclass(frozen=True)
class VesselActuator:
    """Normalized thruster / rudder output sent to the hardware layer.

    throttle     — [0, 1] normalized ahead thrust
    rudder_norm  — [-1, 1] normalized rudder (positive = starboard)
    reverse      — True when backing down
    """

    throttle: float = 0.0
    rudder_norm: float = 0.0
    reverse: bool = False


@dataclass
class MarineTickContext:
    """Mutable context object that flows through one vessel control tick.

    Mirrors the pattern used in AutonomyTickContext from Autonomy_Runtime_Stack.
    Fields are updated in-place as each subsystem runs, then the completed
    context is returned from VesselOrchestrator.tick().

    state        — Current VesselState or StateEstimate from EKF
    perception   — MarinePerception sensor frame
    actuator     — VesselActuator output (populated by orchestrator)
    waypoints    — Ordered sequence of (x, y) waypoints
    colregs_state — Active COLREGs FSM state label
    risk_score   — Normalised risk in [0, 1]
    omega        — Ω capability multiplier
    verdict      — Health verdict string
    ekf          — MarineEKF instance (optional)
    t_s          — Tick timestamp (s)
    """

    state: Optional[Any] = None
    perception: MarinePerception = field(default_factory=MarinePerception)
    actuator: VesselActuator = field(default_factory=VesselActuator)
    waypoints: Tuple[Tuple[float, float], ...] = ()
    colregs_state: str = "STANDBY"
    risk_score: float = 0.0
    omega: float = 1.0
    verdict: str = "HEALTHY"
    ekf: Optional[Any] = None
    t_s: float = 0.0
