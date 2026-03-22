"""
VesselOrchestrator — tick-based marine autonomy controller.

Mirrors the AutonomyOrchestrator pattern from the sibling
Autonomy_Runtime_Stack, adapted for 3-DOF vessel dynamics.

One tick comprises:
  1.  EKF prediction + optional sensor fusion
  2.  COLREGs behaviour assessment
  3.  LOS guidance → desired rudder
  4.  Speed / thrust PID
  5.  Ω (omega) safety capability multiplier
  6.  Actuator normalisation and verdict emission

Ω model:
  ω_risk  = 0.55 if risk > 0.85 else 0.78 if risk > 0.35 else 1.0
  ω_fuel  = 0.70 if fuel_level < 0.2   else 1.0
  ω_vis   = 0.80 if visibility_m < 200  else 1.0
  ω_depth = 0.65 if depth_m < 3·draft_m else 1.0
  Ω       = ω_risk × ω_fuel × ω_vis × ω_depth

Speed PID (surge control):
  error  = target_speed − u
  thrust = clamp(Kp·error − Kd·(u − u_prev)/dt,  −max_thrust, +max_thrust)
  Kp = 500, Kd = 200
"""
from __future__ import annotations

import math
from typing import Optional

from .contracts.schemas import (
    MarineTickContext,
    VesselState,
    VesselActuator,
    VesselCommand,
)
from .guidance import LOSGuidance, LOSConfig
from .colregs import COLREGsBehavior, COLREGsConfig
from .estimation import MarineEKF
from .presets import MarinePreset, get_preset


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


def _normalize_angle(a: float) -> float:
    while a > math.pi:
        a -= 2.0 * math.pi
    while a <= -math.pi:
        a += 2.0 * math.pi
    return a


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

class VesselOrchestrator:
    """Tick-based vessel autonomy controller.

    Usage::

        orch = VesselOrchestrator(preset="harbor")
        ctx = MarineTickContext(
            state=VesselState(),
            waypoints=((0.0, 0.0), (100.0, 50.0)),
        )
        for _ in range(200):
            ctx = orch.tick(ctx, dt_s=0.1)

    Each call to tick() is stateless from the caller's perspective — the
    returned context carries all updated information.  Internal EKF and
    guidance objects maintain state across calls.
    """

    # Speed PID gains
    _KP_SPEED: float = 500.0
    _KD_SPEED: float = 200.0

    def __init__(
        self,
        preset: Optional[str] = None,
        use_ekf: bool = True,
        fuel_level: float = 1.0,
    ) -> None:
        self._preset: MarinePreset = get_preset(preset) if isinstance(preset, str) else (preset or get_preset("coastal"))
        self._los = LOSGuidance(
            LOSConfig(
                lookahead_m=self._preset.lookahead_m,
                acceptance_radius_m=self._preset.acceptance_radius_m,
            )
        )
        self._colregs = COLREGsBehavior()
        self._colregs_config = COLREGsConfig(
            safe_range_m=self._preset.safe_range_m,
            action_range_m=self._preset.action_range_m,
            emergency_range_m=self._preset.emergency_range_m,
        )
        self._ekf: Optional[MarineEKF] = MarineEKF() if use_ekf else None
        self._fuel_level: float = fuel_level

        # Speed PID state
        self._u_prev: float = 0.0
        self._t_s: float = 0.0

    # ------------------------------------------------------------------
    # Public tick interface
    # ------------------------------------------------------------------

    def tick(
        self,
        ctx: MarineTickContext,
        *,
        dt_s: float = 0.1,
    ) -> MarineTickContext:
        """Run one autonomy tick and return the updated context.

        The input ctx is not mutated; a fresh MarineTickContext is returned.
        """
        state = ctx.state
        if state is None:
            state = VesselState()

        # Resolve the numeric state (VesselState or StateEstimate both have
        # x_m, y_m, psi_rad, u_ms attributes)
        x_m = state.x_m
        y_m = state.y_m
        psi_rad = state.psi_rad
        u_ms = getattr(state, "u_ms", 0.0)

        # ----------------------------------------------------------------
        # 1. EKF prediction
        # ----------------------------------------------------------------
        ekf = ctx.ekf or self._ekf
        if ekf is not None:
            r_rads = getattr(state, "r_rads", 0.0)
            ekf.predict(r_rads=r_rads, dt_s=dt_s)
            ekf.update_gps([x_m, y_m])
            ekf.update_heading(psi_rad)
            ekf.update_speed(u_ms)
            est = ekf.estimate()
            # Feed EKF estimate back as state for downstream modules
            fused_state = VesselState(
                x_m=est.x_m,
                y_m=est.y_m,
                psi_rad=est.psi_rad,
                u_ms=est.u_ms,
                v_ms=getattr(state, "v_ms", 0.0),
                r_rads=getattr(state, "r_rads", 0.0),
                t_s=state.t_s,
            )
        else:
            fused_state = state if isinstance(state, VesselState) else VesselState(
                x_m=x_m, y_m=y_m, psi_rad=psi_rad, u_ms=u_ms,
                t_s=getattr(state, "t_s", 0.0),
            )
            est = None

        # ----------------------------------------------------------------
        # 2. COLREGs behaviour
        # ----------------------------------------------------------------
        colregs_result = self._colregs.tick(
            fused_state, ctx.perception, self._colregs_config
        )
        colregs_state = colregs_result["state"]
        avoid_offset = colregs_result["avoid_heading_offset_rad"]
        emergency_stop = colregs_result["stop"]

        # ----------------------------------------------------------------
        # 3. LOS guidance → desired rudder
        # ----------------------------------------------------------------
        if not emergency_stop and len(ctx.waypoints) > 0:
            rudder_rad = self._los.update(fused_state, ctx.waypoints)
            # Apply COLREGs heading offset (positive = starboard = positive rudder)
            rudder_rad = _clamp(
                rudder_rad + avoid_offset,
                -self._los._config.max_rudder_rad,
                self._los._config.max_rudder_rad,
            )
        else:
            rudder_rad = 0.0

        # ----------------------------------------------------------------
        # 4. Speed / thrust PID
        # ----------------------------------------------------------------
        if emergency_stop:
            thrust_n = -self._preset.max_speed_ms * 500.0  # back off
            thrust_n = _clamp(thrust_n, -8000.0, 0.0)
        else:
            target_speed = self._preset.cruise_speed_ms
            u_err = target_speed - fused_state.u_ms
            u_dot = (fused_state.u_ms - self._u_prev) / max(dt_s, 1e-6)
            thrust_n = _clamp(
                self._KP_SPEED * u_err - self._KD_SPEED * u_dot,
                -8000.0,
                8000.0,
            )

        self._u_prev = fused_state.u_ms
        self._t_s += dt_s

        # ----------------------------------------------------------------
        # 5. Ω (omega) safety multiplier
        # ----------------------------------------------------------------
        risk_score = ctx.risk_score
        visibility_m = ctx.perception.visibility_m
        depth_m = ctx.perception.depth_m
        draft_m = self._preset.draft_m

        omega = _compute_omega(
            risk_score=risk_score,
            fuel_level=self._fuel_level,
            visibility_m=visibility_m,
            depth_m=depth_m,
            draft_m=draft_m,
        )

        # Scale thrust by Ω
        thrust_n *= omega

        # Verdict
        verdict = _omega_to_verdict(omega)

        # ----------------------------------------------------------------
        # 6. Actuator normalisation
        # ----------------------------------------------------------------
        max_thrust = 8000.0
        max_rudder = 0.6109

        reverse = thrust_n < 0.0
        throttle = abs(thrust_n) / max_thrust
        throttle = _clamp(throttle, 0.0, 1.0)
        rudder_norm = _clamp(rudder_rad / max_rudder, -1.0, 1.0)

        actuator = VesselActuator(
            throttle=throttle,
            rudder_norm=rudder_norm,
            reverse=reverse,
        )

        # ----------------------------------------------------------------
        # Build and return updated context
        # ----------------------------------------------------------------
        from .contracts.schemas import MarineTickContext as _Ctx, MarinePerception

        new_ctx = _Ctx(
            state=fused_state,
            perception=ctx.perception,
            actuator=actuator,
            waypoints=ctx.waypoints,
            colregs_state=colregs_state,
            risk_score=risk_score,
            omega=omega,
            verdict=verdict,
            ekf=ekf,
            t_s=self._t_s,
        )
        return new_ctx

    # ------------------------------------------------------------------
    # Fuel level setter (for simulation)
    # ------------------------------------------------------------------

    def set_fuel(self, level: float) -> None:
        """Set normalised fuel level [0, 1]."""
        self._fuel_level = _clamp(level, 0.0, 1.0)


# ---------------------------------------------------------------------------
# Ω helper functions (pure, no side effects)
# ---------------------------------------------------------------------------

def _compute_omega(
    risk_score: float,
    fuel_level: float,
    visibility_m: float,
    depth_m: float,
    draft_m: float,
) -> float:
    """Compute the Ω capability multiplier.

    ω_risk  = 0.55 if risk > 0.85 else 0.78 if risk > 0.35 else 1.0
    ω_fuel  = 0.70 if fuel < 0.20 else 1.0
    ω_vis   = 0.80 if visibility_m < 200 else 1.0
    ω_depth = 0.65 if depth_m < 3·draft_m else 1.0
    Ω       = ω_risk × ω_fuel × ω_vis × ω_depth
    """
    if risk_score > 0.85:
        omega_risk = 0.55
    elif risk_score > 0.35:
        omega_risk = 0.78
    else:
        omega_risk = 1.0

    omega_fuel = 0.70 if fuel_level < 0.2 else 1.0
    omega_vis = 0.80 if visibility_m < 200.0 else 1.0
    omega_depth = 0.65 if depth_m < 3.0 * draft_m else 1.0

    return omega_risk * omega_fuel * omega_vis * omega_depth


def _omega_to_verdict(omega: float) -> str:
    """Map Ω value to a health verdict string."""
    if omega >= 0.90:
        return "HEALTHY"
    elif omega >= 0.60:
        return "DEGRADED"
    elif omega >= 0.30:
        return "CRITICAL"
    else:
        return "EMERGENCY"
