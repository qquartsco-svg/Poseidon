"""
Fossen 3-DOF simplified vessel dynamics.

Equations of motion (linear Fossen model):

  M · ν̇ = τ - D · ν
  η̇   = J(ψ) · ν

where:
  η = [x, y, ψ]ᵀ           — earth-fixed position / heading
  ν = [u, v, r]ᵀ           — body-fixed velocities (surge, sway, yaw-rate)

  M = diag(m - X_u̇,  m - Y_v̇,  Iz - N_ṙ)   — inertia + added mass (diagonal approx)
  D = diag(-X_u,      -Y_v,      -N_r)         — linear damping (negative by convention)

  J(ψ) = [[cos ψ, -sin ψ, 0],
           [sin ψ,  cos ψ, 0],
           [0,      0,      1]]

  τ = [thrust_n,  0,  L_rudder_m · thrust_n · sin(rudder_rad)]ᵀ

Integration:
  Euler  → vessel_step()
  RK4    → vessel_step_rk4()   (preferred for production use)

Reference:
  Fossen, T.I. (2011) "Handbook of Marine Craft Hydrodynamics and Motion Control"
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict

from .contracts.schemas import VesselState, VesselCommand


# ---------------------------------------------------------------------------
# Vessel parameters
# ---------------------------------------------------------------------------

@dataclass
class VesselParams:
    """Hydrodynamic parameters for a 10 m coastal patrol vessel.

    All values approximate — tune with dedicated sea-trials or CFD data.

    mass_kg     — Vessel displacement mass (kg)
    Iz_kgm2     — Moment of inertia about z-axis (kg·m²)
    X_udot      — Added mass surge (kg)   — negative by convention
    Y_vdot      — Added mass sway  (kg)   — negative by convention
    N_rdot      — Added mass yaw   (kg·m²) — negative by convention
    Xu          — Linear surge damping coefficient (N·s/m)  — negative
    Yv          — Linear sway  damping coefficient (N·s/m)  — negative
    Nr          — Linear yaw   damping coefficient (N·m·s)  — negative
    L_rudder_m  — Rudder moment arm (m)
    max_thrust_n   — Maximum forward thrust (N)
    max_rudder_rad — Maximum rudder deflection (rad)
    """

    mass_kg: float = 5000.0
    Iz_kgm2: float = 12000.0

    # Added-mass terms (negative convention → subtracted from inertia)
    X_udot: float = -500.0
    Y_vdot: float = -800.0
    N_rdot: float = -300.0

    # Linear damping (negative convention → damping force opposes motion)
    Xu: float = -100.0
    Yv: float = -200.0
    Nr: float = -150.0

    # Actuation geometry
    L_rudder_m: float = 1.5

    # Saturation limits
    max_thrust_n: float = 8000.0
    max_rudder_rad: float = 0.6109  # ≈ 35 °


# ---------------------------------------------------------------------------
# Internal helper
# ---------------------------------------------------------------------------

def _clamp(v: float, lo: float, hi: float) -> float:
    """Clamp v to [lo, hi]."""
    return max(lo, min(hi, v))


def _vessel_derivatives(
    state: VesselState,
    cmd: VesselCommand,
    params: VesselParams,
) -> Dict[str, float]:
    """Compute η̇ and ν̇ at the given state/command.

    Returns a dict with keys matching VesselState fields
    (x_m, y_m, psi_rad, u_ms, v_ms, r_rads).
    t_s derivative is always 1.0 (wall-clock rate).

    Governing equations:
      ν̇ = M⁻¹ · (τ - D · ν)
      η̇ = J(ψ) · ν

    Because M and D are diagonal the inversion is trivial:
      u̇ = (τ_u - (-Xu)·u)  / (mass - X_udot)
      v̇ = (     - (-Yv)·v)  / (mass - Y_vdot)   [no direct sway force]
      ṙ = (τ_r - (-Nr)·r)  / (Iz   - N_rdot)
    """
    psi = state.psi_rad
    u = state.u_ms
    v = state.v_ms
    r = state.r_rads

    # --- Saturate actuator inputs ---
    thrust = _clamp(cmd.thrust_n, -params.max_thrust_n, params.max_thrust_n)
    rudder = _clamp(cmd.rudder_rad, -params.max_rudder_rad, params.max_rudder_rad)

    # --- Generalised forces τ ---
    tau_u = thrust
    tau_v = 0.0  # no direct sway thruster
    tau_r = params.L_rudder_m * thrust * math.sin(rudder)

    # --- Effective inertia (M diagonal) ---
    m_u = params.mass_kg - params.X_udot   # > 0
    m_v = params.mass_kg - params.Y_vdot   # > 0
    m_r = params.Iz_kgm2 - params.N_rdot   # > 0

    # --- Body-frame accelerations  ν̇ = M⁻¹(τ − D·ν) ---
    # D diagonal: D_uu = -Xu > 0, etc.
    u_dot = (tau_u - (-params.Xu) * u) / m_u
    v_dot = (tau_v - (-params.Yv) * v) / m_v
    r_dot = (tau_r - (-params.Nr) * r) / m_r

    # --- Earth-frame kinematics  η̇ = J(ψ)·ν ---
    x_dot = math.cos(psi) * u - math.sin(psi) * v
    y_dot = math.sin(psi) * u + math.cos(psi) * v
    psi_dot = r

    return {
        "x_m":    x_dot,
        "y_m":    y_dot,
        "psi_rad": psi_dot,
        "u_ms":   u_dot,
        "v_ms":   v_dot,
        "r_rads": r_dot,
    }


# ---------------------------------------------------------------------------
# Public integrators
# ---------------------------------------------------------------------------

def vessel_step(
    state: VesselState,
    cmd: VesselCommand,
    params: VesselParams,
    dt_s: float,
) -> VesselState:
    """Euler integration of Fossen 3-DOF model.

    Adequate for small dt_s; prefer vessel_step_rk4 for production.

    Returns a new VesselState at t + dt_s.
    """
    d = _vessel_derivatives(state, cmd, params)
    return VesselState(
        x_m=state.x_m + d["x_m"] * dt_s,
        y_m=state.y_m + d["y_m"] * dt_s,
        psi_rad=state.psi_rad + d["psi_rad"] * dt_s,
        u_ms=state.u_ms + d["u_ms"] * dt_s,
        v_ms=state.v_ms + d["v_ms"] * dt_s,
        r_rads=state.r_rads + d["r_rads"] * dt_s,
        t_s=state.t_s + dt_s,
    )


def vessel_step_rk4(
    state: VesselState,
    cmd: VesselCommand,
    params: VesselParams,
    dt_s: float,
) -> VesselState:
    """4th-order Runge-Kutta integration of Fossen 3-DOF model.

    RK4 scheme:
      k1 = f(s)
      k2 = f(s + dt/2 · k1)
      k3 = f(s + dt/2 · k2)
      k4 = f(s + dt   · k3)
      s_new = s + dt/6 · (k1 + 2·k2 + 2·k3 + k4)

    Returns a new VesselState at t + dt_s.
    """
    def _make_state(base: VesselState, d: Dict[str, float], scale: float) -> VesselState:
        return VesselState(
            x_m=base.x_m + d["x_m"] * scale,
            y_m=base.y_m + d["y_m"] * scale,
            psi_rad=base.psi_rad + d["psi_rad"] * scale,
            u_ms=base.u_ms + d["u_ms"] * scale,
            v_ms=base.v_ms + d["v_ms"] * scale,
            r_rads=base.r_rads + d["r_rads"] * scale,
            t_s=base.t_s + scale,
        )

    k1 = _vessel_derivatives(state, cmd, params)
    s2 = _make_state(state, k1, dt_s / 2)
    k2 = _vessel_derivatives(s2, cmd, params)
    s3 = _make_state(state, k2, dt_s / 2)
    k3 = _vessel_derivatives(s3, cmd, params)
    s4 = _make_state(state, k3, dt_s)
    k4 = _vessel_derivatives(s4, cmd, params)

    def _combine(key: str) -> float:
        return (k1[key] + 2 * k2[key] + 2 * k3[key] + k4[key]) / 6.0

    return VesselState(
        x_m=state.x_m + _combine("x_m") * dt_s,
        y_m=state.y_m + _combine("y_m") * dt_s,
        psi_rad=state.psi_rad + _combine("psi_rad") * dt_s,
        u_ms=state.u_ms + _combine("u_ms") * dt_s,
        v_ms=state.v_ms + _combine("v_ms") * dt_s,
        r_rads=state.r_rads + _combine("r_rads") * dt_s,
        t_s=state.t_s + dt_s,
    )
