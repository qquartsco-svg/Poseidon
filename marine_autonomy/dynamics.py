"""
Fossen 3-DOF nonlinear vessel dynamics + Coriolis + disturbance forces.

Equations of motion (nonlinear Fossen model):

  M · ν̇ = τ_ctrl + τ_dist - C(ν) · ν - D(ν) · ν
  η̇   = J(ψ) · ν

where:
  η = [x, y, ψ]ᵀ           — earth-fixed position / heading
  ν = [u, v, r]ᵀ           — body-fixed velocities (surge, sway, yaw-rate)

  M = diag(m - X_u̇,  m - Y_v̇,  Iz - N_ṙ)   — inertia + added mass (diagonal approx)

  Coriolis matrix (3-DOF):
    m11 = mass - X_udot
    m22 = mass - Y_vdot
    m26 = -Y_rdot

    C(ν) = [[0,          0,          -m22*v - m26*r],
             [0,          0,           m11*u        ],
             [m22*v+m26*r, -m11*u,     0            ]]

  Nonlinear damping:
    D(ν)·ν = [Xu*u + Xuu*|u|*u,
              Yv*v + Yvv*|v|*v,
              Nr*r + Nrr*|r|*r]

  J(ψ) = [[cos ψ, -sin ψ, 0],
           [sin ψ,  cos ψ, 0],
           [0,      0,      1]]

Integration:
  Euler  → vessel_step()
  RK4    → vessel_step_rk4()   (preferred for production use)

Reference:
  Fossen, T.I. (2011) "Handbook of Marine Craft Hydrodynamics and Motion Control"
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Optional

from .contracts.schemas import VesselState, VesselCommand, DisturbanceState, SubmarineState


# ---------------------------------------------------------------------------
# Vessel parameters — hull-class aware
# ---------------------------------------------------------------------------

@dataclass
class VesselParams:
    """Hydrodynamic parameters covering all hull classes.

    Identity
    --------
    hull_class  — "surface_vessel" | "submarine" | "yacht" | "boat" | "autonomous_usv"
    name        — human-readable label

    Mass/inertia
    ------------
    mass_kg     — Vessel displacement mass (kg)
    Iz_kgm2     — Moment of inertia about z-axis (kg·m²)

    Added mass (diagonal Mα approximation)
    ---------------------------------------
    X_udot      — surge added mass (kg)  — negative convention
    Y_vdot      — sway added mass (kg)   — negative convention
    N_rdot      — yaw added inertia (kg·m²) — negative convention
    Y_rdot      — sway-yaw coupling (Coriolis coupling)

    Linear damping  (negative convention → force opposes motion)
    --------------------------------------------------------------
    Xu, Yv, Nr

    Nonlinear damping coefficients (|u|·u form)
    --------------------------------------------
    Xuu, Yvv, Nrr

    Actuator limits
    ---------------
    max_thrust_n, max_rudder_rad, L_rudder_m

    Geometry
    --------
    Lpp_m       — length between perpendiculars (m)
    Beam_m      — beam (m)
    Draft_m     — design draft (m)

    Disturbance coefficients
    ------------------------
    Cx_wind, Cy_wind, Cn_wind

    Submarine only
    --------------
    depth_max_m — maximum safe dive depth (m); 0 = surface only
    Kz          — heave damping coefficient
    Bz          — depth control gain
    """

    # Identity
    hull_class: str = "surface_vessel"
    name: str = "default"

    # Mass/inertia
    mass_kg: float    = 5000.0
    Iz_kgm2: float    = 12000.0

    # Added mass (diagonal Mα approximation)
    X_udot: float = -500.0   # surge added mass
    Y_vdot: float = -800.0   # sway added mass
    N_rdot: float = -300.0   # yaw added inertia
    Y_rdot: float = -100.0   # sway-yaw coupling (Coriolis)

    # Linear damping
    Xu: float = -100.0
    Yv: float = -200.0
    Nr: float = -150.0

    # Nonlinear damping coefficients (|u|·u form)
    Xuu: float = -50.0
    Yvv: float = -100.0
    Nrr: float = -80.0

    # Actuator limits
    max_thrust_n: float    = 8000.0
    max_rudder_rad: float  = 0.6109
    L_rudder_m: float      = 1.5

    # Geometry
    Lpp_m: float   = 10.0    # length between perpendiculars
    Beam_m: float  = 3.0     # beam
    Draft_m: float = 1.5     # draft

    # Disturbance coefficients
    Cx_wind: float = 0.6     # wind force coeff (longitudinal)
    Cy_wind: float = 0.8     # wind force coeff (lateral)
    Cn_wind: float = 0.1     # wind moment coeff

    # Submarine only
    depth_max_m: float = 0.0    # 0 = surface only
    Kz: float = 0.0             # heave damping
    Bz: float = 0.0             # depth control gain


# ---------------------------------------------------------------------------
# Hull-class factory functions
# ---------------------------------------------------------------------------

def surface_vessel_params(**kw) -> VesselParams:
    """10m coastal patrol vessel, ~5000kg."""
    base = dict(
        hull_class="surface_vessel",
        name="surface_vessel",
        mass_kg=5000.0, Iz_kgm2=12000.0,
        X_udot=-500.0, Y_vdot=-800.0, N_rdot=-300.0, Y_rdot=-100.0,
        Xu=-100.0, Yv=-200.0, Nr=-150.0,
        Xuu=-50.0, Yvv=-100.0, Nrr=-80.0,
        max_thrust_n=8000.0, max_rudder_rad=0.6109, L_rudder_m=1.5,
        Lpp_m=10.0, Beam_m=3.0, Draft_m=1.5,
        Cx_wind=0.6, Cy_wind=0.8, Cn_wind=0.1,
        depth_max_m=0.0, Kz=0.0, Bz=0.0,
    )
    base.update(kw)
    return VesselParams(**base)


def submarine_params(**kw) -> VesselParams:
    """50m submarine, ~200000kg, depth_max=300m."""
    base = dict(
        hull_class="submarine",
        name="submarine",
        mass_kg=200000.0, Iz_kgm2=5000000.0,
        X_udot=-20000.0, Y_vdot=-40000.0, N_rdot=-8000000.0, Y_rdot=-5000.0,
        Xu=-3000.0, Yv=-5000.0, Nr=-2000000.0,
        Xuu=-800.0, Yvv=-2000.0, Nrr=-500000.0,
        max_thrust_n=500000.0, max_rudder_rad=0.5236, L_rudder_m=5.0,
        Lpp_m=50.0, Beam_m=8.0, Draft_m=8.0,
        Cx_wind=0.0, Cy_wind=0.0, Cn_wind=0.0,  # submerged: no wind
        depth_max_m=300.0, Kz=5000.0, Bz=8000.0,
    )
    base.update(kw)
    return VesselParams(**base)


def yacht_params(**kw) -> VesselParams:
    """12m sailing yacht, ~3500kg (lighter than 5000kg patrol vessel)."""
    base = dict(
        hull_class="yacht",
        name="yacht",
        mass_kg=3500.0, Iz_kgm2=8000.0,
        X_udot=-600.0, Y_vdot=-1200.0, N_rdot=-500.0, Y_rdot=-150.0,
        Xu=-80.0, Yv=-180.0, Nr=-120.0,
        Xuu=-30.0, Yvv=-80.0, Nrr=-60.0,
        max_thrust_n=5000.0, max_rudder_rad=0.6109, L_rudder_m=1.8,
        Lpp_m=12.0, Beam_m=3.5, Draft_m=1.8,
        Cx_wind=0.8, Cy_wind=1.2, Cn_wind=0.15,
        depth_max_m=0.0, Kz=0.0, Bz=0.0,
    )
    base.update(kw)
    return VesselParams(**base)


def boat_params(**kw) -> VesselParams:
    """6m speedboat, ~800kg, high Xuu nonlinear drag."""
    base = dict(
        hull_class="boat",
        name="boat",
        mass_kg=800.0, Iz_kgm2=800.0,
        X_udot=-60.0, Y_vdot=-120.0, N_rdot=-50.0, Y_rdot=-20.0,
        Xu=-40.0, Yv=-80.0, Nr=-50.0,
        Xuu=-120.0, Yvv=-200.0, Nrr=-100.0,  # higher nonlinear drag
        max_thrust_n=15000.0, max_rudder_rad=0.6109, L_rudder_m=0.8,
        Lpp_m=6.0, Beam_m=2.2, Draft_m=0.5,
        Cx_wind=0.5, Cy_wind=0.7, Cn_wind=0.08,
        depth_max_m=0.0, Kz=0.0, Bz=0.0,
    )
    base.update(kw)
    return VesselParams(**base)


def usv_params(**kw) -> VesselParams:
    """3m autonomous USV, ~150kg, small form factor."""
    base = dict(
        hull_class="autonomous_usv",
        name="autonomous_usv",
        mass_kg=150.0, Iz_kgm2=60.0,
        X_udot=-15.0, Y_vdot=-30.0, N_rdot=-10.0, Y_rdot=-5.0,
        Xu=-15.0, Yv=-30.0, Nr=-20.0,
        Xuu=-25.0, Yvv=-50.0, Nrr=-25.0,
        max_thrust_n=500.0, max_rudder_rad=0.6109, L_rudder_m=0.4,
        Lpp_m=3.0, Beam_m=1.2, Draft_m=0.3,
        Cx_wind=0.4, Cy_wind=0.6, Cn_wind=0.07,
        depth_max_m=0.0, Kz=0.0, Bz=0.0,
    )
    base.update(kw)
    return VesselParams(**base)


# ---------------------------------------------------------------------------
# Submarine 4th DOF (depth)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class SubmarineFullState(VesselState):
    """VesselState extended with depth and heave velocity."""
    depth_m: float = 0.0
    w_ms: float = 0.0


def submarine_depth_step(
    state,
    target_depth_m: float,
    params: VesselParams,
    dt_s: float,
) -> tuple:
    """Simple PD depth controller (4th DOF, heave axis).

    e_z    = target_depth - depth
    w_dot  = Bz * e_z - Kz * w
    depth_new = depth + w * dt

    Returns (new_depth_m, new_w_ms).
    """
    depth = getattr(state, "depth_m", 0.0)
    w = getattr(state, "w_ms", 0.0)

    e_z = target_depth_m - depth
    w_dot = params.Bz * e_z - params.Kz * w
    new_w = w + w_dot * dt_s
    new_depth = depth + new_w * dt_s

    # Clamp to valid range
    if params.depth_max_m > 0.0:
        new_depth = max(0.0, min(params.depth_max_m, new_depth))

    return (new_depth, new_w)


# ---------------------------------------------------------------------------
# Disturbance forces
# ---------------------------------------------------------------------------

def compute_disturbance_forces(
    state: VesselState,
    params: VesselParams,
    dist: DisturbanceState,
) -> tuple:
    """Compute combined environmental disturbance forces in body frame.

    Wave disturbance (1st-order sinusoidal approximation):
      F_wave = amplitude * sin(2π*t/Tp) modulated by relative wave direction

    Wind disturbance:
      q = 0.5 * ρ_air * Vw²    (dynamic pressure)
      Fx_wind =  q * Cx * A_lat * cos(ψ_wind - ψ)
      Fy_wind = -q * Cy * A_lat * sin(ψ_wind - ψ)
      Mz_wind = -q * Cn * A_lat * Lpp * sin(ψ_wind - ψ)

    Ocean current: modifies relative velocity for damping (handled in
    _vessel_derivatives_nonlinear via u_r, v_r).

    Returns: (tau_X, tau_Y, tau_N) in body frame (N, N, N·m).
    """
    rho_water = 1025.0  # kg/m³
    rho_air   = 1.225   # kg/m³

    # --- Wave ---
    Hs = dist.wave_height_m
    Tp = max(dist.wave_period_s, 0.1)
    Awp = params.Lpp_m * params.Beam_m  # waterplane area approx
    wave_angle = dist.wave_dir_rad - state.psi_rad
    wave_phase = 2.0 * math.pi * dist.t_s / Tp
    wave_amp = 0.05 * rho_water * 9.81 * Hs ** 2 * Awp / params.Lpp_m
    tau_wave_X = wave_amp * math.cos(wave_angle) * math.sin(wave_phase)
    tau_wave_Y = wave_amp * math.sin(wave_angle) * math.sin(wave_phase)
    tau_wave_N = 0.1 * tau_wave_Y * params.Lpp_m

    # --- Wind ---
    Vw = dist.wind_speed_ms
    q  = 0.5 * rho_air * Vw ** 2
    A_lat = params.Lpp_m * (params.Draft_m + 1.0)  # lateral area approx
    wind_angle = dist.wind_dir_rad - state.psi_rad
    tau_wind_X =  q * params.Cx_wind * A_lat * math.cos(wind_angle)
    tau_wind_Y = -q * params.Cy_wind * A_lat * math.sin(wind_angle)
    tau_wind_N = -q * params.Cn_wind * A_lat * params.Lpp_m * math.sin(wind_angle)

    return (
        tau_wave_X + tau_wind_X,
        tau_wave_Y + tau_wind_Y,
        tau_wave_N + tau_wind_N,
    )


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _clamp(v: float, lo: float, hi: float) -> float:
    """Clamp v to [lo, hi]."""
    return max(lo, min(hi, v))


def _vessel_derivatives_nonlinear(
    state: VesselState,
    cmd: VesselCommand,
    params: VesselParams,
    disturbance: Optional[DisturbanceState] = None,
) -> Dict[str, float]:
    """Compute η̇ and ν̇ using nonlinear Fossen model with Coriolis and disturbance.

    M·ν̇ = τ_ctrl + τ_dist - C(ν)·ν - D(ν)·ν
    η̇   = J(ψ)·ν

    Coriolis matrix (3-DOF, Fossen 2011, Eq. 6.32):
      m11 = mass - X_udot
      m22 = mass - Y_vdot
      m26 = -Y_rdot

      C(ν) = [[0,          0,          -m22*v - m26*r],
               [0,          0,           m11*u        ],
               [m22*v+m26*r, -m11*u,     0            ]]

    Nonlinear damping:
      D(ν)·ν = [Xu*u + Xuu*|u|*u,
                Yv*v + Yvv*|v|*v,
                Nr*r + Nrr*|r|*r]
    """
    psi = state.psi_rad
    u = state.u_ms
    v = state.v_ms
    r = state.r_rads

    # --- Saturate actuator inputs ---
    thrust = _clamp(cmd.thrust_n, -params.max_thrust_n, params.max_thrust_n)
    rudder = _clamp(cmd.rudder_rad, -params.max_rudder_rad, params.max_rudder_rad)

    # --- Control forces τ ---
    tau_u = thrust
    tau_v = 0.0  # no direct sway thruster
    tau_r = params.L_rudder_m * thrust * math.sin(rudder)

    # --- Effective inertia (M diagonal) ---
    m_u = params.mass_kg - params.X_udot   # > 0
    m_v = params.mass_kg - params.Y_vdot   # > 0
    m_r = params.Iz_kgm2 - params.N_rdot   # > 0

    # Coriolis-centripetal matrix C(ν) (3-DOF, Fossen 2011 Eq. 6.32)
    # M = diag(m11, m22, m33) with m26 off-diagonal coupling
    m11 = params.mass_kg - params.X_udot
    m22 = params.mass_kg - params.Y_vdot
    m26 = -params.Y_rdot

    # C(ν)·ν gives the Coriolis/centripetal forces
    #   [C]·ν = [[0, 0, -(m22*v + m26*r)],
    #            [0, 0,   m11*u          ],
    #            [m22*v+m26*r, -m11*u, 0 ]] · [u, v, r]ᵀ
    cor_u = -(m22 * v + m26 * r) * r
    cor_v =   m11 * u * r
    cor_r =   (m22 * v + m26 * r) * u - m11 * u * v

    # --- Nonlinear damping D(ν)·ν ---
    d_u = params.Xu * u + params.Xuu * abs(u) * u
    d_v = params.Yv * v + params.Yvv * abs(v) * v
    d_r = params.Nr * r + params.Nrr * abs(r) * r

    # --- Disturbance forces ---
    if disturbance is not None:
        dist_X, dist_Y, dist_N = compute_disturbance_forces(state, params, disturbance)
    else:
        dist_X, dist_Y, dist_N = 0.0, 0.0, 0.0

    # --- Ocean current: adjust relative velocity for damping ---
    if disturbance is not None and (disturbance.current_u_ms != 0.0 or disturbance.current_v_ms != 0.0):
        cu = disturbance.current_u_ms
        cv = disturbance.current_v_ms
        # Transform current to body frame
        u_c =  cu * math.cos(psi) + cv * math.sin(psi)
        v_c = -cu * math.sin(psi) + cv * math.cos(psi)
        u_r = u - u_c
        v_r = v - v_c
        # Recompute damping with relative velocity (signs still negative/opposing)
        d_u = params.Xu * u_r + params.Xuu * abs(u_r) * u_r
        d_v = params.Yv * v_r + params.Yvv * abs(v_r) * v_r

    # --- Body-frame accelerations  ν̇ = M⁻¹(τ_ctrl + τ_dist + d·ν - C·ν) ---
    # Note: d_u, d_v, d_r already carry the correct sign (negative, opposing motion)
    # because Xu, Xuu, Yv, Yvv, Nr, Nrr are all defined negative.
    # cor_u, cor_v, cor_r are the Coriolis/centripetal forces in body frame.
    u_dot = (tau_u + dist_X + d_u - cor_u) / m_u
    v_dot = (tau_v + dist_Y + d_v - cor_v) / m_v
    r_dot = (tau_r + dist_N + d_r - cor_r) / m_r

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


def _vessel_derivatives(
    state: VesselState,
    cmd: VesselCommand,
    params: VesselParams,
    disturbance: Optional[DisturbanceState] = None,
) -> Dict[str, float]:
    """Compatibility alias — delegates to nonlinear model."""
    return _vessel_derivatives_nonlinear(state, cmd, params, disturbance)


# ---------------------------------------------------------------------------
# Public integrators
# ---------------------------------------------------------------------------

def vessel_step(
    state: VesselState,
    cmd: VesselCommand,
    params: VesselParams,
    dt_s: float,
    disturbance: Optional[DisturbanceState] = None,
) -> VesselState:
    """Euler integration of Fossen 3-DOF nonlinear model.

    Adequate for small dt_s; prefer vessel_step_rk4 for production.

    Returns a new VesselState at t + dt_s.
    """
    d = _vessel_derivatives_nonlinear(state, cmd, params, disturbance)
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
    disturbance: Optional[DisturbanceState] = None,
) -> VesselState:
    """4th-order Runge-Kutta integration of Fossen 3-DOF nonlinear model.

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

    k1 = _vessel_derivatives_nonlinear(state, cmd, params, disturbance)
    s2 = _make_state(state, k1, dt_s / 2)
    k2 = _vessel_derivatives_nonlinear(s2, cmd, params, disturbance)
    s3 = _make_state(state, k2, dt_s / 2)
    k3 = _vessel_derivatives_nonlinear(s3, cmd, params, disturbance)
    s4 = _make_state(state, k3, dt_s)
    k4 = _vessel_derivatives_nonlinear(s4, cmd, params, disturbance)

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
