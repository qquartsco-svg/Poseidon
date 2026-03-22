# Marine Autonomy Stack

Autonomous vessel runtime for edge AI platforms.
Sibling package to **Autonomy_Runtime_Stack** — same design philosophy,
adapted for 3-DOF marine vessel dynamics.

---

## What it is

A pure-Python package (zero external dependencies) that provides:

- **Fossen 3-DOF vessel model** — simplified linear dynamics with RK4 integration
- **LOS guidance** — Line-of-Sight heading control with waypoint sequencing
- **COLREGs FSM** — heuristic Rules 8/13/14/15/16 collision avoidance
- **Marine EKF** — GPS + heading + speed fusion (4-state, pure Python matrices)
- **Omega (Ω) multiplier** — safety capability scaling based on risk, fuel, visibility, depth
- **Operating presets** — Harbor / Coastal / Ocean / River profiles

> **Disclaimer:** This is not a certified COLREGs compliance system.
> All avoidance logic is internal heuristics intended for research and simulation.
> Any manoeuvre aboard a real vessel requires confirmation by a qualified officer.

---

## Architecture

```
                         ┌─────────────────────────────────┐
External sensors         │       Marine Autonomy Stack      │
                         │                                  │
  AIS / Radar msgs ─────>│  AIS Adapter                    │
  GNSS / IMU ──────────>│    ego_state_from_nmea()         │
                         │    contacts_to_perception()      │
                         │           │                      │
                         │           v                      │
                         │   MarinePerception               │
                         │   VesselState                    │
                         │           │                      │
                         │           v                      │
                         │   VesselOrchestrator.tick()      │
                         │   ┌───────────────────────────┐  │
                         │   │  1. MarineEKF (predict+   │  │
                         │   │     GPS/hdg/speed update) │  │
                         │   │  2. COLREGsBehavior FSM   │  │
                         │   │  3. LOSGuidance → rudder  │  │
                         │   │  4. Speed PID → thrust    │  │
                         │   │  5. Ω safety multiplier   │  │
                         │   │  6. Actuator normalise    │  │
                         │   └───────────────────────────┘  │
                         │           │                      │
                         │           v                      │
                         │   VesselActuator                 │
                         │   (throttle, rudder_norm, rev)   │
                         └─────────────────────────────────┘
                                     │
                         ┌───────────┴───────────┐
                         v                       v
                   Thruster controller    Rudder servo
```

---

## Key equations

### Fossen 3-DOF linear model

```
M · ν̇ = τ - D · ν
η̇   = J(ψ) · ν

η = [x, y, ψ]ᵀ      (earth-fixed position + heading)
ν = [u, v, r]ᵀ      (surge, sway, yaw-rate)

M = diag(m - X_u̇,  m - Y_v̇,  Iz - N_ṙ)   body inertia + added mass
D = diag(-X_u,      -Y_v,      -N_r)         linear damping

J(ψ) = [[cos ψ, -sin ψ, 0],
         [sin ψ,  cos ψ, 0],
         [0,      0,      1]]

τ = [F_thrust,  0,  L_rudder · F_thrust · sin(δ)]ᵀ
```

### LOS guidance

```
α_k   = atan2(y_{k+1} - y_k,  x_{k+1} - x_k)   path angle
e     = -(x - x_k)·sin(α_k) + (y - y_k)·cos(α_k)  cross-track error
ψ_d   = α_k + atan2(-e, Δ)                       desired heading
δ     = clamp(Kp·(ψ_d - ψ) - Kd·r, -δ_max, δ_max)  PD rudder
```

### Ω capability multiplier

```
ω_risk  = 0.55 if risk > 0.85 else 0.78 if risk > 0.35 else 1.0
ω_fuel  = 0.70 if fuel < 0.20 else 1.0
ω_vis   = 0.80 if visibility < 200 m else 1.0
ω_depth = 0.65 if depth < 3·draft else 1.0
Ω       = ω_risk × ω_fuel × ω_vis × ω_depth
```

### COLREGs situation classification

| Geometry | Rule | Action |
|---|---|---|
| Contact bearing < 15°, courses opposing | 14 HEAD_ON | Both alter starboard |
| Contact on own starboard bow (10°–112.5°) | 15 CROSSING | Give-way: alter/slow |
| Contact on own port bow (247.5°–350°) | 15 CROSSING | Stand-on: maintain |
| Contact from astern (112.5°–247.5°) | 13 OVERTAKING | Keep clear |
| Contact < emergency_range_m | 8 SAFETY | Emergency stop |

---

## Presets

| Name | Cruise (kn) | Max (kn) | Lookahead (m) | Safe range (m) | Draft (m) |
|---|---|---|---|---|---|
| harbor | 3 | 5 | 20 | 100 | 1.5 |
| coastal | 10 | 15 | 80 | 500 | 2.0 |
| ocean | 20 | 25 | 200 | 2000 | 4.0 |
| river | 5 | 8 | 40 | 150 | 1.0 |

---

## Quick start

```python
from marine_autonomy import (
    VesselOrchestrator, MarineTickContext,
    VesselState, MarinePerception, get_preset,
)
from marine_autonomy.dynamics import VesselParams, vessel_step_rk4
from marine_autonomy.contracts.schemas import VesselCommand

# Set up
orch   = VesselOrchestrator(preset="coastal")
params = VesselParams()

ctx = MarineTickContext(
    state=VesselState(x_m=0.0, y_m=0.0),
    waypoints=((0.0, 0.0), (500.0, 300.0), (1000.0, 0.0)),
    perception=MarinePerception(depth_m=30.0, visibility_m=10000.0),
)

for step in range(500):
    ctx = orch.tick(ctx, dt_s=0.1)

    # Convert normalised actuator to physical command
    thrust = ctx.actuator.throttle * params.max_thrust_n
    if ctx.actuator.reverse:
        thrust = -thrust
    rudder = ctx.actuator.rudder_norm * params.max_rudder_rad
    cmd = VesselCommand(thrust_n=thrust, rudder_rad=rudder)

    # Integrate physics
    new_state = vessel_step_rk4(ctx.state, cmd, params, dt_s=0.1)
    ctx.state = new_state   # update for next tick

print(f"Final position: ({ctx.state.x_m:.1f}, {ctx.state.y_m:.1f}) m")
print(f"Omega: {ctx.omega:.3f}  |  Verdict: {ctx.verdict}")
```

CLI example:

```bash
python examples/run_harbor.py --preset harbor --steps 200 --waypoints "0,0 100,50 200,0"
python examples/run_harbor.py --preset coastal --steps 500 --waypoints "0,0 500,300 1000,0"
```

---

## Extending — adding a new vessel type

1. Subclass or instantiate `VesselParams` with your vessel's mass, damping, and actuation geometry:

```python
from marine_autonomy.dynamics import VesselParams

my_ferry = VesselParams(
    mass_kg=80_000,
    Iz_kgm2=500_000,
    X_udot=-8_000,  Y_vdot=-15_000, N_rdot=-4_000,
    Xu=-800,        Yv=-2_000,      Nr=-1_500,
    L_rudder_m=4.0,
    max_thrust_n=200_000,
    max_rudder_rad=0.524,  # 30°
)
```

2. Optionally define a custom `MarinePreset` and register it:

```python
from marine_autonomy.presets import MarinePreset, PRESET_REGISTRY

PRESET_REGISTRY["ferry"] = MarinePreset(
    name="ferry",
    max_speed_ms=10.3, cruise_speed_ms=8.2,
    lookahead_m=150.0, acceptance_radius_m=60.0,
    safe_range_m=800.0, action_range_m=2000.0, emergency_range_m=80.0,
    draft_m=5.0, description="RoPax ferry",
)
```

3. Pass the preset name to `VesselOrchestrator(preset="ferry")`.
   The orchestrator, guidance, and COLREGs modules require no further changes.

---

## Relationship to Autonomy_Runtime_Stack

`Marine_Autonomy_Stack` is a sibling package that applies the same
architecture to marine surface vessels:

| Aspect | Autonomy_Runtime_Stack | Marine_Autonomy_Stack |
|---|---|---|
| Vehicle domain | Ground / aerial | Marine surface |
| Dynamics | Kinematic bicycle / IMU | Fossen 3-DOF |
| Guidance | Pure-pursuit / path | LOS waypoint |
| Collision avoidance | TTC-based FSM | COLREGs FSM |
| Estimation | EKF (position + heading) | EKF (x, y, ψ, u) |
| Orchestrator | AutonomyOrchestrator | VesselOrchestrator |
| Dependencies | stdlib only | stdlib only |

Both packages can be used independently or composed — e.g. a USV (unmanned
surface vessel) operating as part of a larger autonomous system could have the
`VesselOrchestrator` report state to an outer mission manager built on
`Autonomy_Runtime_Stack`.

---

## Running tests

```bash
# From the package root:
python -m pytest tests/ -v

# Or without pytest:
python tests/test_marine.py
```

All 22 tests pass with Python 3.8+ and no external dependencies.
