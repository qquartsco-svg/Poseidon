"""
Marine Autonomy Stack — CLI dry-run example.

Runs a closed-loop simulation using the VesselOrchestrator + Fossen dynamics.
The dynamics are stepped with vessel_step_rk4 inside the loop while the
orchestrator provides guidance commands.

Usage:
  python examples/run_harbor.py
  python examples/run_harbor.py --preset coastal --steps 300 --waypoints "0,0 200,100 400,0"
  python examples/run_harbor.py --preset ocean   --steps 100 --dt 0.5

Options:
  --preset    harbor | coastal | ocean | river  (default: harbor)
  --steps     number of simulation ticks        (default: 200)
  --waypoints space-separated "x,y" pairs       (default: "0,0 100,50 200,0")
  --dt        timestep in seconds               (default: 0.1)
  --no-ekf    disable EKF (use raw state only)
"""
from __future__ import annotations

import argparse
import math
import os
import sys

# Allow running from the repo root without installing the package
_ROOT = os.path.join(os.path.dirname(__file__), "..")
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from marine_autonomy.contracts.schemas import MarineTickContext, MarinePerception
from marine_autonomy.dynamics import VesselParams, vessel_step_rk4
from marine_autonomy.contracts.schemas import VesselState, VesselCommand
from marine_autonomy.orchestrator import VesselOrchestrator
from marine_autonomy.presets import get_preset


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def _parse_waypoints(s: str):
    """Parse "x1,y1 x2,y2 ..." into a tuple of (float, float) pairs."""
    wps = []
    for token in s.strip().split():
        parts = token.split(",")
        if len(parts) != 2:
            raise argparse.ArgumentTypeError(f"Bad waypoint: {token!r} — expected x,y")
        wps.append((float(parts[0]), float(parts[1])))
    return tuple(wps)


def _parse_args():
    p = argparse.ArgumentParser(description="Marine Autonomy Stack — dry-run simulation")
    p.add_argument("--preset",     default="harbor", help="Operating preset")
    p.add_argument("--steps",      type=int,   default=200, help="Number of ticks")
    p.add_argument("--waypoints",  default="0,0 100,50 200,0", help="Waypoint list")
    p.add_argument("--dt",         type=float, default=0.1, help="Timestep (s)")
    p.add_argument("--no-ekf",     action="store_true", help="Disable EKF")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Table header / row helpers
# ---------------------------------------------------------------------------

_HDR = (
    f"{'t_s':>6}  {'x_m':>8}  {'y_m':>8}  "
    f"{'hdg°':>6}  {'spd_kn':>7}  {'rud°':>6}  "
    f"{'colregs':<16}  {'Ω':>5}  {'verdict':<10}"
)
_SEP = "-" * len(_HDR)


def _row(
    t_s: float,
    x_m: float,
    y_m: float,
    psi_rad: float,
    u_ms: float,
    rudder_rad: float,
    colregs: str,
    omega: float,
    verdict: float,
) -> str:
    hdg_deg = math.degrees(psi_rad) % 360.0
    spd_kn = u_ms / 0.5144
    rud_deg = math.degrees(rudder_rad)
    return (
        f"{t_s:6.1f}  {x_m:8.1f}  {y_m:8.1f}  "
        f"{hdg_deg:6.1f}  {spd_kn:7.2f}  {rud_deg:6.1f}  "
        f"{colregs:<16}  {omega:5.3f}  {verdict:<10}"
    )


# ---------------------------------------------------------------------------
# Main simulation loop
# ---------------------------------------------------------------------------

def main():
    args = _parse_args()

    preset = get_preset(args.preset)
    waypoints = _parse_waypoints(args.waypoints)
    params = VesselParams()
    use_ekf = not args.no_ekf

    orch = VesselOrchestrator(preset=args.preset, use_ekf=use_ekf)

    # Initial state: at first waypoint with zero velocity
    x0, y0 = waypoints[0] if waypoints else (0.0, 0.0)
    state = VesselState(x_m=x0, y_m=y0)

    ctx = MarineTickContext(
        state=state,
        waypoints=waypoints,
        perception=MarinePerception(depth_m=20.0, visibility_m=5000.0),
    )

    print()
    print(f"  Marine Autonomy Stack — {args.preset.upper()} preset")
    print(f"  Waypoints : {waypoints}")
    print(f"  Steps     : {args.steps}  |  dt = {args.dt} s  |  EKF: {use_ekf}")
    print(f"  Cruise    : {preset.speed_kn():.1f} kn  |  Max: {preset.max_speed_kn():.1f} kn")
    print()
    print(_HDR)
    print(_SEP)

    for step in range(args.steps):
        # 1. Orchestrator: compute actuator from current state
        ctx = orch.tick(ctx, dt_s=args.dt)

        # 2. Convert normalised actuator → physical command
        max_thrust = params.max_thrust_n
        max_rudder = params.max_rudder_rad
        thrust = ctx.actuator.throttle * max_thrust
        if ctx.actuator.reverse:
            thrust = -thrust
        rudder = ctx.actuator.rudder_norm * max_rudder

        cmd = VesselCommand(thrust_n=thrust, rudder_rad=rudder)

        # 3. Integrate physics (RK4)
        raw_state = ctx.state
        if not isinstance(raw_state, VesselState):
            raw_state = VesselState(
                x_m=raw_state.x_m,
                y_m=raw_state.y_m,
                psi_rad=raw_state.psi_rad,
                u_ms=raw_state.u_ms,
            )
        new_phys = vessel_step_rk4(raw_state, cmd, params, args.dt)

        # 4. Feed updated physics state back into context
        from marine_autonomy.contracts.schemas import MarineTickContext as _MTC
        ctx = _MTC(
            state=new_phys,
            perception=ctx.perception,
            actuator=ctx.actuator,
            waypoints=ctx.waypoints,
            colregs_state=ctx.colregs_state,
            risk_score=ctx.risk_score,
            omega=ctx.omega,
            verdict=ctx.verdict,
            ekf=ctx.ekf,
            t_s=ctx.t_s,
        )

        # 5. Print every 10 steps (or every step if ≤ 20 total)
        if step % max(1, args.steps // 20) == 0 or step == args.steps - 1:
            print(_row(
                t_s=new_phys.t_s,
                x_m=new_phys.x_m,
                y_m=new_phys.y_m,
                psi_rad=new_phys.psi_rad,
                u_ms=new_phys.u_ms,
                rudder_rad=rudder,
                colregs=ctx.colregs_state,
                omega=ctx.omega,
                verdict=ctx.verdict,
            ))

        # 6. Stop early if guidance is complete
        if orch._los.is_complete(ctx.waypoints):
            print(_SEP)
            print(f"  All waypoints reached at t = {new_phys.t_s:.1f} s")
            break
    else:
        print(_SEP)
        print(f"  Simulation complete ({args.steps} steps).")

    print()


if __name__ == "__main__":
    main()
