"""
Marine Autonomy Stack
=====================
Autonomous vessel dynamics engine providing:
  - Fossen 3-DOF vessel model (RK4 integration)
  - LOS (Line-of-Sight) guidance with waypoint sequencing
  - COLREGs Rule 8/13/14/15/16 FSM
  - EKF state estimation (GPS + heading + speed)
  - Ω capability multiplier
  - Harbor / Coastal / Ocean / River presets

Design principles (mirroring Autonomy_Runtime_Stack):
  - Zero external dependencies in core (stdlib only)
  - Duck-typing contracts
  - Fail-safe defaults
  - Deterministic ticks
  - Graceful degradation
"""
from .contracts.schemas import (
    VesselState,
    VesselCommand,
    ContactVessel,
    MarinePerception,
    VesselActuator,
    MarineTickContext,
    HullClass,
    DisturbanceState,
    SubmarineState,
)
from .dynamics import (
    VesselParams,
    vessel_step,
    vessel_step_rk4,
    compute_disturbance_forces,
    submarine_depth_step,
    SubmarineFullState,
    surface_vessel_params,
    submarine_params,
    yacht_params,
    boat_params,
    usv_params,
)
from .guidance import LOSGuidance, LOSConfig, DepthChart, maritime_astar
from .estimation import MarineEKF, MarineEKFNoise, StateEstimate
from .colregs import COLREGsBehavior, COLREGsConfig, classify_contact
from .orchestrator import VesselOrchestrator
from .presets import MarinePreset, get_preset, PRESET_REGISTRY, HULL_PRESETS, get_hull_preset

__all__ = [
    # Contracts
    "VesselState",
    "VesselCommand",
    "ContactVessel",
    "MarinePerception",
    "VesselActuator",
    "MarineTickContext",
    "HullClass",
    "DisturbanceState",
    "SubmarineState",
    # Dynamics
    "VesselParams",
    "vessel_step",
    "vessel_step_rk4",
    "compute_disturbance_forces",
    "submarine_depth_step",
    "SubmarineFullState",
    "surface_vessel_params",
    "submarine_params",
    "yacht_params",
    "boat_params",
    "usv_params",
    # Guidance
    "LOSGuidance",
    "LOSConfig",
    "DepthChart",
    "maritime_astar",
    # Estimation
    "MarineEKF",
    "MarineEKFNoise",
    "StateEstimate",
    # COLREGs
    "COLREGsBehavior",
    "COLREGsConfig",
    "classify_contact",
    # Orchestrator
    "VesselOrchestrator",
    # Presets
    "MarinePreset",
    "get_preset",
    "PRESET_REGISTRY",
    "HULL_PRESETS",
    "get_hull_preset",
]

__version__ = "0.2.0"
