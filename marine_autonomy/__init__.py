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
)
from .dynamics import VesselParams, vessel_step, vessel_step_rk4
from .guidance import LOSGuidance, LOSConfig
from .estimation import MarineEKF, MarineEKFNoise, StateEstimate
from .colregs import COLREGsBehavior, COLREGsConfig, classify_contact
from .orchestrator import VesselOrchestrator
from .presets import MarinePreset, get_preset, PRESET_REGISTRY

__all__ = [
    # Contracts
    "VesselState",
    "VesselCommand",
    "ContactVessel",
    "MarinePerception",
    "VesselActuator",
    "MarineTickContext",
    # Dynamics
    "VesselParams",
    "vessel_step",
    "vessel_step_rk4",
    # Guidance
    "LOSGuidance",
    "LOSConfig",
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
]

__version__ = "0.1.0"
