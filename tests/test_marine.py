"""
Marine Autonomy Stack — unit tests.

≥ 22 tests covering:
  Contracts, Dynamics, Guidance, Estimation, COLREGs,
  Orchestrator, AIS Adapter, Presets, and Ω logic.

Run with:
  python -m pytest tests/test_marine.py -v
or plain Python:
  python tests/test_marine.py
"""
from __future__ import annotations

import math
import sys
import os
import unittest

# Ensure the package root is on the path when running directly
_ROOT = os.path.join(os.path.dirname(__file__), "..")
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from marine_autonomy.contracts.schemas import (
    VesselState,
    VesselCommand,
    ContactVessel,
    MarinePerception,
    VesselActuator,
    MarineTickContext,
)
from marine_autonomy.dynamics import VesselParams, vessel_step, vessel_step_rk4
from marine_autonomy.guidance import LOSGuidance, LOSConfig
from marine_autonomy.estimation import MarineEKF, MarineEKFNoise
from marine_autonomy.colregs import (
    COLREGsBehavior,
    COLREGsConfig,
    classify_contact,
)
from marine_autonomy.orchestrator import VesselOrchestrator, _compute_omega
from marine_autonomy.presets import get_preset
from marine_autonomy.adapters.ais_adapter import (
    parse_ais_contact,
    contacts_to_perception,
    ego_state_from_nmea,
)


class TestContracts(unittest.TestCase):
    """Tests 1–2: data contract defaults."""

    def test_vessel_state_defaults(self):
        """1. VesselState default construction yields all-zero floats."""
        s = VesselState()
        self.assertEqual(s.x_m, 0.0)
        self.assertEqual(s.y_m, 0.0)
        self.assertEqual(s.psi_rad, 0.0)
        self.assertEqual(s.u_ms, 0.0)
        self.assertEqual(s.v_ms, 0.0)
        self.assertEqual(s.r_rads, 0.0)
        self.assertEqual(s.t_s, 0.0)

    def test_vessel_command_defaults(self):
        """2. VesselCommand default construction yields zero thrust and rudder."""
        c = VesselCommand()
        self.assertEqual(c.thrust_n, 0.0)
        self.assertEqual(c.rudder_rad, 0.0)

    def test_marine_perception_defaults(self):
        """18. MarinePerception defaults: empty contacts, high visibility."""
        p = MarinePerception()
        self.assertEqual(p.contacts, ())
        self.assertGreater(p.visibility_m, 0.0)
        self.assertGreater(p.depth_m, 0.0)


class TestDynamics(unittest.TestCase):
    """Tests 3–5: Fossen 3-DOF vessel dynamics."""

    def setUp(self):
        self.params = VesselParams()

    def test_forward_thrust_increases_surge(self):
        """3. Positive thrust increases surge velocity over time."""
        state = VesselState(psi_rad=0.0)
        cmd = VesselCommand(thrust_n=2000.0)
        new_state = vessel_step(state, cmd, self.params, dt_s=0.5)
        self.assertGreater(new_state.u_ms, state.u_ms)

    def test_zero_thrust_decelerates(self):
        """4. Zero thrust with non-zero surge decelerates due to damping."""
        state = VesselState(u_ms=3.0)
        cmd = VesselCommand(thrust_n=0.0)
        new_state = vessel_step(state, cmd, self.params, dt_s=1.0)
        self.assertLess(new_state.u_ms, state.u_ms)

    def test_rk4_same_direction_as_euler(self):
        """5. RK4 and Euler both predict positive surge under same thrust."""
        state = VesselState(psi_rad=0.0)
        cmd = VesselCommand(thrust_n=2000.0)
        euler = vessel_step(state, cmd, self.params, dt_s=0.1)
        rk4 = vessel_step_rk4(state, cmd, self.params, dt_s=0.1)
        # Both should show increased surge
        self.assertGreater(euler.u_ms, 0.0)
        self.assertGreater(rk4.u_ms, 0.0)
        # RK4 and Euler should be close for small dt
        self.assertAlmostEqual(euler.u_ms, rk4.u_ms, places=3)

    def test_time_advances(self):
        """Dynamics integrator advances t_s by dt."""
        state = VesselState(t_s=10.0)
        cmd = VesselCommand()
        new_state = vessel_step_rk4(state, cmd, VesselParams(), dt_s=0.1)
        self.assertAlmostEqual(new_state.t_s, 10.1, places=5)

    def test_heading_unchanged_with_zero_rudder(self):
        """Zero rudder and no initial yaw rate: heading stays constant."""
        state = VesselState(psi_rad=0.5, u_ms=2.0)
        cmd = VesselCommand(thrust_n=1000.0, rudder_rad=0.0)
        new_state = vessel_step_rk4(state, cmd, VesselParams(), dt_s=0.5)
        self.assertAlmostEqual(new_state.psi_rad, 0.5, places=3)


class TestGuidance(unittest.TestCase):
    """Tests 6–8: LOS guidance."""

    def test_cross_track_error_sign_port(self):
        """6a. Vessel to port of track → positive cross-track error."""
        los = LOSGuidance()
        # Track along y-axis (North): wp_from=(0,0), wp_to=(0,100)
        wp_from = (0.0, 0.0)
        wp_to = (0.0, 100.0)
        # Vessel at x=-10 (port of the North track)
        state = VesselState(x_m=-10.0, y_m=50.0, psi_rad=0.0)
        e = los._cross_track_error(state, wp_from, wp_to)
        self.assertGreater(e, 0.0)

    def test_cross_track_error_sign_starboard(self):
        """6b. Vessel to starboard of track → negative cross-track error."""
        los = LOSGuidance()
        wp_from = (0.0, 0.0)
        wp_to = (0.0, 100.0)
        state = VesselState(x_m=10.0, y_m=50.0, psi_rad=0.0)
        e = los._cross_track_error(state, wp_from, wp_to)
        self.assertLess(e, 0.0)

    def test_waypoint_switching_on_acceptance(self):
        """7. Waypoint index advances when vessel is within acceptance radius."""
        config = LOSConfig(acceptance_radius_m=20.0)
        los = LOSGuidance(config)
        # First waypoint at (0, 0); vessel is right on top of it
        waypoints = ((0.0, 0.0), (100.0, 100.0))
        state = VesselState(x_m=0.0, y_m=0.0)
        # update() should switch to waypoint index 1
        los.update(state, waypoints)
        self.assertEqual(los._wp_idx, 1)

    def test_guidance_rudder_direction_correct(self):
        """8. LOS rudder is positive (starboard) when waypoint is to port of current track.

        Coordinate convention: psi=0 means ship heading along +x axis (East).
        A waypoint to the North (+y) from a ship pointing East (psi=0)
        requires a port-to-starboard (left) turn — negative psi_err → positive rudder
        in the LOS PD law: rudder = Kp*psi_err, psi_d(=pi/2) - psi(=0) > 0 → positive.
        """
        config = LOSConfig(lookahead_m=50.0, acceptance_radius_m=5.0)
        los = LOSGuidance(config)
        # Waypoint directly North (+y) of origin
        waypoints = ((0.0, 100.0),)
        # Vessel at origin heading East (psi=0 in our convention = +x direction)
        state = VesselState(x_m=0.0, y_m=0.0, psi_rad=0.0)
        rudder = los.update(state, waypoints)
        # Path angle = atan2(100, 0) = pi/2; psi=0; error = pi/2 > 0 → positive rudder
        self.assertGreater(rudder, 0.0)

    def test_guidance_complete_when_no_waypoints(self):
        """is_complete returns True when waypoints is empty."""
        los = LOSGuidance()
        self.assertTrue(los.is_complete(()))

    def test_guidance_zero_rudder_when_complete(self):
        """update returns 0.0 when all waypoints are exhausted."""
        los = LOSGuidance(LOSConfig(acceptance_radius_m=200.0))
        # Vessel right at the waypoint → it gets consumed → returns 0
        waypoints = ((0.0, 0.0),)
        state = VesselState(x_m=0.0, y_m=0.0)
        rudder = los.update(state, waypoints)
        self.assertEqual(rudder, 0.0)


class TestEstimation(unittest.TestCase):
    """Tests 9–10: Marine EKF."""

    def test_predict_increases_uncertainty(self):
        """9. EKF predict step increases covariance diagonal (P grows)."""
        ekf = MarineEKF()
        P_before = [ekf._P[i][i] for i in range(4)]
        ekf.predict(r_rads=0.0, dt_s=1.0)
        P_after = [ekf._P[i][i] for i in range(4)]
        for b, a in zip(P_before, P_after):
            self.assertGreaterEqual(a, b)

    def test_gps_update_reduces_position_uncertainty(self):
        """10. GPS update reduces position covariance diagonal."""
        ekf = MarineEKF()
        # Force high initial uncertainty
        ekf._P[0][0] = 100.0
        ekf._P[1][1] = 100.0
        P_before_x = ekf._P[0][0]
        P_before_y = ekf._P[1][1]
        ekf.update_gps([0.0, 0.0])
        self.assertLess(ekf._P[0][0], P_before_x)
        self.assertLess(ekf._P[1][1], P_before_y)

    def test_ekf_estimate_returns_state(self):
        """EKF estimate() returns a StateEstimate with correct fields."""
        from marine_autonomy.estimation import StateEstimate
        ekf = MarineEKF()
        ekf.set_state(10.0, 20.0, 0.5, 2.0)
        est = ekf.estimate()
        self.assertIsInstance(est, StateEstimate)
        self.assertAlmostEqual(est.x_m, 10.0)
        self.assertAlmostEqual(est.y_m, 20.0)

    def test_heading_update_reduces_heading_uncertainty(self):
        """Heading update reduces psi covariance diagonal."""
        ekf = MarineEKF()
        ekf._P[2][2] = 50.0
        P_before = ekf._P[2][2]
        ekf.update_heading(0.0)
        self.assertLess(ekf._P[2][2], P_before)


class TestCOLREGs(unittest.TestCase):
    """Tests 11–13: COLREGs classification and FSM."""

    def _make_ego(self, psi_rad: float = 0.0) -> VesselState:
        return VesselState(x_m=0.0, y_m=0.0, psi_rad=psi_rad)

    def test_head_on_classification(self):
        """11. HEAD_ON when contact is ahead with opposing course."""
        config = COLREGsConfig(safe_range_m=1000.0)
        ego = self._make_ego(psi_rad=0.0)
        # Contact dead ahead (bearing 0°), opposite course (cog ≈ π)
        contact = ContactVessel(
            id="TGT1",
            range_m=300.0,
            bearing_rad=0.0,
            cog_rad=math.pi,  # heading South (opposing)
            sog_ms=5.0,
        )
        result = classify_contact(ego, contact, config)
        self.assertEqual(result, "HEAD_ON")

    def test_crossing_give_way_classification(self):
        """12. CROSSING_GIVE_WAY when contact is on own starboard bow."""
        config = COLREGsConfig(safe_range_m=1000.0)
        ego = self._make_ego(psi_rad=0.0)
        # Contact at 45° on starboard (bearing_deg=45)
        bearing_rad = math.radians(45.0)
        contact = ContactVessel(
            id="TGT2",
            range_m=300.0,
            bearing_rad=bearing_rad,
            cog_rad=math.pi,  # any non-opposing course
            sog_ms=5.0,
        )
        result = classify_contact(ego, contact, config)
        self.assertEqual(result, "CROSSING_GIVE_WAY")

    def test_emergency_range_triggers_stop(self):
        """13. Contact inside emergency_range_m triggers EMERGENCY_STOP."""
        config = COLREGsConfig(
            safe_range_m=300.0,
            action_range_m=500.0,
            emergency_range_m=50.0,
        )
        colregs = COLREGsBehavior()
        ego = VesselState()
        contact = ContactVessel(id="CLOSE", range_m=20.0)
        perception = MarinePerception(contacts=(contact,))
        result = colregs.tick(ego, perception, config)
        self.assertEqual(result["state"], "EMERGENCY_STOP")
        self.assertTrue(result["stop"])

    def test_safe_contact_yields_cruise(self):
        """Contacts beyond safe_range_m yield CRUISE state."""
        config = COLREGsConfig(safe_range_m=100.0)
        colregs = COLREGsBehavior()
        ego = VesselState()
        contact = ContactVessel(id="FAR", range_m=500.0)
        perception = MarinePerception(contacts=(contact,))
        result = colregs.tick(ego, perception, config)
        self.assertEqual(result["state"], "CRUISE")
        self.assertFalse(result["stop"])


class TestOrchestrator(unittest.TestCase):
    """Tests 14–16 + 19–20: VesselOrchestrator."""

    def test_tick_returns_marine_tick_context(self):
        """14. tick() returns a MarineTickContext instance."""
        orch = VesselOrchestrator(preset="harbor")
        ctx = MarineTickContext(state=VesselState())
        result = orch.tick(ctx, dt_s=0.1)
        self.assertIsInstance(result, MarineTickContext)

    def test_orchestrator_with_waypoints_nonzero_rudder(self):
        """15. With a waypoint requiring a turn, rudder_norm should be nonzero.

        Ship at origin heading East (psi=0), waypoint to the North (+y).
        Path angle = pi/2; heading error = pi/2; should produce positive rudder.
        """
        orch = VesselOrchestrator(preset="harbor", use_ekf=False)
        ctx = MarineTickContext(
            state=VesselState(x_m=0.0, y_m=0.0, psi_rad=0.0),
            waypoints=((0.0, 200.0),),  # North — requires starboard turn from East heading
        )
        result = orch.tick(ctx, dt_s=0.1)
        self.assertNotEqual(result.actuator.rudder_norm, 0.0)

    def test_orchestrator_without_waypoints_zero_rudder(self):
        """16. Without waypoints, rudder_norm should be zero."""
        orch = VesselOrchestrator(preset="harbor", use_ekf=False)
        ctx = MarineTickContext(
            state=VesselState(),
            waypoints=(),
        )
        result = orch.tick(ctx, dt_s=0.1)
        self.assertEqual(result.actuator.rudder_norm, 0.0)

    def test_omega_healthy_when_no_threats(self):
        """19. Ω = 1.0 and verdict HEALTHY under benign conditions."""
        omega = _compute_omega(
            risk_score=0.0,
            fuel_level=1.0,
            visibility_m=5000.0,
            depth_m=50.0,
            draft_m=2.0,
        )
        self.assertAlmostEqual(omega, 1.0, places=5)
        from marine_autonomy.orchestrator import _omega_to_verdict
        self.assertEqual(_omega_to_verdict(omega), "HEALTHY")

    def test_omega_critical_when_depth_shallow(self):
        """20. Ω drops to 0.65 when depth < 3 × draft."""
        omega = _compute_omega(
            risk_score=0.0,
            fuel_level=1.0,
            visibility_m=5000.0,
            depth_m=3.0,   # < 3 × 2.0 = 6.0
            draft_m=2.0,
        )
        # Only depth penalty applies: 0.65
        self.assertAlmostEqual(omega, 0.65, places=5)

    def test_orchestrator_verdict_in_result(self):
        """Orchestrator sets verdict field in returned context."""
        orch = VesselOrchestrator(preset="coastal")
        ctx = MarineTickContext(state=VesselState())
        result = orch.tick(ctx, dt_s=0.1)
        self.assertIn(result.verdict, ["HEALTHY", "DEGRADED", "CRITICAL", "EMERGENCY"])

    def test_orchestrator_time_advances(self):
        """Orchestrator t_s advances after each tick."""
        orch = VesselOrchestrator(preset="coastal", use_ekf=False)
        ctx = MarineTickContext(state=VesselState())
        r1 = orch.tick(ctx, dt_s=0.1)
        r2 = orch.tick(r1, dt_s=0.1)
        self.assertGreater(r2.t_s, r1.t_s)


class TestAISAdapter(unittest.TestCase):
    """Test 17: AIS adapter parsing."""

    def test_parse_ais_contact(self):
        """17. parse_ais_contact converts degrees/knots to radians/m/s."""
        msg = {
            "id": "123456789",
            "range_m": 400.0,
            "bearing_deg": 90.0,
            "cog_deg": 270.0,
            "sog_kn": 10.0,
        }
        contact = parse_ais_contact(msg)
        self.assertEqual(contact.id, "123456789")
        self.assertAlmostEqual(contact.range_m, 400.0)
        self.assertAlmostEqual(contact.bearing_rad, math.pi / 2, places=4)
        self.assertAlmostEqual(contact.cog_rad, 3 * math.pi / 2, places=4)
        self.assertAlmostEqual(contact.sog_ms, 10.0 * 0.5144, places=3)

    def test_contacts_to_perception(self):
        """contacts_to_perception assembles MarinePerception correctly."""
        msgs = [
            {"id": "A", "range_m": 200.0, "bearing_deg": 0.0, "cog_deg": 180.0, "sog_kn": 5.0},
            {"id": "B", "range_m": 600.0, "bearing_deg": 45.0, "cog_deg": 90.0, "sog_kn": 8.0},
        ]
        perc = contacts_to_perception(msgs, visibility_m=1500.0, depth_m=20.0)
        self.assertEqual(len(perc.contacts), 2)
        self.assertAlmostEqual(perc.visibility_m, 1500.0)
        self.assertAlmostEqual(perc.depth_m, 20.0)

    def test_ego_state_from_nmea(self):
        """ego_state_from_nmea converts hdg_deg and sog_kn correctly."""
        nmea = {"x_m": 10.0, "y_m": 20.0, "hdg_deg": 90.0, "sog_kn": 5.0, "t_s": 100.0}
        state = ego_state_from_nmea(nmea)
        self.assertAlmostEqual(state.x_m, 10.0)
        self.assertAlmostEqual(state.y_m, 20.0)
        self.assertAlmostEqual(state.psi_rad, math.pi / 2, places=4)
        self.assertAlmostEqual(state.u_ms, 5.0 * 0.5144, places=3)
        self.assertAlmostEqual(state.t_s, 100.0)


class TestPresets(unittest.TestCase):
    """Tests 21–22: preset registry."""

    def test_harbor_cruise_speed_knots(self):
        """21. Harbor preset cruise speed ≈ 3 kn."""
        p = get_preset("harbor")
        kn = p.speed_kn()
        self.assertAlmostEqual(kn, 3.0, delta=0.1)

    def test_ocean_cruise_speed_knots(self):
        """22. Ocean preset cruise speed ≈ 20 kn."""
        p = get_preset("ocean")
        kn = p.speed_kn()
        self.assertAlmostEqual(kn, 20.0, delta=0.1)

    def test_fallback_preset_is_coastal(self):
        """Unknown preset name falls back to coastal."""
        p = get_preset("nonexistent_preset")
        self.assertEqual(p.name, "coastal")

    def test_river_preset_draft(self):
        """River preset has the smallest draft."""
        r = get_preset("river")
        o = get_preset("ocean")
        self.assertLess(r.draft_m, o.draft_m)


if __name__ == "__main__":
    unittest.main(verbosity=2)
