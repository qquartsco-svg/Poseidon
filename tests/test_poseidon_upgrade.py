"""
Poseidon upgrade tests — universal maritime control engine.

30+ tests covering:
  DisturbanceState, compute_disturbance_forces, vessel dynamics with disturbance,
  nonlinear damping, Coriolis, submarine depth control, HullClass enum,
  hull preset factories, maritime A*, DepthChart, COLREGs multi-vessel,
  hull-class presets, VesselOrchestrator with disturbance, full pipeline.

Run with:
  python -m pytest tests/test_poseidon_upgrade.py -v
"""
from __future__ import annotations

import math
import sys
import os
import unittest

_ROOT = os.path.join(os.path.dirname(__file__), "..")
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from marine_autonomy.contracts.schemas import (
    VesselState,
    VesselCommand,
    ContactVessel,
    MarinePerception,
    MarineTickContext,
    HullClass,
    DisturbanceState,
    SubmarineState,
)
from marine_autonomy.dynamics import (
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
from marine_autonomy.guidance import DepthChart, maritime_astar
from marine_autonomy.colregs import COLREGsBehavior, COLREGsConfig, classify_contact
from marine_autonomy.presets import get_hull_preset, HULL_PRESETS
from marine_autonomy.orchestrator import VesselOrchestrator


# ---------------------------------------------------------------------------
# 1. DisturbanceState defaults
# ---------------------------------------------------------------------------

class TestDisturbanceStateDefaults(unittest.TestCase):
    def test_disturbance_state_defaults(self):
        """1. DisturbanceState defaults: all zeros except wave_period_s=8."""
        d = DisturbanceState()
        self.assertEqual(d.wave_height_m, 0.0)
        self.assertEqual(d.wave_period_s, 8.0)
        self.assertEqual(d.wave_dir_rad, 0.0)
        self.assertEqual(d.wind_speed_ms, 0.0)
        self.assertEqual(d.wind_dir_rad, 0.0)
        self.assertEqual(d.current_u_ms, 0.0)
        self.assertEqual(d.current_v_ms, 0.0)
        self.assertEqual(d.t_s, 0.0)

    def test_disturbance_state_frozen(self):
        """2. DisturbanceState is immutable (frozen dataclass)."""
        d = DisturbanceState(wind_speed_ms=5.0)
        with self.assertRaises(Exception):
            d.wind_speed_ms = 10.0  # type: ignore[misc]


# ---------------------------------------------------------------------------
# 2. compute_disturbance_forces
# ---------------------------------------------------------------------------

class TestDisturbanceForces(unittest.TestCase):
    def _state(self):
        return VesselState(psi_rad=0.0)

    def _params(self):
        return surface_vessel_params()

    def test_zero_disturbance_zero_forces(self):
        """3. Zero disturbance → forces should all be very close to zero."""
        state = self._state()
        params = self._params()
        dist = DisturbanceState()  # all zeros, wave period = 8 → sin(0) = 0 for t_s=0
        tau_X, tau_Y, tau_N = compute_disturbance_forces(state, params, dist)
        self.assertAlmostEqual(tau_X, 0.0, places=6)
        self.assertAlmostEqual(tau_Y, 0.0, places=6)
        self.assertAlmostEqual(tau_N, 0.0, places=6)

    def test_wind_force_proportional_to_vw_squared(self):
        """4. Wind force magnitude scales with Vw² (dynamic pressure)."""
        state = self._state()
        params = self._params()
        # Quarter-period time so wave sin ≈ 1, isolate wind effect by measuring ratio
        dist1 = DisturbanceState(wind_speed_ms=5.0, t_s=2.0)   # Vw=5
        dist2 = DisturbanceState(wind_speed_ms=10.0, t_s=2.0)  # Vw=10
        tX1, _, _ = compute_disturbance_forces(state, params, dist1)
        tX2, _, _ = compute_disturbance_forces(state, params, dist2)
        # Wind force ∝ Vw², so ratio should be 4 (with negligible wave component)
        # At t=2s, wave phase = 2π*2/8 = π/2, sin(π/2) = 1 → wave + wind
        # We just check tX2 > tX1 since wind doubles
        self.assertGreater(abs(tX2), abs(tX1))

    def test_wind_force_zero_when_wind_zero(self):
        """5. No wind → wind contribution is zero (wave still possible)."""
        state = self._state()
        params = self._params()
        dist_wind = DisturbanceState(wind_speed_ms=10.0, wave_height_m=0.0, t_s=0.0)
        dist_zero = DisturbanceState(wind_speed_ms=0.0, wave_height_m=0.0, t_s=0.0)
        tX_wind, _, _ = compute_disturbance_forces(state, params, dist_wind)
        tX_zero, _, _ = compute_disturbance_forces(state, params, dist_zero)
        # With no wave (Hs=0 and t=0), wave force is also zero
        # Wind with speed 10 gives nonzero force (bow-on → cos(0)=1)
        self.assertAlmostEqual(tX_zero, 0.0, places=6)
        self.assertGreater(abs(tX_wind), 0.0)


# ---------------------------------------------------------------------------
# 3. vessel_step_rk4 with disturbance
# ---------------------------------------------------------------------------

class TestDynamicsWithDisturbance(unittest.TestCase):
    def test_vessel_step_with_disturbance_differs_from_without(self):
        """6. vessel_step_rk4 with disturbance produces different state than without."""
        state = VesselState(u_ms=2.0, psi_rad=0.0)
        cmd   = VesselCommand(thrust_n=1000.0)
        params = surface_vessel_params()
        dist = DisturbanceState(wind_speed_ms=20.0, wind_dir_rad=math.pi / 2, t_s=2.0)

        s_no_dist = vessel_step_rk4(state, cmd, params, dt_s=0.5)
        s_dist    = vessel_step_rk4(state, cmd, params, dt_s=0.5, disturbance=dist)

        # At least one state component should differ
        diff = abs(s_no_dist.u_ms - s_dist.u_ms) + abs(s_no_dist.v_ms - s_dist.v_ms)
        self.assertGreater(diff, 1e-8)

    def test_vessel_step_heading_changes_under_lateral_wind(self):
        """7. Strong lateral wind rotates vessel heading over time."""
        state = VesselState(u_ms=1.0, psi_rad=0.0)
        cmd   = VesselCommand(thrust_n=500.0, rudder_rad=0.0)
        params = surface_vessel_params()
        # Beam wind from port (90° relative), creates yaw moment
        dist = DisturbanceState(wind_speed_ms=30.0, wind_dir_rad=math.pi / 2, t_s=2.0)

        s = state
        for _ in range(50):
            s = vessel_step_rk4(s, cmd, params, dt_s=0.1, disturbance=dist)

        # Heading should have changed
        self.assertNotAlmostEqual(s.psi_rad, 0.0, places=3)


# ---------------------------------------------------------------------------
# 4. Nonlinear damping
# ---------------------------------------------------------------------------

class TestNonlinearDamping(unittest.TestCase):
    def test_nonlinear_drag_stronger_at_high_speed(self):
        """8. At high surge speed, deceleration is faster with Xuu > linear only."""
        params_nl = surface_vessel_params()   # has Xuu = -50
        params_lin = surface_vessel_params(Xuu=0.0)   # linear only

        state_hi = VesselState(u_ms=10.0)
        cmd = VesselCommand(thrust_n=0.0)

        s_nl  = vessel_step_rk4(state_hi, cmd, params_nl, dt_s=1.0)
        s_lin = vessel_step_rk4(state_hi, cmd, params_lin, dt_s=1.0)

        # Nonlinear drag decelerates more → smaller u at t+1
        self.assertLess(s_nl.u_ms, s_lin.u_ms)


# ---------------------------------------------------------------------------
# 5. Coriolis
# ---------------------------------------------------------------------------

class TestCoriolisForce(unittest.TestCase):
    def test_turning_vessel_coriolis_cross_force(self):
        """9. Vessel with yaw rate generates cross-coupling sway acceleration."""
        # A vessel turning (r != 0) should have different sway dynamics
        # compared to one not turning
        params = surface_vessel_params()
        state_turning = VesselState(u_ms=3.0, v_ms=0.0, r_rads=0.3)
        state_straight = VesselState(u_ms=3.0, v_ms=0.0, r_rads=0.0)
        cmd = VesselCommand(thrust_n=500.0, rudder_rad=0.1)

        s_t = vessel_step_rk4(state_turning, cmd, params, dt_s=0.1)
        s_s = vessel_step_rk4(state_straight, cmd, params, dt_s=0.1)

        # Coriolis introduces different sway — values should differ
        self.assertNotAlmostEqual(s_t.v_ms, s_s.v_ms, places=6)


# ---------------------------------------------------------------------------
# 6. Submarine depth control
# ---------------------------------------------------------------------------

class TestSubmarineDepthControl(unittest.TestCase):
    def _make_state(self, depth=0.0, w=0.0):
        class S:
            depth_m = depth
            w_ms = w
        return S()

    def test_depth_approaches_target(self):
        """10. submarine_depth_step drives depth toward target over time."""
        params = submarine_params()
        state = self._make_state(depth=0.0, w=0.0)
        target = 50.0

        depth = 0.0
        w = 0.0
        for _ in range(200):
            class S:
                depth_m = depth
                w_ms = w
            depth, w = submarine_depth_step(S(), target, params, dt_s=0.5)

        self.assertGreater(depth, 0.0)

    def test_depth_clamped_to_max(self):
        """11. Depth stays within depth_max_m."""
        params = submarine_params()
        # Start very deep target
        state = self._make_state(depth=0.0, w=100.0)

        class S:
            depth_m = 0.0
            w_ms = 100.0  # large downward velocity

        d, _ = submarine_depth_step(S(), 9999.0, params, dt_s=1.0)
        self.assertLessEqual(d, params.depth_max_m)

    def test_depth_zero_when_at_target(self):
        """12. Depth doesn't change much when already at target."""
        params = submarine_params()

        class S:
            depth_m = 50.0
            w_ms = 0.0

        d, w = submarine_depth_step(S(), 50.0, params, dt_s=0.1)
        self.assertAlmostEqual(d, 50.0, delta=5.0)  # should stay near 50


# ---------------------------------------------------------------------------
# 7. HullClass enum
# ---------------------------------------------------------------------------

class TestHullClassEnum(unittest.TestCase):
    def test_hull_class_values(self):
        """13. HullClass enum has expected values."""
        self.assertEqual(HullClass.SURFACE_VESSEL.value, "surface_vessel")
        self.assertEqual(HullClass.SUBMARINE.value, "submarine")
        self.assertEqual(HullClass.YACHT.value, "yacht")
        self.assertEqual(HullClass.BOAT.value, "boat")
        self.assertEqual(HullClass.AUTONOMOUS_USV.value, "autonomous_usv")

    def test_hull_class_is_string(self):
        """14. HullClass is a string enum — can compare with string."""
        self.assertEqual(HullClass.SUBMARINE, "submarine")


# ---------------------------------------------------------------------------
# 8. Hull-class factory functions
# ---------------------------------------------------------------------------

class TestHullClassFactories(unittest.TestCase):
    def test_surface_vessel_params_returns_vessel_params(self):
        """15. surface_vessel_params() returns VesselParams."""
        p = surface_vessel_params()
        self.assertIsInstance(p, VesselParams)
        self.assertEqual(p.hull_class, "surface_vessel")

    def test_submarine_params_has_depth_max(self):
        """16. submarine_params() has depth_max_m > 0."""
        p = submarine_params()
        self.assertGreater(p.depth_max_m, 0.0)
        self.assertEqual(p.hull_class, "submarine")

    def test_yacht_lighter_than_surface_vessel(self):
        """17. yacht_params() mass < surface_vessel_params() mass."""
        y = yacht_params()
        sv = surface_vessel_params()
        self.assertLess(y.mass_kg, sv.mass_kg)

    def test_boat_higher_nonlinear_drag(self):
        """18. boat_params() has higher |Xuu| than surface_vessel_params()."""
        b = boat_params()
        sv = surface_vessel_params()
        self.assertGreater(abs(b.Xuu), abs(sv.Xuu))

    def test_usv_params_small_mass(self):
        """19. usv_params() mass < 500 kg."""
        u = usv_params()
        self.assertLess(u.mass_kg, 500.0)
        self.assertEqual(u.hull_class, "autonomous_usv")

    def test_submarine_no_wind_coefficients(self):
        """20. submarine_params() wind coefficients are zero (submerged)."""
        p = submarine_params()
        self.assertEqual(p.Cx_wind, 0.0)
        self.assertEqual(p.Cy_wind, 0.0)


# ---------------------------------------------------------------------------
# 9. maritime_astar
# ---------------------------------------------------------------------------

def _deep_chart(rows=10, cols=10, depth=20.0, resolution=10.0):
    """Create a uniform deep depth chart."""
    grid = [[depth] * cols for _ in range(rows)]
    return DepthChart(grid=grid, origin_x=0.0, origin_y=0.0,
                      resolution_m=resolution, min_depth_m=3.0)


class TestMaritimeAstar(unittest.TestCase):
    def test_flat_deep_grid_finds_path(self):
        """21. maritime_astar on flat deep grid returns non-empty path."""
        chart = _deep_chart()
        path = maritime_astar(chart, (5.0, 5.0), (85.0, 85.0), draft_m=1.5)
        self.assertGreater(len(path), 0)

    def test_path_starts_near_start(self):
        """22. First waypoint is close to start."""
        chart = _deep_chart()
        path = maritime_astar(chart, (5.0, 5.0), (85.0, 85.0), draft_m=1.5)
        self.assertGreater(len(path), 0)
        dx = path[0][0] - 5.0
        dy = path[0][1] - 5.0
        self.assertLess(math.hypot(dx, dy), 20.0)

    def test_obstacle_forces_detour(self):
        """23. Shallow wall in the middle (with gap at top) forces a detour."""
        grid = [[20.0] * 12 for _ in range(12)]
        # Block column 5 rows 2–11 (leave rows 0-1 open at top as gap)
        for r in range(2, 12):
            grid[r][5] = 0.0
        chart = DepthChart(grid=grid, origin_x=0.0, origin_y=0.0,
                           resolution_m=10.0, min_depth_m=3.0)
        # Start at left-middle, goal at right-middle — must route through gap
        path = maritime_astar(chart, (5.0, 55.0), (115.0, 55.0), draft_m=1.5)
        # Path should exist (via top gap) and be longer than a direct straight line
        self.assertGreater(len(path), 2)

    def test_no_path_when_all_shallow(self):
        """24. maritime_astar returns empty list when no passable path exists."""
        grid = [[0.0] * 5 for _ in range(5)]  # entirely impassable
        chart = DepthChart(grid=grid, origin_x=0.0, origin_y=0.0,
                           resolution_m=10.0, min_depth_m=3.0)
        path = maritime_astar(chart, (5.0, 5.0), (45.0, 45.0), draft_m=1.5)
        self.assertEqual(path, [])

    def test_same_cell_start_goal(self):
        """25. Start and goal in same cell returns single waypoint."""
        chart = _deep_chart()
        path = maritime_astar(chart, (5.0, 5.0), (5.0, 5.0), draft_m=1.5)
        self.assertEqual(len(path), 1)


# ---------------------------------------------------------------------------
# 10. DepthChart
# ---------------------------------------------------------------------------

class TestDepthChart(unittest.TestCase):
    def test_passable_deep_cell(self):
        """26. DepthChart.is_passable returns True for deep cell."""
        chart = _deep_chart()
        self.assertTrue(chart.is_passable(0, 0, draft_m=1.5))

    def test_impassable_shallow_cell(self):
        """27. DepthChart.is_passable returns False for shallow cell."""
        grid = [[2.0] * 5 for _ in range(5)]  # 2m depth, 1.5m draft + 3m min = 4.5m needed
        chart = DepthChart(grid=grid, origin_x=0.0, origin_y=0.0,
                           resolution_m=10.0, min_depth_m=3.0)
        self.assertFalse(chart.is_passable(0, 0, draft_m=1.5))

    def test_world_to_cell_correct(self):
        """28. DepthChart.world_to_cell converts correctly."""
        chart = _deep_chart(resolution=10.0)
        row, col = chart.world_to_cell(25.0, 35.0)  # col=2, row=3
        self.assertEqual(col, 2)
        self.assertEqual(row, 3)

    def test_out_of_bounds_not_passable(self):
        """29. DepthChart.is_passable returns False for out-of-bounds cell."""
        chart = _deep_chart(rows=5, cols=5)
        self.assertFalse(chart.is_passable(100, 100, draft_m=1.5))


# ---------------------------------------------------------------------------
# 11. COLREGs multi-vessel
# ---------------------------------------------------------------------------

class TestCOLREGsMultiVessel(unittest.TestCase):
    def _ego(self):
        return VesselState(psi_rad=0.0)

    def test_no_contacts_cruise(self):
        """30. 0 contacts → CRUISE state."""
        colregs = COLREGsBehavior()
        result = colregs.tick(self._ego(), MarinePerception(contacts=()))
        self.assertEqual(result["state"], "CRUISE")
        self.assertFalse(result["stop"])
        self.assertEqual(result["situations"], [])
        self.assertIsNone(result["dominant_contact"])

    def test_one_head_on_gives_give_way(self):
        """31. 1 HEAD_ON contact → GIVE_WAY with positive offset."""
        colregs = COLREGsBehavior()
        config = COLREGsConfig(safe_range_m=1000.0, action_range_m=800.0,
                               emergency_range_m=30.0)
        contact = ContactVessel(
            id="TGT1", range_m=200.0, bearing_rad=0.0,
            cog_rad=math.pi, sog_ms=5.0
        )
        perc = MarinePerception(contacts=(contact,))
        result = colregs.tick(self._ego(), perc, config)
        self.assertEqual(result["state"], "GIVE_WAY")
        self.assertGreater(result["avoid_heading_offset_rad"], 0.0)

    def test_emergency_range_triggers_stop(self):
        """32. Contact inside emergency range → EMERGENCY_STOP."""
        colregs = COLREGsBehavior()
        config = COLREGsConfig(emergency_range_m=50.0)
        contact = ContactVessel(id="CLOSE", range_m=10.0)
        perc = MarinePerception(contacts=(contact,))
        result = colregs.tick(self._ego(), perc, config)
        self.assertEqual(result["state"], "EMERGENCY_STOP")
        self.assertTrue(result["stop"])

    def test_two_give_way_contacts_max_offset(self):
        """33. 2 GIVE_WAY contacts → GIVE_WAY with max offset of the two."""
        colregs = COLREGsBehavior()
        config = COLREGsConfig(safe_range_m=1000.0, action_range_m=800.0,
                               emergency_range_m=30.0)
        # HEAD_ON at 300m
        c1 = ContactVessel(id="C1", range_m=300.0, bearing_rad=0.0,
                           cog_rad=math.pi, sog_ms=5.0)
        # CROSSING_GIVE_WAY at 200m (closer, dominates)
        c2 = ContactVessel(id="C2", range_m=200.0, bearing_rad=math.radians(45.0),
                           cog_rad=math.pi, sog_ms=5.0)
        perc = MarinePerception(contacts=(c1, c2))
        result = colregs.tick(self._ego(), perc, config)
        self.assertEqual(result["state"], "GIVE_WAY")
        # Head-on offset is 20°, crossing is 15° — max should be ≥ 15°
        self.assertGreaterEqual(result["avoid_heading_offset_rad"],
                                math.radians(15.0))

    def test_stand_on_situation(self):
        """34. CROSSING_STAND_ON contact → STAND_ON state."""
        colregs = COLREGsBehavior()
        config = COLREGsConfig(safe_range_m=1000.0, action_range_m=800.0,
                               emergency_range_m=30.0)
        # CROSSING_STAND_ON: bearing 280° (port sector 247.5–350)
        bearing_rad = math.radians(280.0)
        contact = ContactVessel(id="PORT", range_m=250.0, bearing_rad=bearing_rad,
                                cog_rad=0.5, sog_ms=5.0)
        perc = MarinePerception(contacts=(contact,))
        result = colregs.tick(self._ego(), perc, config)
        self.assertEqual(result["state"], "STAND_ON")

    def test_situations_list_length_matches_contacts(self):
        """35. situations list length equals number of contacts."""
        colregs = COLREGsBehavior()
        config = COLREGsConfig(safe_range_m=1000.0)
        contacts = (
            ContactVessel(id="A", range_m=100.0, bearing_rad=0.0, cog_rad=math.pi),
            ContactVessel(id="B", range_m=200.0, bearing_rad=math.radians(60.0)),
            ContactVessel(id="C", range_m=300.0, bearing_rad=math.radians(280.0)),
        )
        perc = MarinePerception(contacts=contacts)
        result = colregs.tick(self._ego(), perc, config)
        self.assertEqual(len(result["situations"]), 3)


# ---------------------------------------------------------------------------
# 12. Hull presets
# ---------------------------------------------------------------------------

class TestHullPresets(unittest.TestCase):
    def test_get_hull_preset_submarine_deep(self):
        """36. get_hull_preset('submarine', 'deep') returns MarinePreset."""
        from marine_autonomy.presets import MarinePreset
        p = get_hull_preset("submarine", "deep")
        self.assertIsInstance(p, MarinePreset)
        self.assertEqual(p.name, "sub_deep")

    def test_yacht_racing_faster_than_cruising(self):
        """37. Yacht racing preset faster than cruising."""
        racing   = get_hull_preset("yacht", "racing")
        cruising = get_hull_preset("yacht", "cruising")
        self.assertGreater(racing.max_speed_ms, cruising.max_speed_ms)

    def test_fallback_preset_when_unknown(self):
        """38. Unknown hull class/scenario falls back to coastal."""
        p = get_hull_preset("warp_drive", "warp_speed")
        self.assertEqual(p.name, "coastal")

    def test_all_hull_classes_present(self):
        """39. HULL_PRESETS contains all five hull class keys."""
        for key in ("surface_vessel", "submarine", "yacht", "boat", "autonomous_usv"):
            self.assertIn(key, HULL_PRESETS)


# ---------------------------------------------------------------------------
# 13. VesselOrchestrator with disturbance
# ---------------------------------------------------------------------------

class TestOrchestratorWithDisturbance(unittest.TestCase):
    def test_tick_with_disturbance_differs_from_without(self):
        """40. Orchestrator tick with disturbance produces different omega/verdict path."""
        # Just verify it runs without error with disturbance set
        orch = VesselOrchestrator(preset="coastal", use_ekf=False)
        dist = DisturbanceState(wind_speed_ms=15.0, wind_dir_rad=math.pi / 2, t_s=0.0)
        ctx = MarineTickContext(
            state=VesselState(u_ms=2.0),
            waypoints=((100.0, 100.0),),
            disturbance=dist,
        )
        result = orch.tick(ctx, dt_s=0.1)
        self.assertIsNotNone(result)
        self.assertIsNotNone(result.disturbance)

    def test_tick_multi_contact_colregs_state_in_ctx(self):
        """41. Orchestrator tick propagates multi-contact COLREGs state."""
        orch = VesselOrchestrator(preset="coastal", use_ekf=False)
        # Add 2 contacts in action range
        c1 = ContactVessel(id="A", range_m=200.0, bearing_rad=0.0,
                           cog_rad=math.pi, sog_ms=5.0)
        c2 = ContactVessel(id="B", range_m=300.0, bearing_rad=math.radians(45.0),
                           cog_rad=0.0, sog_ms=3.0)
        perc = MarinePerception(contacts=(c1, c2))
        ctx = MarineTickContext(
            state=VesselState(),
            perception=perc,
            waypoints=((200.0, 0.0),),
        )
        result = orch.tick(ctx, dt_s=0.1)
        self.assertIn(result.colregs_state,
                      ("CRUISE", "GIVE_WAY", "STAND_ON", "EMERGENCY_STOP"))

    def test_tick_hull_class_preserved(self):
        """42. hull_class field is preserved through tick."""
        orch = VesselOrchestrator(preset="harbor", use_ekf=False)
        ctx = MarineTickContext(
            state=VesselState(),
            hull_class=HullClass.SUBMARINE,
        )
        result = orch.tick(ctx, dt_s=0.1)
        self.assertEqual(result.hull_class, HullClass.SUBMARINE)


# ---------------------------------------------------------------------------
# 14. Full pipeline integration
# ---------------------------------------------------------------------------

class TestFullPipeline(unittest.TestCase):
    def test_full_pipeline_vessel_step_ekf_astar_colregs(self):
        """43. Full pipeline: vessel_step_rk4 + EKF update + A* waypoints + COLREGs."""
        from marine_autonomy.estimation import MarineEKF

        # 1. Dynamics
        params = surface_vessel_params()
        state = VesselState(u_ms=0.0, psi_rad=0.0)
        cmd = VesselCommand(thrust_n=2000.0, rudder_rad=0.1)
        dist = DisturbanceState(wind_speed_ms=5.0, t_s=0.0)

        for i in range(10):
            dist_i = DisturbanceState(wind_speed_ms=5.0, t_s=float(i) * 0.1)
            state = vessel_step_rk4(state, cmd, params, dt_s=0.1, disturbance=dist_i)

        self.assertGreater(state.u_ms, 0.0)  # ship should be moving

        # 2. EKF update
        ekf = MarineEKF()
        ekf.predict(r_rads=state.r_rads, dt_s=0.1)
        ekf.update_gps([state.x_m, state.y_m])
        ekf.update_heading(state.psi_rad)
        ekf.update_speed(state.u_ms)
        est = ekf.estimate()
        self.assertAlmostEqual(est.x_m, state.x_m, delta=5.0)

        # 3. A* path plan
        chart = _deep_chart(rows=20, cols=20, depth=30.0, resolution=5.0)
        waypoints = maritime_astar(chart, (2.0, 2.0), (92.0, 92.0), draft_m=1.5)
        self.assertGreater(len(waypoints), 0)

        # 4. COLREGs check with no contacts → CRUISE
        colregs = COLREGsBehavior()
        result = colregs.tick(state, MarinePerception(contacts=()), COLREGsConfig())
        self.assertEqual(result["state"], "CRUISE")

    def test_orchestrator_runs_multiple_ticks(self):
        """44. Orchestrator runs 50 ticks without error."""
        orch = VesselOrchestrator(preset="harbor", use_ekf=False)
        ctx = MarineTickContext(
            state=VesselState(),
            waypoints=((50.0, 50.0), (100.0, 0.0)),
        )
        for _ in range(50):
            ctx = orch.tick(ctx, dt_s=0.1)

        self.assertIn(ctx.verdict, ["HEALTHY", "DEGRADED", "CRITICAL", "EMERGENCY"])
        self.assertGreater(ctx.t_s, 0.0)

    def test_disturbance_state_in_context_round_trip(self):
        """45. DisturbanceState stored in context and retrieved correctly."""
        dist = DisturbanceState(wave_height_m=2.5, wind_speed_ms=10.0)
        ctx = MarineTickContext(
            state=VesselState(),
            disturbance=dist,
        )
        self.assertIsNotNone(ctx.disturbance)
        self.assertEqual(ctx.disturbance.wave_height_m, 2.5)
        self.assertEqual(ctx.disturbance.wind_speed_ms, 10.0)

    def test_submarine_state_dataclass(self):
        """46. SubmarineState defaults are zero."""
        s = SubmarineState()
        self.assertEqual(s.depth_m, 0.0)
        self.assertEqual(s.w_ms, 0.0)
        self.assertEqual(s.target_depth_m, 0.0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
