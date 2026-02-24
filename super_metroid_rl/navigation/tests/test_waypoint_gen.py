"""Tests for automatic waypoint generation."""

import unittest

from super_metroid_rl.navigation.map_data import DEFAULT_EXPORT_DIR, load_world
from super_metroid_rl.navigation.waypoint_gen import (
    generate_all_segment_waypoints,
    generate_segment_waypoints,
    needs_waypoints,
)
from super_metroid_rl.navigation.route import SPEEDRUN_ROUTE


def _has_export_data() -> bool:
    return DEFAULT_EXPORT_DIR.exists() and (DEFAULT_EXPORT_DIR / "nav_graph.json").exists()


@unittest.skipUnless(_has_export_data(), "No SM export data at /tmp/sm_export/")
class TestWaypointGeneration(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.world = load_world()

    def test_parlor_descent_waypoints(self):
        """Parlor descent should generate at least 3 waypoints (entry, mid, exit)."""
        waypoints = generate_segment_waypoints(
            self.world,
            "sm_parlor_descent",
            0x92FD,
            (1260, 128),  # entry from Landing Site
            0x96BA,        # exit to Climb
        )
        self.assertGreaterEqual(len(waypoints), 3,
                                f"Parlor descent waypoints: {waypoints}")
        # First waypoint should be near entry
        self.assertAlmostEqual(waypoints[0][0], 1260, delta=10)
        # Last waypoint should be near bottom of room
        room = self.world.rooms[0x92FD]
        room_height = room.height_blocks * 16
        self.assertGreater(waypoints[-1][1], room_height * 0.5,
                           f"Last waypoint y={waypoints[-1][1]} should be in bottom half")

    def test_landing_site_waypoints(self):
        """Landing Site → Parlor should have waypoints going left."""
        waypoints = generate_segment_waypoints(
            self.world,
            "sm_landing_site",
            0x91F8,
            (1100, 600),
            0x92FD,
        )
        self.assertGreaterEqual(len(waypoints), 2)
        # Should move left (decreasing x)
        self.assertGreater(waypoints[0][0], waypoints[-1][0],
                           "Landing Site waypoints should go left (decreasing x)")

    def test_flyway_waypoints(self):
        """Flyway is a simple corridor - should still generate valid waypoints."""
        waypoints = generate_segment_waypoints(
            self.world,
            "sm_flyway_to_torizo",
            0x9879,
            (20, 128),
            0x9804,
        )
        self.assertGreaterEqual(len(waypoints), 2)

    def test_morph_ball_collect_waypoints(self):
        """Morph ball collect (exit_room=0) should generate waypoints going right."""
        waypoints = generate_segment_waypoints(
            self.world,
            "sm_morph_ball_collect",
            0x9E9F,
            (1400, 32),
            0,
        )
        self.assertGreaterEqual(len(waypoints), 2)
        # Should go right
        self.assertGreater(waypoints[-1][0], waypoints[0][0])

    def test_all_segments_generate_waypoints(self):
        """Every segment in the route should produce at least 2 waypoints."""
        all_wps = generate_all_segment_waypoints(self.world)
        for step in SPEEDRUN_ROUTE:
            self.assertIn(step.segment_id, all_wps,
                          f"Missing waypoints for {step.segment_id}")
            wps = all_wps[step.segment_id]
            self.assertGreaterEqual(len(wps), 2,
                                    f"{step.segment_id} has only {len(wps)} waypoints")

    def test_waypoints_are_float_tuples(self):
        """All waypoints should be (float, float) tuples."""
        all_wps = generate_all_segment_waypoints(self.world)
        for seg_id, wps in all_wps.items():
            for i, (x, y) in enumerate(wps):
                self.assertIsInstance(x, float, f"{seg_id} wp[{i}].x is not float")
                self.assertIsInstance(y, float, f"{seg_id} wp[{i}].y is not float")


@unittest.skipUnless(_has_export_data(), "No SM export data at /tmp/sm_export/")
class TestNeedsWaypoints(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.world = load_world()

    def test_parlor_needs_waypoints(self):
        """Parlor (5x5) should need waypoints."""
        self.assertTrue(needs_waypoints("sm_parlor_descent", self.world))

    def test_flyway_does_not_need_waypoints(self):
        """Flyway (3x1) should not need waypoints."""
        self.assertFalse(needs_waypoints("sm_flyway_to_torizo", self.world))

    def test_elevator_does_not_need_waypoints(self):
        """Elevator (1x1) should not need waypoints."""
        self.assertFalse(needs_waypoints("sm_elevator_descent", self.world))

    def test_morph_collect_does_not_need_waypoints(self):
        """Item collect (exit_room=0) should not need waypoints."""
        self.assertFalse(needs_waypoints("sm_morph_ball_collect", self.world))


if __name__ == "__main__":
    unittest.main()
