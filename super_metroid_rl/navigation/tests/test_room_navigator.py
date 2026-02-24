"""Tests for intra-room screen-level navigation."""

import unittest

from super_metroid_rl.navigation.map_data import DEFAULT_EXPORT_DIR, load_room
from super_metroid_rl.navigation.room_navigator import RoomNavigator

ROOMS_DIR = DEFAULT_EXPORT_DIR / "rooms"


def _has_export_data() -> bool:
    return ROOMS_DIR.exists()


@unittest.skipUnless(_has_export_data(), "No SM export data at /tmp/sm_export/")
class TestLandingSiteNavigation(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.room = load_room(ROOMS_DIR / "room_91F8.json")
        cls.nav = RoomNavigator(cls.room)

    def test_screen_dimensions(self):
        self.assertEqual(self.nav._w, 9)
        self.assertEqual(self.nav._h, 5)

    def test_pixel_to_screen(self):
        # Top-left corner
        self.assertEqual(self.nav.pixel_to_screen(0, 0), (0, 0))
        # Center of screen (1,0)
        self.assertEqual(self.nav.pixel_to_screen(384, 128), (1, 0))
        # Bottom-right area
        self.assertEqual(self.nav.pixel_to_screen(2200, 1100), (8, 4))

    def test_screen_center(self):
        cx, cy = self.nav.screen_center(0, 0)
        self.assertEqual(cx, 128)
        self.assertEqual(cy, 128)

    def test_door_position_to_parlor(self):
        """Should find the left-side door to Parlor."""
        pos = self.nav.find_door_position(0x92FD)
        self.assertIsNotNone(pos)
        x, y = pos
        # Left-side door should be at x ≈ 0-16
        self.assertLess(x, 100, f"Parlor door at x={x} should be near left edge")

    def test_screen_path_same_screen(self):
        waypoints = self.nav.screen_path((500, 500), (600, 500))
        self.assertEqual(len(waypoints), 2)
        self.assertEqual(waypoints[0], (500, 500))
        self.assertEqual(waypoints[1], (600, 500))

    def test_screen_path_across_room(self):
        """Path from right to left should have intermediate waypoints.

        Use y=1100 which is screen row 4 (bottom area) — the top-left screens
        of Landing Site are disconnected by terrain.
        """
        start = (2200, 1100)  # right side, bottom row
        end = (50, 1100)      # left side, bottom row
        waypoints = self.nav.screen_path(start, end)
        self.assertGreater(len(waypoints), 2,
                           "Cross-room path should have intermediate screen waypoints")
        self.assertEqual(waypoints[0], start)
        self.assertEqual(waypoints[-1], end)

    def test_screen_adjacency_exists(self):
        """Adjacent screens should have connections."""
        adj = self.nav._adj
        # Screen (0,0) should connect to at least one neighbor
        self.assertGreater(len(adj.get((0, 0), [])), 0,
                           "Screen (0,0) should have at least one connection")


@unittest.skipUnless(_has_export_data(), "No SM export data at /tmp/sm_export/")
class TestParlorNavigation(unittest.TestCase):
    """Test the Parlor room (5x5 screens) - the key non-linear room."""

    @classmethod
    def setUpClass(cls):
        cls.room = load_room(ROOMS_DIR / "room_92FD.json")
        cls.nav = RoomNavigator(cls.room)

    def test_screen_dimensions(self):
        self.assertEqual(self.nav._w, 5)
        self.assertEqual(self.nav._h, 5)

    def test_door_to_climb(self):
        """Should find the down door to Climb."""
        pos = self.nav.find_door_position(0x96BA)
        self.assertIsNotNone(pos, "Should find door to Climb")
        x, y = pos
        # Down door should be near the bottom
        room_height = self.room.height_blocks * 16
        self.assertGreater(y, room_height * 0.5,
                           f"Climb door at y={y} should be in bottom half")

    def test_door_to_landing_site(self):
        """Should find the right door to Landing Site."""
        pos = self.nav.find_door_position(0x91F8)
        self.assertIsNotNone(pos, "Should find door to Landing Site")
        x, y = pos
        # Right door should be near the right edge
        room_width = self.room.width_blocks * 16
        self.assertGreater(x, room_width * 0.5,
                           f"Landing Site door at x={x} should be on right side")

    def test_descent_path_has_waypoints(self):
        """Path from right entry to bottom exit should have multiple waypoints."""
        # Enter from Landing Site (right door) and exit to Climb (bottom)
        right_door = self.nav.find_door_position(0x91F8)
        down_door = self.nav.find_door_position(0x96BA)
        self.assertIsNotNone(right_door)
        self.assertIsNotNone(down_door)

        waypoints = self.nav.screen_path(right_door, down_door)
        self.assertGreater(len(waypoints), 2,
                           "Parlor descent should need intermediate waypoints")

    def test_parlor_screen_adjacency(self):
        """Some screens should be connected, showing the room is navigable."""
        adj = self.nav._adj
        total_connections = sum(len(v) for v in adj.values())
        self.assertGreater(total_connections, 5,
                           "Parlor should have many screen connections")


@unittest.skipUnless(_has_export_data(), "No SM export data at /tmp/sm_export/")
class TestSimpleRoomNavigation(unittest.TestCase):
    """Test simple corridor rooms (Flyway, Pit Room)."""

    def test_flyway(self):
        room = load_room(ROOMS_DIR / "room_9879.json")
        nav = RoomNavigator(room)
        self.assertEqual(nav._w, 3)
        self.assertEqual(nav._h, 1)
        # Left to right should be a simple 3-screen path
        waypoints = nav.screen_path((20, 100), (700, 100))
        self.assertGreaterEqual(len(waypoints), 2)

    def test_pit_room(self):
        room = load_room(ROOMS_DIR / "room_975C.json")
        nav = RoomNavigator(room)
        self.assertEqual(nav._w, 3)
        self.assertEqual(nav._h, 2)


if __name__ == "__main__":
    unittest.main()
