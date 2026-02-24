"""Tests for map data loading and parsing.

Requires /tmp/sm_export/ with actual exported JSON data.
"""

import unittest
from pathlib import Path

from super_metroid_rl.navigation.map_data import (
    DEFAULT_EXPORT_DIR,
    TILE_DOOR,
    load_nav_graph,
    load_room,
    load_world,
)

EXPORT_DIR = DEFAULT_EXPORT_DIR
ROOMS_DIR = EXPORT_DIR / "rooms"


def _has_export_data() -> bool:
    return EXPORT_DIR.exists() and (EXPORT_DIR / "nav_graph.json").exists()


@unittest.skipUnless(_has_export_data(), "No SM export data at /tmp/sm_export/")
class TestNavGraphLoading(unittest.TestCase):
    def test_load_nav_graph(self):
        nodes, edges = load_nav_graph(EXPORT_DIR / "nav_graph.json")
        self.assertEqual(len(nodes), 262)
        self.assertEqual(len(edges), 516)

    def test_nodes_have_required_fields(self):
        nodes, _ = load_nav_graph(EXPORT_DIR / "nav_graph.json")
        for node in nodes:
            self.assertIsInstance(node.room_id, int)
            self.assertGreater(node.room_id, 0)
            self.assertIsInstance(node.name, str)
            self.assertGreater(node.width_screens, 0)
            self.assertGreater(node.height_screens, 0)

    def test_edges_reference_valid_rooms(self):
        nodes, edges = load_nav_graph(EXPORT_DIR / "nav_graph.json")
        room_ids = {n.room_id for n in nodes}
        for edge in edges:
            self.assertIn(edge.from_room_id, room_ids,
                          f"Edge from unknown room 0x{edge.from_room_id:04X}")
            self.assertIn(edge.to_room_id, room_ids,
                          f"Edge to unknown room 0x{edge.to_room_id:04X}")

    def test_landing_site_node(self):
        nodes, _ = load_nav_graph(EXPORT_DIR / "nav_graph.json")
        landing = next((n for n in nodes if n.room_id == 0x91F8), None)
        self.assertIsNotNone(landing)
        self.assertEqual(landing.name, "Landing Site")
        self.assertEqual(landing.width_screens, 9)
        self.assertEqual(landing.height_screens, 5)


@unittest.skipUnless(_has_export_data(), "No SM export data at /tmp/sm_export/")
class TestRoomLoading(unittest.TestCase):
    def test_load_landing_site(self):
        room = load_room(ROOMS_DIR / "room_91F8.json")
        self.assertEqual(room.room_id, 0x91F8)
        self.assertEqual(room.name, "Landing Site")
        self.assertEqual(room.width_screens, 9)
        self.assertEqual(room.height_screens, 5)
        self.assertEqual(room.width_blocks, 144)
        self.assertEqual(room.height_blocks, 80)

    def test_collision_grid_dimensions(self):
        room = load_room(ROOMS_DIR / "room_91F8.json")
        self.assertEqual(len(room.collision), room.height_blocks)
        for row in room.collision:
            self.assertEqual(len(row), room.width_blocks)

    def test_landing_site_has_doors(self):
        room = load_room(ROOMS_DIR / "room_91F8.json")
        self.assertGreaterEqual(len(room.doors), 2)  # at least Left and Right exits

    def test_door_pixel_positions_within_room(self):
        room = load_room(ROOMS_DIR / "room_91F8.json")
        max_x = room.width_blocks * 16
        max_y = room.height_blocks * 16
        for door in room.doors:
            self.assertGreater(door.pixel_x, 0)
            self.assertGreater(door.pixel_y, 0)
            self.assertLessEqual(door.pixel_x, max_x)
            self.assertLessEqual(door.pixel_y, max_y)

    def test_parlor_has_3_doors(self):
        room = load_room(ROOMS_DIR / "room_92FD.json")
        self.assertEqual(room.room_id, 0x92FD)
        self.assertEqual(room.name, "Parlor and Alcatraz")
        self.assertEqual(len(room.doors), 3)

    def test_parlor_doors_have_correct_destinations(self):
        room = load_room(ROOMS_DIR / "room_92FD.json")
        dest_ids = {d.dest_room_id for d in room.doors}
        # Parlor connects to: Terminator (0x990D), Landing Site (0x91F8), Climb (0x96BA)
        self.assertIn(0x91F8, dest_ids, "Missing door to Landing Site")
        self.assertIn(0x96BA, dest_ids, "Missing door to Climb")

    def test_parlor_down_door_position(self):
        """The door going down to Climb should be near the bottom of the room."""
        room = load_room(ROOMS_DIR / "room_92FD.json")
        climb_door = next((d for d in room.doors if d.dest_room_id == 0x96BA), None)
        self.assertIsNotNone(climb_door)
        room_height_px = room.height_blocks * 16
        # Door should be in the bottom half
        self.assertGreater(climb_door.pixel_y, room_height_px * 0.5,
                           f"Down door at y={climb_door.pixel_y} should be in bottom half "
                           f"(room height={room_height_px})")


@unittest.skipUnless(_has_export_data(), "No SM export data at /tmp/sm_export/")
class TestWorldLoading(unittest.TestCase):
    def test_load_world(self):
        world = load_world()
        self.assertGreater(len(world.nodes), 200)
        self.assertGreater(len(world.edges), 400)
        self.assertGreater(len(world.rooms), 100)

    def test_route_rooms_loaded(self):
        """All rooms on the speedrun route should be loaded."""
        world = load_world()
        route_rooms = [0x91F8, 0x92FD, 0x96BA, 0x975C, 0x97B5, 0x9E9F, 0x9879, 0x9804]
        for room_id in route_rooms:
            self.assertIn(room_id, world.rooms,
                          f"Route room 0x{room_id:04X} not loaded")

    def test_door_collision_blocks_detected(self):
        """Every room with doors in JSON should have door positions detected."""
        world = load_world()
        for room_id in [0x91F8, 0x92FD, 0x96BA, 0x9879]:
            room = world.rooms[room_id]
            self.assertGreater(len(room.doors), 0,
                               f"Room 0x{room_id:04X} ({room.name}) has no doors detected")


if __name__ == "__main__":
    unittest.main()
