"""Unit tests for reusable path segments and reactive mountain berry."""

from __future__ import annotations

from pathlib import Path
import sys
import unittest

_TESTS_DIR = Path(__file__).resolve().parent
if str(_TESTS_DIR) not in sys.path:
    sys.path.insert(0, str(_TESTS_DIR))

from harvest.core.ram_catalog import field_spec
from harvest.maps.map_config import ROUTES, SEGMENTS, compose_routes, find_landmark, segment_waypoints
from harvest.planner.day_phase_catalog import MOUNTAIN_BERRY_PHASE, PHASE_SEQUENCES
from harvest.planner.day_phase_types import PhaseKind
from harvest.core.tile_catalog import ADDR_INPUT_LOCK
from harvest.tasks.mountain_berry import (
    BERRY_NAV_SEGMENTS,
    TOWN_NAV_SEGMENTS,
    MountainBerryTask,
    face_toward_grape,
    first_remaining_segment,
    go_to_town_waypoints,
    is_mountain_forage,
    on_grape_pixel,
)
from retro_harness import TaskStatus

from day_plan_test_helpers import make_transition_world, set_player_pos

ADDR_HELD = field_spec("held_item").address


class PathSegmentTests(unittest.TestCase):
    def test_town_and_mountain_share_farm_to_path(self) -> None:
        shared = SEGMENTS["farm_to_path"]
        self.assertEqual(shared[0].tilemap, 0x00)
        self.assertTrue(shared[0].is_exit)
        self.assertEqual(shared[0].exit_direction, "left")
        self.assertEqual(shared[-1].tilemap, 0x0C)
        self.assertEqual(shared[-1].target_px, (132, 128))

        town = segment_waypoints(*TOWN_NAV_SEGMENTS)
        mountain = segment_waypoints("farm_to_path", "path_to_mountain")
        self.assertEqual(town[: len(shared)], shared)
        self.assertEqual(mountain[: len(shared)], shared)
        self.assertEqual(town, go_to_town_waypoints())
        self.assertEqual(town, ROUTES["farm_to_town"])
        self.assertEqual(town, ROUTES["go_to_town"])
        self.assertEqual(mountain, ROUTES["farm_to_mountain"])

    def test_path_to_town_and_mountain_diverge_at_crossroads(self) -> None:
        self.assertEqual(SEGMENTS["path_to_town"][-1].exit_direction, "left")
        self.assertEqual(SEGMENTS["path_to_mountain"][-1].exit_direction, "up")
        self.assertNotEqual(
            SEGMENTS["path_to_town"][-1].target_px,
            SEGMENTS["path_to_mountain"][-1].target_px,
        )

    def test_compose_routes_matches_named_full_route(self) -> None:
        composed = compose_routes(
            SEGMENTS["farm_to_path"],
            SEGMENTS["path_to_mountain"],
            SEGMENTS["mountain_entry_to_first_berry"],
        )
        self.assertEqual(composed, ROUTES["farm_to_first_mountain_berry"])
        self.assertIsNone(composed[-1].action_on_arrive)
        self.assertEqual(composed[-1].target_px, (326, 409))

    def test_first_berry_landmark_matches_route_stand(self) -> None:
        found = find_landmark("first_berry", tilemap_id=0x10)
        self.assertIsNotNone(found)
        _tm, landmark = found
        stand = ROUTES["mountain_entry_to_first_berry"][-1]
        self.assertEqual(landmark.tile, (20, 25))
        self.assertEqual(stand.target_px, (326, 409))
        hops = SEGMENTS["mountain_entry_to_first_berry"]
        # Cliff blocks due-north from land; recorded gap is east to x=32.
        self.assertEqual(hops[0].target_px, (328, 728))
        self.assertGreaterEqual(max(wp.target_px[0] for wp in hops), 500)
        self.assertEqual(hops[-1].target_px, (326, 409))

    def test_approach_only_succeeds_at_stand_without_held_grape(self) -> None:
        world = make_transition_world(0x10, current_tile=(20, 25))
        set_player_pos(world.ram, 326, 409)
        task = MountainBerryTask(approach_only=True, pick_attempts=0)
        task.reset(world)
        result = task.step(world)
        self.assertEqual(result.status, TaskStatus.SUCCESS)
        self.assertIn("unverified", result.reason or "")

    def test_face_toward_grape_from_northwest_stand(self) -> None:
        world = make_transition_world(0x10, current_tile=(19, 24))
        set_player_pos(world.ram, 316, 399)
        self.assertIn(face_toward_grape(world.ram), {"right", "down"})
        self.assertFalse(on_grape_pixel(world.ram))
        set_player_pos(world.ram, 326, 409)
        self.assertEqual(face_toward_grape(world.ram), "down")
        self.assertTrue(on_grape_pixel(world.ram))

    def test_pick_at_stand_faces_down_not_bush(self) -> None:
        world = make_transition_world(0x10, current_tile=(20, 25))
        set_player_pos(world.ram, 326, 409)
        task = MountainBerryTask(approach_only=False, pick_attempts=2)
        task.reset(world)
        result = task.step(world)
        self.assertEqual(result.status, TaskStatus.RUNNING)
        self.assertIn("pick", task.phase_text)
        self.assertIn("pick attempt=1", result.reason or "")

    def test_eat_box_with_held_grape_does_not_count_as_talk(self) -> None:
        world = make_transition_world(0x10, current_tile=(20, 25))
        set_player_pos(world.ram, 326, 409)
        world.ram[ADDR_HELD] = 0x03
        world.ram[ADDR_INPUT_LOCK] = 2
        task = MountainBerryTask(approach_only=False, pick_attempts=2)
        task.reset(world)
        task._picks = 1
        result = task.step(world)
        self.assertEqual(result.status, TaskStatus.RUNNING)
        self.assertNotIn("talked", result.reason or "")

    def test_mountain_dialogue_without_grape_is_talk_fail(self) -> None:
        world = make_transition_world(0x10, current_tile=(20, 25))
        set_player_pos(world.ram, 326, 409)
        world.ram[ADDR_INPUT_LOCK] = 2
        task = MountainBerryTask(approach_only=False, pick_attempts=2)
        task.reset(world)
        result = task.step(world)
        self.assertEqual(result.status, TaskStatus.FAILURE)
        self.assertIn("talked on mountain", result.reason or "")


class MountainBerrySelectTests(unittest.TestCase):
    def test_segment_choice_follows_live_tilemap(self) -> None:
        self.assertEqual(first_remaining_segment(0x00), "farm_to_path")
        self.assertEqual(first_remaining_segment(0x0C), "path_to_mountain")
        self.assertEqual(first_remaining_segment(0x10), "mountain_entry_to_first_berry")
        self.assertIsNone(first_remaining_segment(0x15))

    def test_already_holding_grapes_on_mountain_succeeds(self) -> None:
        world = make_transition_world(0x10, current_tile=(32, 43))
        world.ram[ADDR_HELD] = 0x03
        task = MountainBerryTask()
        task.reset(world)
        result = task.step(world)
        self.assertEqual(result.status, TaskStatus.SUCCESS)
        self.assertIn("grapes", result.reason or "")

    def test_house_start_arms_exit_to_farm(self) -> None:
        world = make_transition_world(0x15, current_tile=(8, 12))
        set_player_pos(world.ram, 136, 120)
        task = MountainBerryTask()
        task.reset(world)
        self.assertEqual(task.phase_text, "exit_to_farm")
        result = task.step(world)
        self.assertEqual(result.status, TaskStatus.RUNNING)

    def test_forage_ids_are_decomp_held_items(self) -> None:
        self.assertTrue(is_mountain_forage(0x03))
        self.assertTrue(is_mountain_forage(0x08))
        self.assertFalse(is_mountain_forage(0x0D))
        self.assertFalse(is_mountain_forage(0x00))

    def test_day_plan_sequence_is_registered(self) -> None:
        phases = PHASE_SEQUENCES["mountain_berry"]
        names = [p.phase for p in phases]
        self.assertIn("EXIT_TO_FARM", names)
        self.assertIn("MOUNTAIN_BERRY", names)
        self.assertEqual(MOUNTAIN_BERRY_PHASE.kind, PhaseKind.MOUNTAIN_BERRY)
        self.assertEqual(BERRY_NAV_SEGMENTS[0], "farm_to_path")


if __name__ == "__main__":
    unittest.main()
