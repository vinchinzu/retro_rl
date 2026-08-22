"""rr-20w.2.1: push-facing tile is travel non-walkable (measured player_action=0)."""
from __future__ import annotations

from pathlib import Path
import sys
from types import SimpleNamespace

_TESTS_DIR = Path(__file__).resolve().parent
if str(_TESTS_DIR) not in sys.path:
    sys.path.insert(0, str(_TESTS_DIR))
from day_plan_test_helpers import make_navigation_ram, set_player_pos

import unittest

from harvest.core.ram_catalog import field_spec
from harvest.core.tile_catalog import ADDR_INPUT_LOCK
from harvest.maps.map_config import Waypoint
from harvest.planner.day_plan import MultiMapNavTask, NavTask
from retro_harness.actions import action_names
from harvest.tasks.farm_clearer import TileScanner
from harvest.tasks.nav import Navigator, Pathfinder, Point, TILE_SIZE
from harvest.tasks.travel_walk import (
    PLAYER_ACTION_DIALOGUE,
    PLAYER_ACTION_IDLE,
    PLAYER_ACTION_JUMP,
    PLAYER_DIR_LEFT,
    PLAYER_DIR_UP,
    PUSH_HOLD_FRAMES,
    facing_tile,
    is_push_action,
    read_player_action,
    read_player_direction,
)

ADDR_PLAYER_ACTION = field_spec("player_action").address
ADDR_PLAYER_DIRECTION = field_spec("player_direction").address


def _charge_until_block(nav: Navigator, ram, facing, *, frames: int | None = None):
    blocked = False
    for _ in range(PUSH_HOLD_FRAMES + 3 if frames is None else frames):
        if nav.note_push_facing(ram, facing):
            blocked = True
            break
    return blocked


class TravelWalkPushFacingTests(unittest.TestCase):
    def test_measured_push_is_idle_byte_not_jump_or_dialogue(self) -> None:
        self.assertTrue(is_push_action(PLAYER_ACTION_IDLE))
        self.assertFalse(is_push_action(PLAYER_ACTION_JUMP))
        self.assertFalse(is_push_action(PLAYER_ACTION_DIALOGUE))
        self.assertEqual(facing_tile((13, 8), PLAYER_DIR_UP), (13, 7))
        self.assertEqual(facing_tile((13, 8), "left"), (12, 8))

    def test_pathfinder_blocks_push_facing_on_idle_not_jump(self) -> None:
        ram = make_navigation_ram(current_tile=(13, 8), blocked_tile=(63, 63))
        ram[ADDR_PLAYER_ACTION] = PLAYER_ACTION_IDLE
        pf = Pathfinder(TileScanner())
        facing = (14, 8)

        self.assertTrue(pf.is_walkable(ram, *facing))
        self.assertTrue(pf.block_push_facing(ram, facing, pixel_moved=False))
        self.assertFalse(pf.is_walkable(ram, *facing))
        self.assertIn(facing, pf.temp_blocked)

        pf.temp_blocked.clear()
        ram[ADDR_PLAYER_ACTION] = PLAYER_ACTION_JUMP
        self.assertFalse(pf.block_push_facing(ram, facing, pixel_moved=False))
        self.assertTrue(pf.is_walkable(ram, *facing))

        ram[ADDR_PLAYER_ACTION] = PLAYER_ACTION_IDLE
        self.assertFalse(pf.block_push_facing(ram, facing, pixel_moved=True))
        self.assertTrue(pf.is_walkable(ram, *facing))

    def test_bfs_skips_push_facing_tile(self) -> None:
        ram = make_navigation_ram(current_tile=(13, 8), blocked_tile=(63, 63))
        ram[ADDR_PLAYER_ACTION] = PLAYER_ACTION_IDLE
        pf = Pathfinder(TileScanner())
        pf.block_push_facing(ram, (14, 8), pixel_moved=False)

        path = pf.find_path(ram, (13, 8), (15, 8))
        self.assertIsNotNone(path)
        self.assertNotIn((14, 8), path)

    def test_navigator_marks_facing_after_hold_not_on_first_charge(self) -> None:
        ram = make_navigation_ram(current_tile=(13, 8), blocked_tile=(63, 63))
        ram[ADDR_PLAYER_ACTION] = PLAYER_ACTION_IDLE
        nav = Navigator(Pathfinder(TileScanner()))
        nav.update(ram)
        nav.path = [(14, 8)]

        first = nav.follow_path(ram)
        self.assertIsNotNone(first)
        self.assertNotIn((14, 8), nav.pathfinder.temp_blocked)

        # Same pixel, keep charging.
        later = None
        for _ in range(PUSH_HOLD_FRAMES + 1):
            later = nav.follow_path(ram)
        self.assertIn((14, 8), nav.pathfinder.temp_blocked)
        self.assertEqual(nav.path, [])
        self.assertIsNone(later)

    def test_navigator_does_not_block_when_pixels_move(self) -> None:
        ram = make_navigation_ram(current_tile=(13, 8), blocked_tile=(63, 63))
        ram[ADDR_PLAYER_ACTION] = PLAYER_ACTION_IDLE
        nav = Navigator(Pathfinder(TileScanner()))
        nav.update(ram)
        nav.path = [(14, 8)]
        nav.follow_path(ram)
        set_player_pos(ram, 13 * TILE_SIZE + 10, 8 * TILE_SIZE + 8)
        nav.update(ram)
        nav.path = [(14, 8)]
        nav.follow_path(ram)
        self.assertNotIn((14, 8), nav.pathfinder.temp_blocked)

    def test_multinav_safe_walk_refuses_push_facing(self) -> None:
        ram = make_navigation_ram(current_tile=(13, 8), blocked_tile=(63, 63))
        ram[ADDR_PLAYER_ACTION] = PLAYER_ACTION_IDLE
        world = SimpleNamespace(ram=ram, info={}, obs=None, frame=0)
        task = MultiMapNavTask(
            waypoints=[Waypoint(tilemap=0x00, target_px=(20 * 16 + 8, 8 * 16 + 8), radius=12)],
            timeout=200,
            initial_settle_frames=0,
        )
        task.reset(world)
        task._navigator.update(ram)
        facing = (14, 8)
        self.assertTrue(_charge_until_block(task._navigator, ram, facing))
        self.assertIn(facing, task._pathfinder.temp_blocked)
        self.assertTrue(task._tile_blocks_charge(ram, *facing))
        self.assertIsNone(task._safe_walk_action(ram, "right"))

    def test_navtask_fallback_idles_instead_of_b_charge_into_push(self) -> None:
        ram = make_navigation_ram(current_tile=(13, 8), blocked_tile=(63, 63))
        ram[ADDR_PLAYER_ACTION] = PLAYER_ACTION_IDLE
        world = SimpleNamespace(ram=ram, info={}, obs=None, frame=0)
        task = NavTask(target_px=Point(20 * 16 + 8, 8 * 16 + 8))
        task.reset(world)
        facing = (14, 8)
        self.assertTrue(_charge_until_block(task._navigator, ram, facing))
        action = task._fallback_action(ram)
        self.assertEqual(int(action[7]), 0)  # no right
        self.assertEqual(int(action[0]), 0)  # no B

    def test_navtask_leaves_shed_door_west_when_target_is_not_door(self) -> None:
        ram = make_navigation_ram(current_tile=(26, 30), blocked_tile=(63, 63))
        ram[ADDR_INPUT_LOCK] = 1
        world = SimpleNamespace(ram=ram, info={}, obs=None, frame=0)
        task = NavTask(target_px=Point(13 * 16 + 8, 29 * 16 + 8), radius=16)
        task.reset(world)
        names = set(action_names(task.step(world).action.action))
        self.assertTrue("LEFT" in names or "DOWN" in names)
        self.assertNotIn("UP", names)

    def test_navtask_fallback_does_not_enter_shed_door(self) -> None:
        ram = make_navigation_ram(current_tile=(26, 30), blocked_tile=(63, 63))
        world = SimpleNamespace(ram=ram, info={}, obs=None, frame=0)
        task = NavTask(target_px=Point(13 * 16 + 8, 29 * 16 + 8))
        task.reset(world)
        task._navigator.stasis = 45  # secondary would be UP into the shed
        names = set(action_names(task._fallback_action(ram)))
        self.assertNotIn("UP", names)

    def test_direction_reader(self) -> None:
        ram = make_navigation_ram()
        ram[ADDR_PLAYER_ACTION] = 0
        ram[ADDR_PLAYER_DIRECTION] = PLAYER_DIR_LEFT
        self.assertEqual(read_player_action(ram), 0)
        self.assertEqual(read_player_direction(ram), PLAYER_DIR_LEFT)


if __name__ == "__main__":
    unittest.main()
