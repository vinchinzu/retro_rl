"""Carry-to-pond policy tests independent of fence selection."""

from __future__ import annotations

from pathlib import Path
import sys
from types import SimpleNamespace
import unittest

_TESTS_DIR = Path(__file__).resolve().parent
if str(_TESTS_DIR) not in sys.path:
    sys.path.insert(0, str(_TESTS_DIR))

from water_refill_helpers import _blank_ram, _set_player_tile, _set_tile

from harvest.core.tile_catalog import ADDR_INPUT_LOCK, ADDR_TILEMAP, ADDR_X, ADDR_Y
from harvest.maps.farm_pond import (
    COW_BARN_EAST_FACE_TILES,
    EAST_SPUR_FA_STAND,
    HORSE_BARN_LEAVE_TILE,
    HORSE_BARN_WALL_TILES,
)
from harvest.tasks.carry_toss import (
    CarryToPondStand,
    farm_toss_stands,
)
from harvest.tasks.nav import VIEWPORT_HOP_TILES
from harvest.tasks.pond_policy import PRIMARY_POND_STAND
from retro_harness import TaskStatus
from retro_harness.actions import action_names


class CarryToPondStandTests(unittest.TestCase):
    def _world(self, player=(20, 20)):
        ram = _blank_ram()
        ram[ADDR_TILEMAP] = 0x00
        ram[ADDR_INPUT_LOCK] = 1
        for y in range(64):
            for x in range(64):
                _set_tile(ram, x, y, 0xA1)
        _set_player_tile(ram, player)
        ram[0xD2] = 0x02
        return SimpleNamespace(ram=ram, info={}, obs=None)

    def test_house_band_uses_east_spur_fa_not_sealed_f9(self) -> None:
        stands = farm_toss_stands((3, 13))
        self.assertEqual(stands, ((EAST_SPUR_FA_STAND, "up"),))

    def test_field_band_prefers_primary_f0_stand(self) -> None:
        self.assertEqual(farm_toss_stands((20, 31))[0][0], PRIMARY_POND_STAND)

    def test_path_never_overrides_live_boulder(self) -> None:
        world = self._world(player=(20, 31))
        for y in (31, 32):
            for x in (21, 22):
                _set_tile(world.ram, x, y, 0x0D)
        task = CarryToPondStand()
        task.reset(world)
        result = task.step(world)
        self.assertEqual(result.status, TaskStatus.RUNNING)
        self.assertNotIn((21, 31), task._navigator.path)
        self.assertNotIn((22, 31), task._navigator.path)
        self.assertNotIn((21, 32), task._navigator.path)
        self.assertNotIn((22, 32), task._navigator.path)

    def test_empty_hands_is_success(self) -> None:
        world = self._world()
        world.ram[0xD2] = 0
        task = CarryToPondStand()
        task.reset(world)
        self.assertEqual(task.step(world).status, TaskStatus.SUCCESS)

    def test_stand_toss_does_not_require_pixel_center(self) -> None:
        world = self._world(player=PRIMARY_POND_STAND)
        # Live south-lip landing is off-center because the water edge is solid.
        live_x, live_y = 518, 558
        world.ram[ADDR_X] = live_x & 0xFF
        world.ram[ADDR_X + 1] = live_x >> 8
        world.ram[ADDR_Y] = live_y & 0xFF
        world.ram[ADDR_Y + 1] = live_y >> 8
        task = CarryToPondStand()
        task.reset(world)
        result = task.step(world)
        self.assertEqual(task._navigator.current_tile, PRIMARY_POND_STAND)
        self.assertFalse(task._navigator.at_tile(PRIMARY_POND_STAND))
        self.assertEqual(result.status, TaskStatus.RUNNING)
        self.assertTrue(task._toss_started)
        self.assertIn("toss from pond stand", result.reason)
        queued = [set(action_names(action)) for action in task._actions]
        self.assertTrue(any(names == {"A"} for names in queued))
        self.assertFalse(any("A" in names and "UP" in names for names in queued))

    def test_completed_south_lip_toss_reaches_west_egress_before_success(self) -> None:
        world = self._world(player=PRIMARY_POND_STAND)
        _set_tile(world.ram, 31, 34, 0xA6)
        _set_tile(world.ram, 34, 34, 0xA6)
        task = CarryToPondStand()
        task.reset(world)

        toss = task.step(world)
        self.assertIn("toss from pond stand", toss.reason)
        task._actions.clear()
        world.ram[0xD2] = 0
        _set_player_tile(world.ram, (33, 34))

        result = task.step(world)
        self.assertEqual(result.status, TaskStatus.RUNNING)
        self.assertIn("egress boxed pond stand", result.reason)
        self.assertTrue(
            {"LEFT", "DOWN"}.intersection(action_names(result.action.action))
        )
        self.assertEqual(task._navigator.path[-1], (29, 35))

        _set_player_tile(world.ram, (33, 35))
        result = task.step(world)
        self.assertEqual(result.status, TaskStatus.RUNNING)
        self.assertIn("LEFT", action_names(result.action.action))
        self.assertEqual(task._navigator.path[-1], (29, 35))

        _set_player_tile(world.ram, (29, 35))
        self.assertEqual(task.step(world).status, TaskStatus.SUCCESS)

    def test_north_of_barn_dumps_at_east_spur_fa(self) -> None:
        self.assertEqual(farm_toss_stands((20, 12)), ((EAST_SPUR_FA_STAND, "up"),))
        self.assertEqual(farm_toss_stands((48, 13)), ((EAST_SPUR_FA_STAND, "up"),))

    def test_north_of_barn_uses_y14_then_x31_highway(self) -> None:
        world = self._world(player=(20, 12))
        task = CarryToPondStand()
        task.reset(world)
        result = task.step(world)
        self.assertEqual(result.status, TaskStatus.RUNNING)
        path = task._navigator.path
        self.assertIn((20, 14), path)
        self.assertIn((31, 14), path)
        self.assertIn(EAST_SPUR_FA_STAND, path)
        self.assertNotIn(PRIMARY_POND_STAND, path)
        self.assertLess(path.index((20, 14)), path.index((31, 14)))
        barn = HORSE_BARN_WALL_TILES | COW_BARN_EAST_FACE_TILES | {
            (x, y) for x in range(29, 31) for y in range(18, 22)
        }
        self.assertFalse(barn.intersection(path))
        prev = (20, 12)
        for tile in path:
            self.assertEqual(abs(tile[0] - prev[0]) + abs(tile[1] - prev[1]), 1)
            prev = tile

    def test_south_field_does_not_use_barn_east_corridor(self) -> None:
        world = self._world(player=(20, 31))
        task = CarryToPondStand()
        task.reset(world)
        result = task.step(world)
        self.assertEqual(result.status, TaskStatus.RUNNING)
        path = task._navigator.path
        self.assertNotIn((31, 14), path)
        self.assertFalse(any(y == 14 for _x, y in path))
        self.assertLessEqual(len(path), VIEWPORT_HOP_TILES)

    def test_x31_south_of_y16_still_uses_barn_east_corridor(self) -> None:
        world = self._world(player=(31, 20))
        task = CarryToPondStand()
        task.reset(world)
        result = task.step(world)
        self.assertEqual(result.status, TaskStatus.RUNNING)
        path = task._navigator.path
        self.assertIn((31, 14), path)
        self.assertIn(EAST_SPUR_FA_STAND, path)
        self.assertNotIn((31, 26), path)
        self.assertNotIn(PRIMARY_POND_STAND, path)
        barn = HORSE_BARN_WALL_TILES | COW_BARN_EAST_FACE_TILES | {
            (x, y) for x in range(29, 31) for y in range(18, 22)
        }
        self.assertFalse(barn.intersection(path))

    def test_horse_barn_takeoff_leaves_south_not_through_sprite(self) -> None:
        world = self._world(player=(17, 20))
        task = CarryToPondStand()
        task.reset(world)
        result = task.step(world)
        self.assertEqual(result.status, TaskStatus.RUNNING)
        path = task._navigator.path
        self.assertTrue(path)
        self.assertEqual(path[0], HORSE_BARN_LEAVE_TILE)
        self.assertIn(EAST_SPUR_FA_STAND, path)
        self.assertFalse(HORSE_BARN_WALL_TILES.intersection(path))
        self.assertFalse(COW_BARN_EAST_FACE_TILES.intersection(path))
        self.assertNotEqual(path[0], (18, 20))
        self.assertNotEqual(path[0], (16, 20))

    def test_ne_carry_goes_around_live_boulder_not_through_it(self) -> None:
        world = self._world(player=(39, 5))
        _set_tile(world.ram, 38, 6, 0x0D)
        _set_tile(world.ram, 39, 6, 0x0E)
        task = CarryToPondStand()
        task.reset(world)
        result = task.step(world)
        self.assertEqual(result.status, TaskStatus.RUNNING)
        path = task._navigator.path
        boulder = {(38, 6), (39, 6), (38, 7), (39, 7)}
        self.assertFalse(boulder.intersection(path))
        self.assertTrue(path)
        self.assertNotEqual(path[0], (39, 6))

    def test_east_bypass_skips_pond_edge_a6(self) -> None:
        world = self._world(player=(20, 12))
        _set_tile(world.ram, 35, 31, 0xA6)
        _set_tile(world.ram, 31, 34, 0xA6)
        _set_tile(world.ram, 34, 34, 0xA6)
        task = CarryToPondStand()
        task.reset(world)
        result = task.step(world)
        self.assertEqual(result.status, TaskStatus.RUNNING)
        path = task._navigator.path
        self.assertNotIn((35, 31), path)
        self.assertIn((20, 14), path)
        self.assertIn((31, 14), path)
        self.assertIn(EAST_SPUR_FA_STAND, path)


if __name__ == "__main__":
    unittest.main()
