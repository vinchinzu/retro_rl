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
from harvest.tasks.carry_toss import (
    CarryToPondStand,
    farm_toss_stands,
)
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

    def test_house_band_uses_verified_main_pond_not_sealed_f9(self) -> None:
        stands = farm_toss_stands((3, 13))
        self.assertEqual(stands, ((PRIMARY_POND_STAND, "up"),))

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


if __name__ == "__main__":
    unittest.main()
