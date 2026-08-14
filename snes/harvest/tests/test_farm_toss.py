"""Reactive pocket toss + carry-pair tool swap (d2_farm_plant)."""

from __future__ import annotations

import unittest

import numpy as np

from harvest.core.carry import ADDR_TOOL_BACKPACK
from harvest.core.tile_catalog import (
    ADDR_MAP,
    ADDR_TOOL,
    ADDR_X,
    ADDR_Y,
    MAP_WIDTH,
    TILE_SIZE,
    WEED,
    Tool,
)
from harvest.core.animal_status import ADDR_HELD_ITEM
from harvest.tasks.farm_ops import ToolManager
from harvest.tasks.farm_toss import (
    FENCE_WALL_Y,
    HELD_STONE,
    HELD_WEED,
    FenceJumpTossSkill,
    fence_jump_action,
    nearest_pocket_drop,
    needs_south_fence_drop,
    open_toss_face,
)
from retro_harness import TaskStatus, WorldState
from retro_harness.actions import action_names


def _world(*, tile=(12, 29), held=HELD_STONE) -> WorldState:
    ram = np.zeros(0x20000, dtype=np.uint8)
    px = tile[0] * TILE_SIZE + 8
    py = tile[1] * TILE_SIZE + 8
    ram[ADDR_X] = px & 0xFF
    ram[ADDR_X + 1] = (px >> 8) & 0xFF
    ram[ADDR_Y] = py & 0xFF
    ram[ADDR_Y + 1] = (py >> 8) & 0xFF
    ram[ADDR_HELD_ITEM] = held
    for index in range(MAP_WIDTH * MAP_WIDTH):
        ram[ADDR_MAP + index] = 0x01
    return WorldState(frame=0, ram=ram, info={}, obs=None)


def _set_tile(world: WorldState, tile: tuple[int, int], tile_id: int) -> None:
    world.ram[ADDR_MAP + tile[1] * MAP_WIDTH + tile[0]] = tile_id


class PocketTossTests(unittest.TestCase):
    def test_stone_north_of_fence_needs_south_drop(self) -> None:
        self.assertTrue(needs_south_fence_drop((12, 29), HELD_STONE))
        self.assertTrue(needs_south_fence_drop((13, 30), HELD_WEED))
        self.assertFalse(needs_south_fence_drop((15, 32), HELD_STONE))
        self.assertFalse(needs_south_fence_drop((12, 29), 0))

    def test_drop_tiles_are_south_of_fence(self) -> None:
        self.assertEqual(nearest_pocket_drop((12, 29))[1], 32)
        self.assertGreater(nearest_pocket_drop((15, 29))[1], FENCE_WALL_Y)

    def test_fence_jump_is_straight_south(self) -> None:
        """No east align — B+Down from wherever we lifted."""
        for tile in ((12, 29), (15, 29), (18, 28), (5, 28)):
            action = fence_jump_action(tile, HELD_STONE)
            self.assertIsNotNone(action)
            names = set(action_names(action))
            self.assertIn("DOWN", names)
            self.assertIn("B", names)
            self.assertNotIn("RIGHT", names)
            self.assertNotIn("LEFT", names)
        self.assertIsNone(fence_jump_action((15, 32), HELD_STONE))
        self.assertIsNone(fence_jump_action((12, 29), 0))

    def test_fence_jump_skill_runs_south_then_clears(self) -> None:
        world = _world(tile=(12, 29), held=HELD_STONE)
        skill = FenceJumpTossSkill()
        skill.reset(world)
        result = skill.step(world)
        self.assertEqual(result.status, TaskStatus.RUNNING)
        self.assertIn("UP", set(action_names(result.action.action)))
        world.ram[ADDR_HELD_ITEM] = 0
        done = skill.step(world)
        self.assertEqual(done.status, TaskStatus.SUCCESS)

    def test_bush_pin_tosses_north_when_south_and_east_are_weeds(self) -> None:
        world = _world(tile=(13, 27), held=HELD_WEED)
        _set_tile(world, (13, 28), WEED)
        _set_tile(world, (14, 27), WEED)
        self.assertEqual(open_toss_face(world.ram, (13, 27)), "up")

        skill = FenceJumpTossSkill()
        skill.reset(world)
        actions = [skill.step(world).action.action for _ in range(12)]
        names = [set(action_names(action)) for action in actions]
        self.assertTrue(all("DOWN" not in pressed for pressed in names))
        self.assertTrue(any({"UP", "A"} <= pressed for pressed in names))


class ToolManagerCarryTests(unittest.TestCase):
    def test_spent_bag_swaps_to_backpack_can(self) -> None:
        ram = np.zeros(0x1000, dtype=np.uint8)
        ram[ADDR_TOOL] = 0
        ram[ADDR_TOOL_BACKPACK] = int(Tool.WATERING_CAN)
        mgr = ToolManager()
        mgr.update(ram)
        self.assertEqual(mgr.current, 0)
        self.assertTrue(mgr.has(int(Tool.WATERING_CAN)))
        self.assertTrue(mgr.needs_swap(int(Tool.WATERING_CAN)))
        self.assertFalse(mgr.needs_swap(int(Tool.HOE)))


class FarmClearerTossPolicyTests(unittest.TestCase):
    def test_held_stone_in_pocket_is_fence_jump(self) -> None:
        from harvest.tasks.farm_toss import held_toss_actions

        policy, frames = held_toss_actions((12, 29), HELD_STONE)
        self.assertEqual(policy, "pocket_south")
        names = set(action_names(frames[0]))
        self.assertIn("DOWN", names)
        self.assertIn("B", names)


if __name__ == "__main__":
    unittest.main()
