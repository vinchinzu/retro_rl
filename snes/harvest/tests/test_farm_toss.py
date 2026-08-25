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
    STONE,
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
    evaluate_lift_verify,
    fence_jump_action,
    nearest_pocket_drop,
    needs_south_fence_drop,
    open_toss_face,
    pocket_no_toss_tiles,
    toss_pulse_action,
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

    def test_fence_jump_skill_carries_east_when_plot_blocks_toss(self) -> None:
        world = _world(tile=(12, 29), held=HELD_STONE)
        skill = FenceJumpTossSkill()
        skill.reset(world)
        result = skill.step(world)
        self.assertEqual(result.status, TaskStatus.RUNNING)
        names = set(action_names(result.action.action))
        # Hoe stands + 3x3 + y=30 lip eat every adjacent face. Carry east.
        self.assertIn("RIGHT", names)
        self.assertIn("B", names)
        self.assertNotIn("UP", names)
        world.ram[ADDR_HELD_ITEM] = 0
        done = skill.step(world)
        self.assertEqual(done.status, TaskStatus.SUCCESS)

    def test_open_toss_skips_3x3_plot_and_hoe_stands(self) -> None:
        world = _world(tile=(12, 29), held=HELD_STONE)
        self.assertIsNone(open_toss_face(world.ram, (12, 29)))
        self.assertIn((11, 28), pocket_no_toss_tiles())
        self.assertIn((12, 28), pocket_no_toss_tiles())

    def test_open_toss_does_not_relend_on_lift_origin(self) -> None:
        world = _world(tile=(11, 29), held=HELD_STONE)
        _set_tile(world, (10, 29), 0xA6)
        self.assertNotEqual(open_toss_face(world.ram, (11, 29)), "up")
        self.assertIsNone(open_toss_face(world.ram, (11, 29), blocked={(11, 28)}))

    def test_pond_lip_carry_east_not_toss_north(self) -> None:
        world = _world(tile=(11, 29), held=HELD_STONE)
        _set_tile(world, (10, 29), 0xA6)
        skill = FenceJumpTossSkill(blocked=frozenset({(11, 28)}))
        skill.reset(world)
        result = skill.step(world)
        names = set(action_names(result.action.action))
        self.assertIn("RIGHT", names)
        self.assertIn("B", names)
        self.assertNotIn("UP", names)
        self.assertNotIn("A", names)

    def test_toss_pulse_does_not_hold_face_during_throw(self) -> None:
        face_tap = set(action_names(toss_pulse_action(0, face="up")))
        self.assertEqual(face_tap, {"UP"})
        settle = set(action_names(toss_pulse_action(4, face="up")))
        self.assertEqual(settle, set())
        throw = set(action_names(toss_pulse_action(12, face="up")))
        self.assertEqual(throw, {"A"})
        self.assertNotIn("UP", throw)

    def test_evaluate_lift_verify_carrying_is_not_cleared(self) -> None:
        world = _world(tile=(11, 29), held=HELD_STONE)
        _set_tile(world, (11, 28), 0x01)
        self.assertEqual(evaluate_lift_verify(world.ram, (11, 28)), "carrying")
        world.ram[ADDR_HELD_ITEM] = 0
        self.assertEqual(evaluate_lift_verify(world.ram, (11, 28)), "cleared")
        _set_tile(world, (11, 28), STONE)
        self.assertEqual(evaluate_lift_verify(world.ram, (11, 28)), "blocked")

    def test_bush_pin_does_not_toss_onto_hoe_stand(self) -> None:
        world = _world(tile=(13, 27), held=HELD_WEED)
        _set_tile(world, (13, 28), WEED)
        _set_tile(world, (14, 27), WEED)
        self.assertNotEqual(open_toss_face(world.ram, (13, 27)), "up")

        skill = FenceJumpTossSkill()
        skill.reset(world)
        first = skill.step(world)
        names = set(action_names(first.action.action))
        self.assertNotIn("UP", names)
        self.assertNotIn("A", names)


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
