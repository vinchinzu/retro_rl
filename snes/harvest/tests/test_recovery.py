from __future__ import annotations

import unittest
from types import SimpleNamespace

import numpy as np

from harvest.core.ram_catalog import LIVE_RAM_WRAM_OFFSET, field_spec
from harvest.core.recovery import RecoveryTask
from harvest.planner.day_plan import ActionResult, TaskResult, TaskStatus
from harvest.tasks.farm_clearer import make_action


def _live_base(ram: np.ndarray) -> int:
    return LIVE_RAM_WRAM_OFFSET if len(ram) > 0x20000 else 0


def _write_u16(ram: np.ndarray, addr: int, value: int) -> None:
    ram[addr] = value & 0xFF
    ram[addr + 1] = (value >> 8) & 0xFF


def _world(tilemap: int = 0x00):
    ram = np.zeros(0x24000, dtype=np.uint8)
    ram[field_spec("tilemap").address] = tilemap
    ram[field_spec("input_lock").address] = 1
    _write_u16(ram, field_spec("player_x").address + _live_base(ram), 136)
    _write_u16(ram, field_spec("player_y").address + _live_base(ram), 424)
    return SimpleNamespace(ram=ram, info={}, obs=None)


class RecoveryTaskTests(unittest.TestCase):
    def test_succeeds_after_stable_target_scene(self) -> None:
        world = _world(0x00)
        task = RecoveryTask(stable_frames=2)
        task.reset(world)

        first = task.step(world)
        second = task.step(world)

        self.assertEqual(first.status, TaskStatus.RUNNING)
        self.assertEqual(second.status, TaskStatus.SUCCESS)
        self.assertIn("normal_map@farm", second.reason)

    def test_dismisses_dialogue_before_stabilizing(self) -> None:
        world = _world(0x00)
        world.ram[field_spec("input_lock").address] = 0
        _write_u16(world.ram, field_spec("dialog_text_id").address, 0x03A6)
        task = RecoveryTask(stable_frames=1)
        task.reset(world)

        dialog = task.step(world)
        world.ram[field_spec("input_lock").address] = 1
        _write_u16(world.ram, field_spec("dialog_text_id").address, 0)
        recovered = task.step(world)

        self.assertEqual(dialog.status, TaskStatus.RUNNING)
        self.assertIsNotNone(dialog.action)
        self.assertEqual(recovered.status, TaskStatus.SUCCESS)

    def test_dismisses_dialogue_even_on_unknown_event_tilemap(self) -> None:
        world = _world(0xFE)
        world.ram[field_spec("input_lock").address] = 0
        _write_u16(world.ram, field_spec("dialog_text_id").address, 0x0315)
        task = RecoveryTask(stable_frames=1)
        task.reset(world)

        dialog = task.step(world)

        self.assertEqual(dialog.status, TaskStatus.RUNNING)
        self.assertIsNotNone(dialog.action)
        self.assertIn("recovering dialogue", dialog.reason)

        world.ram[field_spec("input_lock").address] = 1
        dialog = task.step(world)
        self.assertEqual(dialog.status, TaskStatus.RUNNING)
        self.assertIsNotNone(dialog.action)

    def test_waits_through_map_transition(self) -> None:
        world = _world(0x00)
        world.ram[field_spec("player_state").address + _live_base(world.ram)] = 0x80
        task = RecoveryTask(stable_frames=1)
        task.reset(world)

        waiting = task.step(world)
        world.ram[field_spec("player_state").address + _live_base(world.ram)] = 0
        recovered = task.step(world)

        self.assertEqual(waiting.status, TaskStatus.RUNNING)
        self.assertIn("map_transition", waiting.reason)
        self.assertEqual(recovered.status, TaskStatus.SUCCESS)

    def test_mashes_through_cutscene_unknown_tilemap_then_blocks(self) -> None:
        world = _world(0xFE)
        task = RecoveryTask(stable_frames=1, cutscene_mash_limit=2)
        task.reset(world)

        first = task.step(world)
        second = task.step(world)
        blocked = task.step(world)

        self.assertEqual(first.status, TaskStatus.RUNNING)
        self.assertIn("recovering cutscene_event", first.reason)
        self.assertEqual(second.status, TaskStatus.RUNNING)
        self.assertEqual(blocked.status, TaskStatus.BLOCKED)
        self.assertIn("cutscene did not clear", blocked.reason)

    def test_cutscene_mash_recovers_when_map_becomes_normal(self) -> None:
        world = _world(0xFE)
        task = RecoveryTask(stable_frames=1, cutscene_mash_limit=20)
        task.reset(world)

        mash = task.step(world)
        world.ram[field_spec("tilemap").address] = 0x00
        recovered = task.step(world)

        self.assertEqual(mash.status, TaskStatus.RUNNING)
        self.assertEqual(recovered.status, TaskStatus.SUCCESS)
        self.assertIn("normal_map@farm", recovered.reason)

    def test_routes_normal_non_target_scene_to_farm(self) -> None:
        world = _world(0x15)
        _write_u16(world.ram, field_spec("player_y").address + _live_base(world.ram), 120)

        class FakeRoute:
            def __init__(self) -> None:
                self.steps = 0

            def reset(self, world) -> None:
                return None

            def can_start(self, world) -> bool:
                return True

            def step(self, world) -> TaskResult:
                self.steps += 1
                if self.steps == 1:
                    return TaskResult(status=TaskStatus.RUNNING, action=ActionResult(make_action(right=True)))
                world.ram[field_spec("tilemap").address] = 0x00
                _write_u16(world.ram, field_spec("player_y").address + _live_base(world.ram), 424)
                return TaskResult(status=TaskStatus.SUCCESS, reason="arrived")

        task = RecoveryTask(stable_frames=1, route_to_target_factory=FakeRoute)
        task.reset(world)

        first = task.step(world)
        second = task.step(world)

        self.assertEqual(first.status, TaskStatus.RUNNING)
        self.assertEqual(second.status, TaskStatus.SUCCESS)
        self.assertIn("route recovered", second.reason)

    def test_blocks_normal_non_target_scene_without_route(self) -> None:
        world = _world(0x15)
        _write_u16(world.ram, field_spec("player_y").address + _live_base(world.ram), 120)
        task = RecoveryTask(stable_frames=1)
        task.reset(world)

        result = task.step(world)

        self.assertEqual(result.status, TaskStatus.BLOCKED)
        self.assertIn("not target location", result.reason)

    def test_can_target_any_normal_scene(self) -> None:
        world = _world(0x15)
        _write_u16(world.ram, field_spec("player_y").address + _live_base(world.ram), 120)
        task = RecoveryTask(target_location=None, stable_frames=1)
        task.reset(world)

        result = task.step(world)

        self.assertEqual(result.status, TaskStatus.SUCCESS)
        self.assertIn("normal_map@house", result.reason)


if __name__ == "__main__":
    unittest.main()
