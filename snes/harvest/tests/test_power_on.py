from __future__ import annotations

import unittest

import numpy as np

from retro_harness import TaskStatus, WorldState

from harvest.core.ram_catalog import field_spec
from harvest.runtime.power_on import PowerOnStartTask
from harvest.tasks.nav import make_action


def make_world(*, tilemap: int, input_lock: int, day: int = 0, hour: int = 0) -> WorldState:
    ram = np.zeros(0x20000, dtype=np.uint8)
    ram[field_spec("tilemap").address] = tilemap
    ram[field_spec("input_lock").address] = input_lock
    ram[field_spec("day").address] = day
    ram[field_spec("hour").address] = hour
    # A valid normal-map position is needed for the final scene classifier.
    ram[field_spec("player_x").address] = 32
    ram[field_spec("player_y").address] = 32
    return WorldState(frame=0, ram=ram, info={}, obs=None)


class PowerOnStartTaskTests(unittest.TestCase):
    def test_title_selects_start_before_confirming(self) -> None:
        task = PowerOnStartTask()
        world = make_world(tilemap=0x5C, input_lock=4)
        world.ram[0x95] = 5
        world.ram[0x98D] = 2
        task.reset(world)
        task._boot_skip_sent = True
        task._step_count = 700

        result = task.step(world)

        self.assertEqual(result.status, TaskStatus.RUNNING)
        np.testing.assert_array_equal(result.action.action, make_action(up=True))
        self.assertFalse(task._title_confirmed)

        world.ram[0x98D] = 1
        task._step_count += 100
        result = task.step(world)

        np.testing.assert_array_equal(result.action.action, make_action(a=True))
        self.assertTrue(task._title_confirmed)

    def test_name_grid_uses_physical_left_for_logical_right(self) -> None:
        task = PowerOnStartTask()
        world = make_world(tilemap=0x5F, input_lock=5, day=1)
        world.ram[0x994] = 4
        task.reset(world)
        task._boot_skip_sent = True
        task._step_count = 700

        world.ram[0x991] = 0
        result = task.step(world)
        np.testing.assert_array_equal(result.action.action, make_action(left=True))

        task._step_count += 100
        world.ram[0x991] = 40
        result = task.step(world)
        np.testing.assert_array_equal(result.action.action, make_action(up=True))

        task._step_count += 100
        world.ram[0x991] = 70
        result = task.step(world)
        np.testing.assert_array_equal(result.action.action, make_action(a=True))
        self.assertTrue(task._name_submitted)

    def test_day_one_is_not_ready_until_name_is_submitted_and_settled(self) -> None:
        task = PowerOnStartTask()
        world = make_world(tilemap=0x04, input_lock=1, day=1, hour=7)
        task.reset(world)
        task._boot_skip_sent = True
        task._step_count = 700

        self.assertEqual(task.step(world).status, TaskStatus.RUNNING)

        task._name_submitted = True
        for _ in range(119):
            self.assertEqual(task.step(world).status, TaskStatus.RUNNING)
        self.assertEqual(task.step(world).status, TaskStatus.SUCCESS)
