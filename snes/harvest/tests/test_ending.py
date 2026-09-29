"""Ending detection and the EndingTask wait (ROM facts: docs/ram_map.md)."""

from __future__ import annotations

import unittest

import numpy as np

from retro_harness import TaskStatus, WorldState

from harvest.core.ram_catalog import field_spec, ram_index
from harvest.core.scene import SceneMode, classify_scene_from_ram, scene_indicates_ending
from harvest.planner.tasks.ending import (
    ENDING_TERMINAL_INDEX,
    EndingTask,
    is_ending_date,
    read_ending_signals,
)


def _ram(**fields: int) -> np.ndarray:
    ram = np.zeros(0x24000, dtype=np.uint8)
    _set(ram, input_lock=1)
    _set(ram, **fields)
    return ram


def _set(ram: np.ndarray, **fields: int) -> None:
    for name, value in fields.items():
        spec = field_spec(name)
        ram[ram_index(ram, spec.address, live_offset=spec.live_offset)] = value & 0xFF


def _world(ram: np.ndarray, frame: int = 0) -> WorldState:
    return WorldState(frame, ram, {}, np.zeros((1, 1, 3), dtype=np.uint8))


class EndingDetectionTests(unittest.TestCase):
    def test_no_ending_in_normal_play(self) -> None:
        ram = _ram()
        sig = read_ending_signals(ram)
        self.assertFalse(sig.started)
        self.assertFalse(sig.terminal)
        self.assertFalse(scene_indicates_ending(classify_scene_from_ram(ram)))

    def test_stepper_marks_ending_scene(self) -> None:
        ram = _ram(ending_scene_index=0x14)
        self.assertTrue(read_ending_signals(ram).started)
        scene = classify_scene_from_ram(ram)
        self.assertEqual(scene.mode, SceneMode.ENDING_CREDITS)
        self.assertTrue(scene_indicates_ending(scene))

    def test_terminal_index(self) -> None:
        self.assertFalse(read_ending_signals(_ram(ending_scene_index=0x24)).terminal)
        self.assertTrue(read_ending_signals(_ram(ending_scene_index=ENDING_TERMINAL_INDEX)).terminal)

    def test_ending_date_is_year3_summer_d30(self) -> None:
        self.assertTrue(is_ending_date(2, 1, 30))
        for other in ((2, 1, 29), (1, 1, 30), (2, 2, 30), (2, 0, 30), (0, 1, 30)):
            self.assertFalse(is_ending_date(*other))


class EndingTaskTests(unittest.TestCase):
    def _task(self, **kw) -> EndingTask:
        task = EndingTask(idle_frames=20, **kw)
        task.reset(_world(_ram()))
        return task

    def test_waits_with_neutral_input_until_static_terminal(self) -> None:
        task = self._task()
        ram = _ram(ending_scene_index=0x14)
        for frame in range(1, 50):
            res = task.step(_world(ram, frame))
            self.assertEqual(res.status, TaskStatus.RUNNING)
            self.assertFalse(np.any(res.action.action))
        _set(ram, ending_scene_index=ENDING_TERMINAL_INDEX)
        statuses = [task.step(_world(ram, 50 + i)).status for i in range(25)]
        self.assertEqual(statuses[0], TaskStatus.RUNNING)
        self.assertEqual(statuses[-1], TaskStatus.SUCCESS)
        self.assertEqual(task.peak_index, ENDING_TERMINAL_INDEX)

    def test_dialogue_change_restarts_the_idle_count(self) -> None:
        task = self._task()
        ram = _ram(ending_scene_index=ENDING_TERMINAL_INDEX, dialog_text_id=1092 & 0xFF)
        for i in range(15):
            self.assertEqual(task.step(_world(ram, i)).status, TaskStatus.RUNNING)
        _set(ram, dialog_text_id=(1093 & 0xFF))
        for i in range(15):
            self.assertEqual(task.step(_world(ram, 15 + i)).status, TaskStatus.RUNNING)
        results = [task.step(_world(ram, 30 + i)).status for i in range(10)]
        self.assertEqual(results[-1], TaskStatus.SUCCESS)

    def test_not_terminal_never_succeeds(self) -> None:
        task = self._task()
        ram = _ram(ending_scene_index=0x20)
        for i in range(200):
            self.assertNotEqual(task.step(_world(ram, i)).status, TaskStatus.SUCCESS)

    def test_fails_when_ending_never_starts(self) -> None:
        task = self._task(start_timeout_frames=30)
        ram = _ram()
        last = None
        for i in range(40):
            last = task.step(_world(ram, i))
        self.assertEqual(last.status, TaskStatus.FAILURE)
        self.assertIn("never started", last.reason)

    def test_stalled_below_terminal_pulses_a(self) -> None:
        task = self._task(stall_pulse_frames=10)
        ram = _ram(ending_scene_index=0x14)
        pressed = [
            bool(np.any(task.step(_world(ram, i)).action.action)) for i in range(40)
        ]
        self.assertTrue(any(pressed[12:]))
        self.assertFalse(any(pressed[:10]))


if __name__ == "__main__":
    unittest.main()
