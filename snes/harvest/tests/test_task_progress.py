"""Tests for task progress snapshots."""

from __future__ import annotations

import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace

import numpy as np

from harvest.core.ram_catalog import LIVE_RAM_WRAM_OFFSET, field_spec
from harvest.core.task_progress import (
    GOAL_STALL_FRAMES,
    MOTION_STALL_FRAMES,
    ProgressSnapshot,
    RunProgressSidecar,
    atomic_write_json,
    build_run_progress_record,
    checkpoint_progress_key,
    format_run_progress_line,
    safe_to_checkpoint,
    stall_progress_key,
    stalled,
    task_progress_snapshot,
)
from harvest.planner.day_plan_orchestrator import DayPlanTask, MultiDayPlannerTask


class TaskProgressTests(unittest.TestCase):
    def test_day_plan_task_progress_snapshot_uses_public_accessors(self) -> None:
        task = DayPlanTask(phase_sequence=[])
        task._phase_index = 0
        task._step_count = 3
        task._current_task = SimpleNamespace(
            progress_snapshot=lambda: ProgressSnapshot(
                task_name="NavTask",
                phase_text="nav",
                details=(("target", (1, 2)),),
            )
        )

        snap = task.progress_snapshot()
        self.assertEqual(snap.task_name, "DayPlanTask")
        self.assertEqual(snap.phase_index, 0)
        self.assertEqual(snap.step_count, 3)
        self.assertIsNotNone(snap.child)
        self.assertEqual(snap.child.task_name, "NavTask")

    def test_multi_day_planner_progress_snapshot_includes_days_completed(self) -> None:
        task = MultiDayPlannerTask()
        task._phase = "plan_day"
        task._days_completed = 2
        task._step_count = 5

        snap = task.progress_snapshot()
        self.assertEqual(snap.task_name, "MultiDayPlannerTask")
        self.assertEqual(snap.phase_text, "PLAN_DAY")
        self.assertEqual(snap.step_count, 5)
        self.assertIn(("days_completed", 2), snap.details)

    def test_task_progress_snapshot_falls_back_without_method(self) -> None:
        leaf = SimpleNamespace(
            phase_text="water",
            phase_index=2,
            step_count=10,
        )
        snap = task_progress_snapshot(leaf)
        self.assertIsNotNone(snap)
        assert snap is not None
        self.assertEqual(snap.task_name, "SimpleNamespace")
        self.assertEqual(snap.phase_text, "water")
        self.assertEqual(snap.phase_index, 2)

    def test_signature_ignores_step_count_field(self) -> None:
        base = dict(
            task_name="DayPlanTask",
            phase_text="CLEAR",
            phase_index=1,
            details=(("cleared", 4),),
        )
        first = ProgressSnapshot(step_count=3, **base)
        second = ProgressSnapshot(step_count=300, **base)
        self.assertEqual(first.signature(), second.signature())
        self.assertNotEqual(first.step_count, second.step_count)

    def test_day_plan_step_count_ticks_do_not_change_signature(self) -> None:
        task = DayPlanTask(phase_sequence=[])
        task._phase_index = 0
        task._current_task = SimpleNamespace(
            progress_snapshot=lambda: ProgressSnapshot(
                task_name="NavTask",
                phase_text="nav",
                details=(("target", (1, 2)),),
            )
        )
        task._step_count = 3
        first = task.progress_snapshot().signature()
        task._step_count = 300
        second = task.progress_snapshot().signature()
        self.assertEqual(first, second)
        self.assertEqual(task.progress_snapshot().step_count, 300)

    def test_signature_ignores_step_count_detail_pair(self) -> None:
        first = ProgressSnapshot(
            task_name="Leaf",
            details=(("target", (1, 2)), ("step_count", 3)),
        )
        second = ProgressSnapshot(
            task_name="Leaf",
            details=(("target", (1, 2)), ("step_count", 300)),
        )
        self.assertEqual(first.signature(), second.signature())

    def test_signature_includes_target_detail(self) -> None:
        first = ProgressSnapshot(task_name="Leaf", details=(("target", (1, 2)),))
        second = ProgressSnapshot(task_name="Leaf", details=(("target", (3, 4)),))
        self.assertNotEqual(first.signature(), second.signature())

    def test_motion_stall_window(self) -> None:
        self.assertTrue(stalled(0, 360, MOTION_STALL_FRAMES))
        self.assertFalse(stalled(0, 359, MOTION_STALL_FRAMES))

    def test_goal_stall_window(self) -> None:
        self.assertTrue(stalled(0, 24000, GOAL_STALL_FRAMES))

    def test_to_record_is_json_safe_and_keeps_step_count(self) -> None:
        snap = ProgressSnapshot(
            task_name="FarmClearTask",
            phase_text="clearing",
            step_count=12,
            details=(("target", (34, 42)), ("hits", 3)),
            child=ProgressSnapshot(task_name="NavTask", details=(("approach", (33, 42)),)),
        )
        row = snap.to_record()
        encoded = json.dumps(row)
        self.assertIn("FarmClearTask", encoded)
        self.assertEqual(row["step_count"], 12)
        self.assertEqual(row["details"]["target"], [34, 42])
        self.assertEqual(row["child"]["task_name"], "NavTask")


def _progress_ram(**fields: int) -> np.ndarray:
    ram = np.zeros(0x20000, dtype=np.uint8)
    for key, value in fields.items():
        addr = field_spec(key).address
        width = field_spec(key).width
        for base in (0, LIVE_RAM_WRAM_OFFSET):
            idx = addr + base
            ram[idx] = value & 0xFF
            if width == 2 and idx + 1 < len(ram):
                ram[idx + 1] = (value >> 8) & 0xFF
    return ram


class _FakeD2:
    def __init__(self, **values) -> None:
        self.stamina = SimpleNamespace(current=40, maximum=100)
        self.animating = False
        self.input_stable = True
        self._row = {
            "weeds": 0,
            "fences": 0,
            "stones": 0,
            "small_rocks": 0,
            "large_rocks": 0,
            "trees_or_stumps": 5,
            "planted": 8,
            "wet": 8,
            "outcome": "work_remaining",
            "input_stable": True,
            "animating": False,
            **values,
        }

    def to_record(self) -> dict:
        return dict(self._row)


class RunProgressSidecarTests(unittest.TestCase):
    def test_atomic_write_replaces_a_complete_document(self) -> None:
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "sidecar.json"
            atomic_write_json(path, {"frame": 1, "ok": True})
            first = json.loads(path.read_text(encoding="utf-8"))
            atomic_write_json(path, {"frame": 2, "ok": True, "nested": {"a": [1, 2]}})
            second = json.loads(path.read_text(encoding="utf-8"))
            self.assertEqual(first["frame"], 1)
            self.assertEqual(second["frame"], 2)
            self.assertFalse(path.with_name(path.name + ".tmp").exists())

    def test_record_carries_date_map_pose_phase_d2_and_watchdogs(self) -> None:
        ram = _progress_ram(
            season=0,
            day=2,
            hour=18,
            minute=12,
            money=100,
            player_x=536,
            player_y=679,
            stamina=90,
            max_stamina=100,
            tilemap=0,
            input_lock=1,
        )
        tactic = SimpleNamespace(
            phase_text="CLEAR_STUMPS",
            progress_text="chunk=se",
            _step=900,
            _motion_at=880,
            _goal_at=100,
            _unobs=0,
            current_task=SimpleNamespace(
                progress_snapshot=lambda: ProgressSnapshot(
                    task_name="FarmClearTask",
                    phase_text="clearing",
                    details=(("target", (34, 42)), ("hits", 4)),
                )
            ),
        )
        record = build_run_progress_record(
            frame=12000,
            planner_frames=8000,
            wall_seconds=12.34,
            ram=ram,
            task=tactic,
            d2_status=_FakeD2(),
        )
        self.assertEqual(record["date"]["day"], 2)
        self.assertEqual(record["date"]["hour"], 18)
        self.assertEqual(record["map"]["tilemap"], 0)
        self.assertEqual(record["position"]["tile"], [33, 42])
        self.assertEqual(record["phase"], "CLEAR_STUMPS")
        self.assertEqual(record["debris"]["trees_or_stumps"], 5)
        self.assertEqual(record["stamina"]["current"], 90)
        self.assertEqual(record["d2"]["stamina"]["current"], 40)
        self.assertEqual(record["watchdog"]["task"], "SimpleNamespace")
        self.assertEqual(record["watchdog"]["motion_age"], 20)
        self.assertEqual(record["watchdog"]["goal_age"], 800)
        self.assertEqual(record["task"]["child"]["details"]["hits"], 4)
        line = format_run_progress_line({**record, "stall": {"semantic_age": 0}})
        self.assertIn("CLEAR_STUMPS", line)
        self.assertIn("u5", line)

    def test_clock_ticks_are_not_semantic_progress(self) -> None:
        base = {
            "date": {"season": 0, "day": 2, "hour": 18, "minute": 1, "money": 100},
            "phase": "CLEAR_STUMPS",
            "debris": {
                "weeds": 0,
                "fences": 0,
                "stones": 0,
                "large_rocks": 0,
                "trees_or_stumps": 5,
            },
            "d2": {"planted": 8, "wet": 8, "outcome": "work_remaining"},
            "task": {
                "phase_text": "CLEAR_STUMPS",
                "details": {"target": [34, 42]},
                "child": {"phase_text": "clearing", "details": {"hits": 4}},
            },
        }
        later = dict(base)
        later["date"] = dict(base["date"], hour=19, minute=40)
        self.assertEqual(stall_progress_key(base), stall_progress_key(later))
        self.assertEqual(checkpoint_progress_key(base), checkpoint_progress_key(later))
        moved = dict(base)
        moved["task"] = {
            "phase_text": "CLEAR_STUMPS",
            "details": {"target": [34, 42]},
            "child": {"phase_text": "clearing", "details": {"hits": 5}},
        }
        self.assertNotEqual(stall_progress_key(base), stall_progress_key(moved))
        self.assertEqual(checkpoint_progress_key(base), checkpoint_progress_key(moved))

    def test_checkpoint_waits_for_input_stable_then_saves(self) -> None:
        ram = _progress_ram(
            season=0,
            day=2,
            hour=18,
            minute=0,
            stamina=90,
            max_stamina=100,
            tilemap=0,
            input_lock=1,
            player_x=16,
            player_y=16,
        )
        task = SimpleNamespace(phase_text="CLEAR_STUMPS", progress_text="")
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            sidecar = RunProgressSidecar(
                root / "progress.json",
                checkpoint_dir=root / "pins",
                checkpoint_on_progress=True,
                keep=2,
            )
            saved: list[Path] = []

            def save_state(path: Path) -> Path:
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(b"state")
                saved.append(path)
                return path

            busy = _FakeD2(trees_or_stumps=5, input_stable=False, animating=True)
            busy.animating = True
            busy.input_stable = False
            first = sidecar.write(
                frame=100,
                planner_frames=100,
                wall_seconds=1.0,
                ram=ram,
                task=task,
                d2_status=busy,
                save_state=save_state,
            )
            self.assertTrue(sidecar.pending_checkpoint)
            self.assertIsNone(first["checkpoint"])
            self.assertFalse(safe_to_checkpoint(first))
            self.assertEqual(saved, [])

            settled = _FakeD2(trees_or_stumps=5, input_stable=True, animating=False)
            second = sidecar.write(
                frame=140,
                planner_frames=140,
                wall_seconds=1.5,
                ram=ram,
                task=task,
                d2_status=settled,
                save_state=save_state,
            )
            self.assertIsNotNone(second["checkpoint"])
            self.assertTrue((root / "pins" / "latest.state").is_file())
            self.assertEqual(json.loads((root / "progress.json").read_text())["frame"], 140)

            sidecar.write(
                frame=200,
                planner_frames=200,
                wall_seconds=2.0,
                ram=ram,
                task=task,
                d2_status=_FakeD2(trees_or_stumps=4),
                save_state=save_state,
            )
            sidecar.write(
                frame=300,
                planner_frames=300,
                wall_seconds=3.0,
                ram=ram,
                task=task,
                d2_status=_FakeD2(trees_or_stumps=3),
                save_state=save_state,
            )
            named = list((root / "pins").glob("progress_f*.state"))
            self.assertEqual(len(named), 2)
            self.assertTrue((root / "pins" / "progress_f300.state").is_file())
            self.assertFalse((root / "pins" / "progress_f140.state").exists())


if __name__ == "__main__":
    unittest.main()
