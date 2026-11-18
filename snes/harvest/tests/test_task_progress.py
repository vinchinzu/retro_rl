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
    WorkFrameLedger,
    atomic_write_json,
    build_run_progress_record,
    checkpoint_progress_key,
    format_run_progress_line,
    safe_to_checkpoint,
    stall_progress_key,
    stalled,
    task_progress_snapshot,
    work_snapshot,
    work_receipt,
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
    def test_required_failure_records_work_before_recovery_discards_the_child(self):
        from unittest.mock import patch
        from harvest.planner.day_phase_catalog import CROP_WATER_PHASE
        from retro_harness import TaskResult, TaskStatus
        ram = _progress_ram(day=30, hour=12, tilemap=0x15)
        day = DayPlanTask(phase_sequence=[CROP_WATER_PHASE])
        day._phase_start_frame = 10
        day._current_task = SimpleNamespace(watered_count=0, skipped_water=1)
        world = SimpleNamespace(frame=30, ram=ram)
        with patch.object(day, "_start_recovery", return_value=TaskResult(status=TaskStatus.RUNNING)):
            day._handle_failed_phase(CROP_WATER_PHASE, TaskStatus.FAILURE, "dry crop remains", world)
        self.assertEqual(day.phase_results[0]["status"], "failure")
        self.assertEqual(day.phase_results[0]["work"]["counts"]["skipped_water"], 1)
        self.assertEqual(day.phase_results[0]["frames"], 20)

    def test_day_floor_preserves_inflight_work_counters_and_elapsed_frames(self):
        from harvest.planner.day_phase_catalog import CROP_WATER_PHASE
        ram = _progress_ram(day=9, hour=18, tilemap=0x15)
        day = DayPlanTask(phase_sequence=[CROP_WATER_PHASE])
        day._phase_start_frame = 10
        day._current_task = SimpleNamespace(watered_count=4, skipped_water=2)
        planner = MultiDayPlannerTask()
        planner._current_task = day
        planner._capture_day_plan_outcomes(SimpleNamespace(frame=30, ram=ram))
        planner._capture_day_plan_outcomes()
        self.assertEqual(len(planner._last_day_phase_results), 1)
        row = planner._last_day_phase_results[0]
        self.assertEqual((row["status"], row["frames"]), ("interrupted", 20))
        self.assertEqual(row["work"]["counts"], {"watered_count": 4, "skipped_water": 2})

    def test_work_ledger_counts_only_actions_and_preserves_target_visits(self):
        from harvest.tasks.crop_skills import water_until_wet_skill
        ram = _progress_ram(day=4, hour=8, tilemap=0x15, stamina=80)
        original = ram.copy()
        leaf = water_until_wet_skill(target_tile=(12, 27))
        composer = DayPlanTask(phase_sequence=[])
        composer._current_task = leaf
        ledger = WorkFrameLedger()
        ledger.observe(frame=100, ram=ram, task=composer)
        ledger.observe(frame=101, ram=ram, task=composer)
        leaf.target_tile = (13, 27)
        ledger.observe(frame=102, ram=ram, task=composer)
        row = ledger.report(ram)
        self.assertEqual(row["frames"], 3)
        self.assertEqual([s["frames"] for s in row["segments"]], [2, 1])
        self.assertEqual(row["segments"][-1]["end_frame"], 103)
        self.assertEqual(row["segments"][-1]["context"][-1][3], [13, 27])
        np.testing.assert_array_equal(ram, original)

    def test_work_ledger_idle_counts_dead_runs_and_ignores_pressed_or_moving_frames(self):
        from harvest.core.task_progress import DEAD_RUN_MIN_FRAMES
        from harvest.tasks.nav import make_action
        composer = DayPlanTask(phase_sequence=[])
        ledger = WorkFrameLedger()
        still = _progress_ram(day=4, tilemap=0x15, player_x=100, player_y=200)
        moved = _progress_ram(day=4, tilemap=0x15, player_x=101, player_y=200)
        frame = 0
        for ram, action in (
            [(still, make_action())] * (DEAD_RUN_MIN_FRAMES + 5)   # waiting: dead
            + [(still, make_action(A=1))] * 10                      # button held, not moving: busy
            + [(moved, make_action())] * (DEAD_RUN_MIN_FRAMES - 7)  # 1 moving + 112 dead: too short to list
            + [(moved, make_action(right=1))]
        ):
            ledger.observe(frame=frame, ram=ram, task=composer, action=action)
            frame += 1
        idle = ledger.report(moved)["idle"]
        self.assertEqual(idle["input_frames"], 11)
        self.assertEqual(idle["idle_frames"], frame - 11)
        runs = idle["dead_runs"]
        self.assertEqual(len(runs), 1)
        # The first frame has no earlier pose to equal, so the run starts on frame 1.
        self.assertEqual((runs[0]["start_frame"], runs[0]["frames"]), (1, DEAD_RUN_MIN_FRAMES + 4))
        self.assertEqual(ledger.segments[0]["input_frames"], 11)

    def test_work_ledger_without_actions_reports_no_idle_section(self):
        ledger = WorkFrameLedger()
        ledger.observe(frame=0, ram=_progress_ram(day=4), task=DayPlanTask(phase_sequence=[]))
        self.assertNotIn("idle", ledger.report(_progress_ram(day=4)))

    def test_failed_partial_work_keeps_counts_and_indoor_gate_is_unknown(self):
        before = _progress_ram(tilemap=0x15, stamina=80, watering_can=10)
        after = _progress_ram(tilemap=0x15, stamina=72, watering_can=6)
        receipt = work_receipt(work_snapshot(before), after,
                               SimpleNamespace(watered_count=4, skipped_water=1))
        self.assertEqual(receipt["counts"], {"watered_count": 4, "skipped_water": 1})
        self.assertEqual(receipt["delta"]["watering_can"], -4)
        self.assertIsNone(receipt["after"]["gates"]["all_growing_crops_wet"])
        self.assertIsNone(receipt["after"]["crops"])

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


def test_scorecard_idle_section_lists_dead_runs_and_is_absent_without_data():
    from harvest.tools import final_score

    assert final_score.render_idle({"work_metrics": {}}) == []
    report = {"work_metrics": {"idle": {
        "input_frames": 900, "idle_frames": 100, "dead_frames": 60, "dead_run_min_frames": 120,
        "dead_runs": [{"start_frame": 5000, "frames": 700, "phase": "SLEEP", "date": [1, 2, 9], "tilemap": 21}],
    }}}
    text = "\n".join(final_score.render_idle(report))
    assert "| 1,000 | 900 | 100 | 60 | 1 | 700 | 700 |" in text
    assert "| SLEEP | 1 | 700 |" in text
    assert "700 frames (11.7 s) in SLEEP, Y2 Fa D9, map 0x15, from frame 5,000" in text
