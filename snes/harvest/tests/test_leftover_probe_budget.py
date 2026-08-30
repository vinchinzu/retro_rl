"""Leftover spa spends remaining frames; budget RUNNING becomes FAILURE."""

from __future__ import annotations

import unittest
from pathlib import Path
from types import SimpleNamespace

from retro_harness import TaskResult, TaskStatus

from harvest.core.stamina import Stamina
from harvest.planner.d2_work import (
    bush_clear_phase,
    ensure_hammer_phase,
    fence_dump_phase,
    rock_clear_phase,
    stump_clear_phase,
)
from harvest.planner.day_phase_stamina import full_restore_spa_phase
from harvest.scripts.leftover_exec import (
    _phase_timeout,
    _phase_timeout_result,
    leftover_chain_decision,
    leftover_stall_should_abort,
    _task_phase_key,
)


class LeftoverPhaseTimeoutTests(unittest.TestCase):
    def test_spa_phase_timeout_spends_remaining_not_12k_estimate(self) -> None:
        remaining = 80_000
        spec = full_restore_spa_phase()
        self.assertEqual(spec.contract.estimated_frames, 12_000)
        self.assertNotIn("timeout", spec.params or {})
        self.assertEqual(_phase_timeout(spec, remaining), remaining)

    def test_spa_kind_without_phase_name_still_spends_remaining(self) -> None:
        spec = SimpleNamespace(
            phase="SPA_OTHER",
            kind="hot_spring",
            params={},
            contract=SimpleNamespace(estimated_frames=12_000),
        )
        self.assertEqual(_phase_timeout(spec, 80_000), 80_000)

    def test_exhaustive_smash_timeout_zero_spends_remaining(self) -> None:
        remaining = 200_000
        self.assertEqual(_phase_timeout(bush_clear_phase(), remaining), remaining)
        self.assertEqual(_phase_timeout(fence_dump_phase(), remaining), remaining)
        self.assertEqual(_phase_timeout(rock_clear_phase(), remaining), remaining)
        self.assertEqual(_phase_timeout(stump_clear_phase(), remaining), remaining)
        self.assertEqual(_phase_timeout(stump_clear_phase(), 50_000), 50_000)

    def test_non_spa_estimated_phase_caps_at_estimate(self) -> None:
        remaining = 80_000
        spec = ensure_hammer_phase()
        estimated = int(spec.contract.estimated_frames)
        self.assertNotIn("timeout", spec.params or {})
        self.assertGreater(remaining, estimated)
        self.assertEqual(_phase_timeout(spec, remaining), estimated)
        self.assertEqual(_phase_timeout(spec, 100), 100)


class LeftoverProbeDefaultTimeoutTests(unittest.TestCase):
    def test_leftover_probe_argparse_default_timeout_is_two_million(self) -> None:
        src = (
            Path(__file__).resolve().parents[1]
            / "harvest"
            / "scripts"
            / "d2_leftover_probe.py"
        ).read_text(encoding="utf-8")
        self.assertIn("default=2_000_000", src)
        self.assertNotIn("default=400_000", src)
        self.assertIn("default=24_000", src)


class LeftoverBudgetResultTests(unittest.TestCase):
    def test_running_or_none_at_budget_becomes_failure(self) -> None:
        running = TaskResult(status=TaskStatus.RUNNING)
        failed = _phase_timeout_result(running, 12_000)
        self.assertEqual(failed.status, TaskStatus.FAILURE)
        self.assertEqual(failed.reason, "phase timeout 12000f")
        none_failed = _phase_timeout_result(None, 80_000)
        self.assertEqual(none_failed.status, TaskStatus.FAILURE)
        self.assertEqual(none_failed.reason, "phase timeout 80000f")

    def test_terminal_status_is_left_alone(self) -> None:
        success = TaskResult(status=TaskStatus.SUCCESS, reason="soaked")
        self.assertIs(_phase_timeout_result(success, 80_000), success)
        stall = TaskResult(
            status=TaskStatus.FAILURE,
            reason="no debris progress 24000f (last_progress=1000)",
        )
        self.assertIs(_phase_timeout_result(stall, 80_000), stall)

    def test_spa_child_does_not_count_as_debris_stall(self) -> None:
        spa = SimpleNamespace(
            _spec=SimpleNamespace(phase="HOT_SPRING_STAMINA", kind="hot_spring", params={}),
            current_task=SimpleNamespace(name="hot_spring_stamina"),
            name="d2_farm_clear",
        )
        smash = SimpleNamespace(
            _spec=SimpleNamespace(
                phase="CLEAR_STUMPS", kind="clear_field", params={"chunk": "se"}
            ),
            current_task=SimpleNamespace(name="farm_clear"),
            name="d2_farm_clear",
        )
        idle = SimpleNamespace(
            _spec=None,
            current_task=None,
            name="d2_farm_clear",
        )
        self.assertFalse(leftover_stall_should_abort(spa, 24_000, 0, 24_000))
        self.assertTrue(leftover_stall_should_abort(smash, 24_000, 0, 24_000))
        self.assertTrue(leftover_stall_should_abort(idle, 24_000, 0, 24_000))
        self.assertEqual(_task_phase_key(spa), ("HOT_SPRING_STAMINA", None))
        self.assertEqual(_task_phase_key(smash), ("CLEAR_STUMPS", "se"))
        self.assertNotEqual(_task_phase_key(spa), _task_phase_key(smash))

    def test_spa_failure_aborts_not_spa_retry(self) -> None:
        self.assertEqual(
            leftover_chain_decision(
                "HOT_SPRING_STAMINA",
                TaskStatus.FAILURE,
                "phase timeout 80000f",
                Stamina(current=4, maximum=100),
                ("ENSURE_HAMMER", "CLEAR_ROCKS"),
            ),
            "abort",
        )
        self.assertEqual(
            leftover_chain_decision(
                "CLEAR_ROCKS",
                TaskStatus.FAILURE,
                "stamina_low cleared=2",
                Stamina(current=8, maximum=100),
                ("ENSURE_AXE", "CLEAR_STUMPS"),
            ),
            "spa_retry",
        )


if __name__ == "__main__":
    unittest.main()
