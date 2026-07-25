"""Tests for typed phase kinds and the task builder registry."""

from __future__ import annotations

import unittest

import numpy as np

from harvest.planner.day_phase_registry import (
    PHASE_TASK_BUILDERS,
    TaskBuildContext,
    build_phase_task,
)
from harvest.planner.day_phase_types import PhaseKind, PhaseSpec, SKIP_MAP_LOCK_KINDS
from harvest.planner.day_task_factory import DayTaskFactory
from retro_harness import WorldState


class DayPhaseRegistryTests(unittest.TestCase):
    def test_phase_kind_coerces_from_string(self) -> None:
        spec = PhaseSpec("NAV_CROP", "nav")
        self.assertEqual(spec.kind, PhaseKind.NAV)

    def test_unknown_kind_string_preserved_for_tests(self) -> None:
        spec = PhaseSpec("FIRST", "custom")
        self.assertEqual(spec.kind, "custom")

    def test_registry_covers_all_constructible_kinds(self) -> None:
        missing = [
            kind
            for kind in PhaseKind
            if kind not in PHASE_TASK_BUILDERS
            and kind != PhaseKind.DYNAMIC_OUTDOOR_PLAN
        ]
        self.assertEqual(missing, [])

    def test_skip_map_lock_kinds_are_registry_subset(self) -> None:
        self.assertTrue(SKIP_MAP_LOCK_KINDS.issubset(set(PHASE_TASK_BUILDERS)))

    def test_deadline_builder_returns_task(self) -> None:
        spec = PhaseSpec(
            "BUY_SEEDS_WINDOW",
            PhaseKind.DEADLINE,
            {"latest_hour": 7, "latest_minute": 0},
        )
        world = WorldState(frame=0, ram=np.zeros(0x24000, dtype=np.uint8), info={}, obs=None)
        task = build_phase_task(TaskBuildContext(), spec, world)
        self.assertIsNotNone(task)

    def test_factory_delegates_to_registry(self) -> None:
        spec = PhaseSpec("RETURN_HOME", PhaseKind.RETURN_HOME)
        world = WorldState(frame=0, ram=np.zeros(0x24000, dtype=np.uint8), info={}, obs=None)
        task = DayTaskFactory().make_task(spec, world)
        self.assertEqual(task.__class__.__name__, "ReturnHomeTask")


if __name__ == "__main__":
    unittest.main()
