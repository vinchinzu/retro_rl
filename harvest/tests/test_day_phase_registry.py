"""Tests for typed phase kinds and the task builder registry."""

from __future__ import annotations

import unittest
from unittest.mock import patch

import numpy as np

from harvest.planner.day_phase_registry import (
    PHASE_TASK_BUILDERS,
    TaskBuildContext,
    build_phase_task,
)
from harvest.planner.day_phase_types import PhaseKind, PhaseSpec, SKIP_MAP_LOCK_KINDS
from harvest.planner.day_plan_orchestrator import DayPlanTask
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

    def test_day_plan_reuses_factory_context_for_all_phases_in_one_day(self) -> None:
        world = WorldState(frame=0, ram=np.zeros(0x24000, dtype=np.uint8), info={}, obs=None)
        plan = DayPlanTask(phase_sequence=[])
        plan.reset(world)
        first_factory = plan._task_factory

        with patch(
            "harvest.planner.day_task_factory.build_phase_task",
            return_value=None,
        ) as build:
            plan._make_task(PhaseSpec("FIRST", PhaseKind.NAV), world)
            plan._make_task(PhaseSpec("SECOND", PhaseKind.NAV), world)

        contexts = [call.args[0].world_context for call in build.call_args_list]
        self.assertEqual(len(contexts), 2)
        self.assertIs(contexts[0], contexts[1])
        self.assertIs(plan._task_factory, first_factory)

        plan.reset(world)
        self.assertIsNot(plan._task_factory, first_factory)

    def test_ready_to_go_home_builder(self) -> None:
        spec = PhaseSpec("READY_TO_GO_HOME", PhaseKind.READY_TO_GO_HOME)
        world = WorldState(frame=0, ram=np.zeros(0x24000, dtype=np.uint8), info={}, obs=None)
        task = DayTaskFactory().make_task(spec, world)
        self.assertEqual(task.__class__.__name__, "ReadyToGoHomeTask")
        result = task.step(world)
        self.assertEqual(result.status.name, "SUCCESS")
        self.assertTrue(result.meta.get("ready_to_go_home"))

    def test_crop_builder_honors_work_mode(self) -> None:
        from harvest.tasks.crop_planter import CropWaterTask

        world = WorldState(frame=0, ram=np.zeros(0x24000, dtype=np.uint8), info={}, obs=None)
        establish = DayTaskFactory().make_task(
            PhaseSpec("CROP_ESTABLISH", PhaseKind.CROP, {"work_mode": "establish"}),
            world,
        )
        water = DayTaskFactory().make_task(
            PhaseSpec("CROP_WATER", PhaseKind.CROP, {"work_mode": "water"}),
            world,
        )
        self.assertIsInstance(establish, CropWaterTask)
        self.assertIsInstance(water, CropWaterTask)
        self.assertEqual(establish.work_mode, "establish")
        self.assertEqual(water.work_mode, "water")

    def test_hot_spring_builder(self) -> None:
        world = WorldState(frame=0, ram=np.zeros(0x24000, dtype=np.uint8), info={}, obs=None)
        task = DayTaskFactory().make_task(
            PhaseSpec("HOT_SPRING_STAMINA", PhaseKind.HOT_SPRING, {"min_stamina": 50}),
            world,
        )
        self.assertEqual(task.__class__.__name__, "HotSpringStaminaTask")
        self.assertEqual(task.min_stamina, 50)


if __name__ == "__main__":
    unittest.main()
