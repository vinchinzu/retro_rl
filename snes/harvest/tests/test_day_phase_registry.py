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
from harvest.planner.day_phase_catalog import HOT_SPRING_STAMINA_PHASE
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
        self.assertEqual(establish.name, "pocket_plant_plot")
        self.assertIsInstance(water, CropWaterTask)
        self.assertEqual(water.work_mode, "water")

    def test_crop_builder_wires_catalog_water_params(self) -> None:
        """Catalog CROP_WATER must pass work_mode=water and north-stream refill bounds."""
        from harvest.planner.day_phase_catalog import CROP_ESTABLISH_PHASE, CROP_WATER_PHASE
        from harvest.tasks.crop_planter import CropWaterTask

        world = WorldState(frame=0, ram=np.zeros(0x24000, dtype=np.uint8), info={}, obs=None)
        establish = DayTaskFactory().make_task(CROP_ESTABLISH_PHASE, world)
        water = DayTaskFactory().make_task(CROP_WATER_PHASE, world)

        self.assertEqual(establish.name, "pocket_plant_plot")
        self.assertIsInstance(water, CropWaterTask)
        self.assertEqual(water.work_mode, "water")
        # North stream (y~16-22) + south pond; south-only left early west plants dry.
        self.assertEqual(water.refill_bounds, (3, 10, 62, 60))
        self.assertEqual(CROP_WATER_PHASE.params.get("refill_bounds"), (3, 10, 62, 60))
        self.assertEqual(CROP_WATER_PHASE.params.get("work_mode"), "water")

    def test_crop_builder_pocket_water_returns_water_skill(self) -> None:
        from harvest.planner.d2_work import pocket_water_phase
        from harvest.tasks.crop_planter import CropWaterTask
        from harvest.tasks.crop_skills import PLOT_RING_SIZE

        world = WorldState(frame=0, ram=np.zeros(0x24000, dtype=np.uint8), info={}, obs=None)
        pocket = DayTaskFactory().make_task(pocket_water_phase(), world)
        catalog = DayTaskFactory().make_task(
            PhaseSpec("CROP_WATER", PhaseKind.CROP, {"work_mode": "water"}),
            world,
        )
        self.assertEqual(pocket.name, "pocket_water_ring")
        self.assertNotIsInstance(pocket, CropWaterTask)
        self.assertEqual(
            sum(1 for t in pocket.tasks if t.name == "water_until_wet"),
            PLOT_RING_SIZE,
        )
        self.assertIsInstance(catalog, CropWaterTask)
        self.assertEqual(catalog.work_mode, "water")

    def test_hot_spring_builder(self) -> None:
        world = WorldState(frame=0, ram=np.zeros(0x24000, dtype=np.uint8), info={}, obs=None)
        task = DayTaskFactory().make_task(
            PhaseSpec("HOT_SPRING_STAMINA", PhaseKind.HOT_SPRING, {"min_stamina": 50}),
            world,
        )
        self.assertEqual(task.__class__.__name__, "HotSpringStaminaTask")
        self.assertEqual(task.min_stamina, 50)

    def test_hot_spring_catalog_fills_to_max(self) -> None:
        world = WorldState(frame=0, ram=np.zeros(0x24000, dtype=np.uint8), info={}, obs=None)
        task = DayTaskFactory().make_task(HOT_SPRING_STAMINA_PHASE, world)
        self.assertEqual(task.__class__.__name__, "HotSpringStaminaTask")
        self.assertIsNone(task.min_stamina)
        self.assertTrue(task.return_to_farm)


if __name__ == "__main__":
    unittest.main()
