"""Tests for typed phase kinds and the task builder registry."""

from __future__ import annotations

import unittest
from unittest.mock import patch

import numpy as np
import pytest

from tests.crop_planter_test_helpers import set_stamina
from harvest.planner.day_phase_registry import (
    PHASE_TASK_BUILDERS,
    TaskBuildContext,
    build_phase_task,
)
from harvest.planner.day_phase_catalog import HOT_SPRING_STAMINA_PHASE
from harvest.planner.day_phase_types import PhaseKind, PhaseSpec, SKIP_MAP_LOCK_KINDS
from harvest.planner.day_plan_orchestrator import DayPlanTask
from retro_harness import WorldState


@pytest.mark.rom
@pytest.mark.parametrize("state", ["LN_marriage_ann_offer", "marriage_feather_probe"])
def test_native_ann_proposal_from_offer_and_morning(state):
    from retro_harness import AuditedEnv, AuditCapabilities, TaskStatus
    from harvest.runtime.retro_setup import make_harvest_env
    from harvest.planner.day_phase_registry import ProposalTask
    from harvest.core.ram_catalog import read_ram_value
    from harvest.tasks.nav import make_action

    env = AuditedEnv(make_harvest_env(state, lean_step=True), capabilities=AuditCapabilities(
        provider="harvest.proposal_rom_test", observes_ram_writes=True,
        observes_mid_run_loads=True, observes_assists=True))
    try:
        env.reset()
        assert not read_ram_value(env.get_ram(), "marriage_flags") & 31
        task = ProposalTask()
        task.reset(WorldState(frame=0, ram=env.get_ram(), info={}, obs=None))
        for frame in range(12000):
            result = task.step(WorldState(frame=frame, ram=env.get_ram(), info={}, obs=None))
            if result.status != TaskStatus.RUNNING:
                break
            env.step(result.action.action if result.action else make_action())
        assert result.status == TaskStatus.SUCCESS, result.reason
        assert read_ram_value(env.get_ram(), "marriage_flags") & 31 == 2
        assert env.audit().ram_writes == env.audit().mid_run_loads == 0
    finally:
        env.close()


@pytest.mark.rom
@pytest.mark.parametrize("state", ["Y1_D3_Morning", "LN_marriage_court_natural_D3"])
def test_native_ann_courting_keeps_two_grapes_and_returns_to_farm(state):
    from retro_harness import AuditedEnv, AuditCapabilities, TaskStatus
    from harvest.runtime.retro_setup import make_harvest_env
    from harvest.tasks.proposal import AnnCourtTask
    from harvest.core.ram_catalog import read_ram_value
    from harvest.tasks.nav import make_action

    env = AuditedEnv(make_harvest_env(state, lean_step=True), capabilities=AuditCapabilities(
        provider="harvest.court_rom_test", observes_ram_writes=True,
        observes_mid_run_loads=True, observes_assists=True))
    try:
        env.reset()
        start = read_ram_value(env.get_ram(), "ann_hearts")
        task = AnnCourtTask()
        task.reset(WorldState(frame=0, ram=env.get_ram(), info={}, obs=None))
        for frame in range(16001):
            result = task.step(WorldState(frame=frame, ram=env.get_ram(), info={}, obs=None))
            if result.status != TaskStatus.RUNNING:
                break
            env.step(result.action.action if result.action else make_action())
        assert result.status == TaskStatus.SUCCESS, result.reason
        assert read_ram_value(env.get_ram(), "ann_hearts") == start + 12
        assert read_ram_value(env.get_ram(), "tilemap") == 0
        assert env.audit().ram_writes == env.audit().mid_run_loads == 0
    finally:
        env.close()


@pytest.mark.rom
@pytest.mark.parametrize("state,tilemap", [("LN_marriage_court_poweron_candidate_FAIL", 0x22),
                                          ("LN_marriage_court_poweron_after_room_fix_FAIL", 0x23)])
def test_tool_shop_exit_from_natural_power_on_failure(state, tilemap):
    from retro_harness import AuditedEnv, AuditCapabilities, TaskStatus
    from harvest.runtime.retro_setup import make_harvest_env
    from harvest.planner.tasks.inventory import ExitToFarmTask
    from harvest.core.ram_catalog import read_ram_value
    from harvest.tasks.nav import make_action

    env = AuditedEnv(make_harvest_env(state, lean_step=True),
                     capabilities=AuditCapabilities(provider="harvest.shop_exit_rom_test",
                         observes_ram_writes=True, observes_mid_run_loads=True, observes_assists=True))
    try:
        env.reset()
        assert read_ram_value(env.get_ram(), "tilemap") == tilemap
        task = ExitToFarmTask()
        task.reset(WorldState(frame=0, ram=env.get_ram(), info={}, obs=None))
        for frame in range(8000):
            result = task.step(WorldState(frame=frame, ram=env.get_ram(), info={}, obs=None))
            if result.status != TaskStatus.RUNNING:
                break
            env.step(result.action.action if result.action else make_action())
        assert result.status == TaskStatus.SUCCESS, result.reason
        assert read_ram_value(env.get_ram(), "tilemap") == 0
        assert env.audit().ram_writes == env.audit().mid_run_loads == 0
    finally:
        env.close()


class DayPhaseRegistryTests(unittest.TestCase):
    def test_courting_preserves_opening_and_peddler_requires_native_eligibility(self):
        from harvest.core.ram_catalog import field_spec
        from harvest.tasks.proposal import marriage_phases_for_ram

        ram = np.zeros(0x20000, dtype=np.uint8)
        def put(name, value):
            spec = field_spec(name)
            value //= spec.display_multiplier
            ram[spec.address] = value & 255
            if spec.width == 2:
                ram[spec.address + 1] = value >> 8

        put("hour", 6)
        put("weekday", 2)
        put("day", 2)
        self.assertEqual(marriage_phases_for_ram(ram), [])
        put("day", 3)
        self.assertEqual([p.phase for p in marriage_phases_for_ram(ram)], ["COURT_ANN"])
        put("weekday", 0)
        put("ann_hearts", 250)
        put("money", 1000)
        self.assertEqual([p.phase for p in marriage_phases_for_ram(ram)], ["BUY_FEATHER"])
        put("money", 990)
        self.assertEqual(marriage_phases_for_ram(ram), [])
        put("money", 1000)
        put("family_event_flags", 0x40)
        self.assertEqual(marriage_phases_for_ram(ram), [])

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
        task = build_phase_task(TaskBuildContext(), spec, world)
        self.assertEqual(task.__class__.__name__, "ReturnHomeTask")

    def test_day_plan_reuses_world_context_for_all_phases_in_one_day(self) -> None:
        world = WorldState(frame=0, ram=np.zeros(0x24000, dtype=np.uint8), info={}, obs=None)
        plan = DayPlanTask(phase_sequence=[])
        plan.reset(world)
        first_context = plan._world_context

        with patch(
            "harvest.planner.day_plan_orchestrator.build_phase_task",
            return_value=None,
        ) as build:
            plan._make_task(PhaseSpec("FIRST", PhaseKind.NAV), world)
            plan._make_task(PhaseSpec("SECOND", PhaseKind.NAV), world)

        contexts = [call.args[0].world_context for call in build.call_args_list]
        self.assertEqual(len(contexts), 2)
        self.assertIs(contexts[0], contexts[1])
        self.assertIs(plan._world_context, first_context)

        plan.reset(world)
        self.assertIsNot(plan._world_context, first_context)

    def test_ready_to_go_home_builder(self) -> None:
        spec = PhaseSpec("READY_TO_GO_HOME", PhaseKind.READY_TO_GO_HOME)
        world = WorldState(frame=0, ram=np.zeros(0x24000, dtype=np.uint8), info={}, obs=None)
        task = build_phase_task(TaskBuildContext(), spec, world)
        self.assertEqual(task.__class__.__name__, "ReadyToGoHomeTask")
        result = task.step(world)
        self.assertEqual(result.status.name, "SUCCESS")
        self.assertTrue(result.meta.get("ready_to_go_home"))

    def test_crop_builder_honors_work_mode(self) -> None:
        from harvest.tasks.crop_planter import CropWaterTask

        world = WorldState(frame=0, ram=np.zeros(0x24000, dtype=np.uint8), info={}, obs=None)
        set_stamina(world.ram)
        establish = build_phase_task(
            TaskBuildContext(),
            PhaseSpec("CROP_ESTABLISH", PhaseKind.CROP, {"work_mode": "establish"}),
            world,
        )
        water = build_phase_task(
            TaskBuildContext(),
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
        set_stamina(world.ram)
        establish = build_phase_task(TaskBuildContext(), CROP_ESTABLISH_PHASE, world)
        water = build_phase_task(TaskBuildContext(), CROP_WATER_PHASE, world)

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
        pocket = build_phase_task(TaskBuildContext(), pocket_water_phase(), world)
        catalog = build_phase_task(
            TaskBuildContext(),
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
        task = build_phase_task(
            TaskBuildContext(),
            PhaseSpec("HOT_SPRING_STAMINA", PhaseKind.HOT_SPRING, {"min_stamina": 50}),
            world,
        )
        self.assertEqual(task.__class__.__name__, "HotSpringStaminaTask")
        self.assertEqual(task.min_stamina, 50)

    def test_hot_spring_catalog_fills_to_max(self) -> None:
        world = WorldState(frame=0, ram=np.zeros(0x24000, dtype=np.uint8), info={}, obs=None)
        task = build_phase_task(TaskBuildContext(), HOT_SPRING_STAMINA_PHASE, world)
        self.assertEqual(task.__class__.__name__, "HotSpringStaminaTask")
        self.assertIsNone(task.min_stamina)
        self.assertTrue(task.return_to_farm)

    def test_proposal_phase_requires_hearts_feather_and_a_clear_wife_bit(self) -> None:
        from harvest.core.ram_catalog import field_spec
        from harvest.planner.day_phase_registry import ProposalTask, proposal_phase_for_ram
        from harvest.planner.day_plan_phases import build_day_phases_from_ram

        empty = np.zeros(0x20000, dtype=np.uint8)
        self.assertIsNone(proposal_phase_for_ram(empty))
        self.assertNotIn("PROPOSE", [phase.phase for phase in build_day_phases_from_ram(empty)])

        ram = np.zeros(0x20000, dtype=np.uint8)
        hearts = field_spec("ann_hearts").address
        flags = field_spec("marriage_flags").address
        ram[hearts] = 851 & 0xFF
        ram[hearts + 1] = 851 >> 8
        ram[flags] = 0x80
        ram[0x11F02] = 0x80
        ram[field_spec("tilemap").address] = 0x1C
        ram[field_spec("hour").address] = 10
        ram[field_spec("weekday").address] = 6
        ram[field_spec("season").address] = 1
        ram[field_spec("day").address] = 1
        phase = proposal_phase_for_ram(ram)
        self.assertIsNotNone(phase)
        self.assertEqual(phase.kind, PhaseKind.PROPOSE)
        self.assertEqual(phase.failure_policy, "optional")
        names = [row.phase for row in build_day_phases_from_ram(ram)]
        self.assertNotIn("PROPOSE", names)
        self.assertNotIn("COURT_ANN", names)

        ram[flags] = 0x82
        self.assertIsNone(proposal_phase_for_ram(ram))

        world = WorldState(frame=0, ram=ram, info={}, obs=None)
        married = ProposalTask()
        married.reset(world)
        self.assertEqual(married.step(world).status.name, "SUCCESS")

        ram[flags] = 0x80
        ram[field_spec("tool_selected").address] = 0x18
        ram[field_spec("hour").address] = 0x11
        gated = ProposalTask()
        gated.reset(world)
        result = gated.step(world)
        self.assertEqual(result.status.name, "FAILURE")
        self.assertIn("hour gate", result.reason or "")

        ram[field_spec("hour").address] = 11
        self.assertIsNone(proposal_phase_for_ram(ram))
        ram[field_spec("hour").address] = 10
        ram[field_spec("weekday").address] = 5
        self.assertIsNone(proposal_phase_for_ram(ram))
        ram[field_spec("weekday").address] = 6
        ram[hearts] = ram[hearts + 1] = 0
        eve = field_spec("eve_hearts").address
        ram[eve] = 250
        self.assertIsNone(proposal_phase_for_ram(ram))


if __name__ == "__main__":
    unittest.main()
