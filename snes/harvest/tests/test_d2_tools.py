"""D2 leftover hammer/axe RAM shelf — EnsureCarryToolTask, not GET_* recordings."""

from __future__ import annotations

from pathlib import Path
import sys
import unittest
import json
import pytest

_TESTS_DIR = Path(__file__).resolve().parent
if str(_TESTS_DIR) not in sys.path:
    sys.path.insert(0, str(_TESTS_DIR))
from day_plan_test_helpers import make_world, set_player_pos

from harvest.core.tile_catalog import Tool
from harvest.planner.d2_work import (
    ensure_axe_phase,
    ensure_hammer_phase,
)
from harvest.planner.day_phase_types import PhaseKind
from harvest.planner.tasks.inventory_shed import EnsureCarryToolTask, SHED_TOOL_SPECS, ShedShelfToolTask
from harvest.planner.tasks.navigation import NavTask
from retro_harness import TaskStatus


@pytest.mark.rom
@pytest.mark.parametrize("state", ["y1_complete/progress_f254000", "y1_complete/progress_f255000"])
def test_d2_hammer_ready_leaves_a_loaded_farm(state):
    """Equipping the hammer indoors must not release the next farm phase."""
    from retro_harness import AuditCapabilities, AuditedEnv, WorldState
    from harvest.runtime.retro_setup import make_harvest_env
    from harvest.core.ram_catalog import read_ram_value
    from harvest.planner.day_plan_status import is_farm_tilemap
    from harvest.tasks.farm_clear_quota import farm_map_loaded
    from harvest.tasks.nav import get_pos_from_ram, make_action

    env = AuditedEnv(make_harvest_env(state, lean_step=True),
                     capabilities=AuditCapabilities.all("harvest.d2_hammer_exit"))
    try:
        env.reset()
        task = EnsureCarryToolTask(tool_id=int(Tool.HAMMER))
        task.reset(WorldState(frame=0, ram=env.get_ram(), info={}, obs=None))
        for frame in range(5000):
            result = task.step(WorldState(frame=frame, ram=env.get_ram(), info={}, obs=None))
            if result.status != TaskStatus.RUNNING:
                break
            env.step(result.action.action if result.action else make_action())
        ram = env.get_ram()
        evidence = dict(state=state, frame=frame, status=result.status,
                        reason=result.reason, tilemap=read_ram_value(ram, "tilemap"),
                        pos=get_pos_from_ram(ram))
        assert result.status == TaskStatus.SUCCESS, evidence
        assert farm_map_loaded(ram), evidence
        from harvest.planner.tasks.transitions import screen_faded_in
        assert screen_faded_in(ram), evidence
        for _ in range(90):
            env.step(make_action())
        ram = env.get_ram()
        evidence.update(tilemap=read_ram_value(ram, "tilemap"), pos=get_pos_from_ram(ram))
        assert is_farm_tilemap(read_ram_value(ram, "tilemap")), evidence
        assert farm_map_loaded(ram), evidence
        assert env.audit().ram_writes == env.audit().mid_run_loads == 0
    finally:
        env.close()


@pytest.mark.rom
@pytest.mark.parametrize("state", ["LN_marriage_shelf_before_take", "marriage_feather_probe"])
def test_blue_feather_native_pickup(state):
    """Exercise the tile-boundary miss and the real morning-to-shed fetch."""
    from retro_harness import AuditCapabilities, AuditedEnv, WorldState
    from harvest.runtime.retro_setup import make_harvest_env
    from harvest.core.ram_catalog import read_ram_u8, read_ram_value
    from harvest.planner.tasks.shed_slots import feather_shelf_taken
    from harvest.tasks.nav import make_action
    from harvest.paths import PROJECT_DIR

    env = AuditedEnv(make_harvest_env(state, lean_step=True), capabilities=AuditCapabilities(
        provider="harvest.feather_rom_test", observes_ram_writes=True,
        observes_mid_run_loads=True, observes_assists=True))
    try:
        env.reset()
        start = {"tool_selected": read_ram_value(env.get_ram(), "tool_selected"),
                 "shed_row": read_ram_u8(env.get_ram(), 0x11F02)}
        assert start["shed_row"] & 0x80
        task = EnsureCarryToolTask(tool_id=0x18, exit_when_done=False)
        task.reset(WorldState(frame=0, ram=env.get_ram(), info={}, obs=None))
        for frame in range(8000):
            result = task.step(WorldState(frame=frame, ram=env.get_ram(), info={}, obs=None))
            if result.status != TaskStatus.RUNNING:
                break
            env.step(result.action.action if result.action else make_action())
        ram = env.get_ram()
        audit = env.audit()
        passed = (result.status == TaskStatus.SUCCESS and feather_shelf_taken(ram)
                  and audit.ram_writes == 0 and audit.mid_run_loads == 0)
        report = {"state": state, "frames": frame, "passed": passed, "start": start,
                  "end": {"tool_selected": read_ram_value(ram, "tool_selected"),
                          "shed_row": read_ram_u8(ram, 0x11F02)},
                  "reason": result.reason,
                  "clean_run": {"ram_writes": audit.ram_writes,
                                "mid_run_state_loads": audit.mid_run_loads}}
        if state == "marriage_feather_probe":
            (PROJECT_DIR / "recordings" / "marriage_feather.json").write_text(json.dumps(report, indent=2) + "\n")
        assert passed, report
    finally:
        env.close()


@pytest.mark.rom
def test_d2_hammer_handoff_breaks_a_real_boulder_through_composer():
    from retro_harness import AuditCapabilities, AuditedEnv, WorldState
    from harvest.runtime.retro_setup import make_harvest_env
    from harvest.planner.day_plan import DayPlanTask
    from harvest.planner.d2_work import rock_clear_phase
    from harvest.planner.d2_farm_chunks import FARM_CHUNK_BOUNDS
    from harvest.tasks.farm_clear_quota import count_debris, farm_map_loaded
    from harvest.tasks.nav import make_action

    env = AuditedEnv(make_harvest_env("y1_complete/progress_f255000", lean_step=True),
                     capabilities=AuditCapabilities.all("harvest.d2_hammer_handoff"))
    try:
        env.reset()
        from harvest.planner.day_plan_phases import DayPlannerPolicy
        task = DayPlanTask(policy=DayPlannerPolicy(include_end_day=False), phase_sequence=[ensure_hammer_phase(),
            rock_clear_phase(farm_bounds=FARM_CHUNK_BOUNDS["nw"], chunk="nw")])
        task.reset(WorldState(frame=0, ram=env.get_ram(), info={}, obs=None))
        before = None
        for frame in range(8000):
            ram = env.get_ram()
            if task._phase_index >= 1 and farm_map_loaded(ram):
                now = count_debris(ram).large_rocks
                if before is None:
                    before = now
                elif now < before:
                    break
            result = task.step(WorldState(frame=frame, ram=ram, info={}, obs=None))
            assert result.status == TaskStatus.RUNNING, result.reason
            env.step(result.action.action if result.action else make_action())
        assert before is not None and now < before, (frame, before, result.reason)
        assert env.audit().ram_writes == env.audit().mid_run_loads == 0
    finally:
        env.close()


class ShedHammerAxeSpecTests(unittest.TestCase):
    def test_tool_fetch_propagates_exit_and_step_off_failure(self) -> None:
        from types import SimpleNamespace
        from harvest.planner.tasks.inventory_shed import ShedFetchItemTask
        from retro_harness import TaskResult

        for phase in ("exit_after_ready", "leave_door"):
            for status in (TaskStatus.FAILURE, TaskStatus.BLOCKED):
                with self.subTest(phase=phase, status=status):
                    world = make_world(0x00)
                    world.ram[0x0921] = int(Tool.HAMMER)
                    task = ShedFetchItemTask(item_id=int(Tool.HAMMER))
                    task.reset(world)
                    task._phase = phase
                    task._task = SimpleNamespace(step=lambda world: TaskResult(status, reason="exit blocked"))
                    result = task.step(world)
                    self.assertEqual(result.status, TaskStatus.FAILURE)
                    self.assertIn("exit blocked", result.reason)

    def test_hammer_and_axe_are_bottom_shelf_stands(self) -> None:
        hammer = SHED_TOOL_SPECS[int(Tool.HAMMER)]
        axe = SHED_TOOL_SPECS[int(Tool.AXE)]
        self.assertEqual(int(Tool.HAMMER), 0x03)
        self.assertEqual(int(Tool.AXE), 0x04)
        self.assertEqual(hammer.inside_stand_px, (176, 168))
        self.assertEqual(axe.inside_stand_px, (192, 168))
        self.assertEqual(hammer.inside_face, "up")
        self.assertEqual(axe.inside_face, "up")
        self.assertIsNone(hammer.inside_recording)
        self.assertIsNone(axe.inside_recording)
        self.assertAlmostEqual(hammer.inside_stand_px[1], 168, delta=2)
        self.assertAlmostEqual(axe.inside_stand_px[1], 168, delta=2)

    def test_blue_feather_stands_above_the_shelf_facing_down(self) -> None:
        feather = SHED_TOOL_SPECS[0x18]
        self.assertEqual(feather.item_id, 0x18)
        self.assertEqual(feather.inside_stand_px, (176, 185))
        self.assertEqual(feather.inside_face, "down")
        self.assertIsNone(feather.inside_recording)
        world = make_world(0x26)
        task = EnsureCarryToolTask(tool_id=0x18)
        task.reset(world)
        result = task.step(world)
        self.assertEqual(result.status, TaskStatus.RUNNING)
        self.assertEqual(task._phase, "inside")
        self.assertIsInstance(task._task, ShedShelfToolTask)
        self.assertEqual(task._task.stand_px, (176, 185))
        self.assertEqual(task._task.face, "down")

    def test_feather_pickup_requires_selected_tool_and_clear_shelf_bit(self) -> None:
        world = make_world(0x26)
        task = ShedShelfToolTask(tool_id=0x18, settle_frames=0, take_attempts=1)
        task.reset(world)
        task._phase = "take"
        task._take_tries = 1

        world.ram[0x0921] = 0x18
        world.ram[0x11F02] = 0x80
        task._step_count = 0
        still_on_shelf = task.step(world)
        self.assertNotEqual(still_on_shelf.status, TaskStatus.SUCCESS)

        world.ram[0x0921] = 0x0F
        world.ram[0x0923] = 0x18
        world.ram[0x11F02] = 0x00
        task._phase = "take"
        task._take_tries = 1
        task._step_count = 0
        backpack_only = task.step(world)
        self.assertNotEqual(backpack_only.status, TaskStatus.SUCCESS)

        world.ram[0x0921] = 0x18
        world.ram[0x0923] = 0x0F
        world.ram[0x11F02] = 0x40
        task._step_count = 0
        taken = task.step(world)
        self.assertEqual(taken.status, TaskStatus.SUCCESS)
        self.assertIn("0x18", taken.reason or "")

    def test_sickle_is_bottom_shelf_between_can_and_hoe_sprite(self) -> None:
        sickle = SHED_TOOL_SPECS[int(Tool.SICKLE)]
        can = SHED_TOOL_SPECS[int(Tool.WATERING_CAN)]
        self.assertEqual(int(Tool.SICKLE), 0x01)
        self.assertEqual(sickle.inside_stand_px, (144, 168))
        self.assertGreater(sickle.inside_stand_px[0], can.inside_stand_px[0])
        self.assertIsNone(sickle.inside_recording)

    def test_ensure_hammer_constructs_and_succeeds_when_already_selected(self) -> None:
        world = make_world(0x00)
        world.ram[0x0921] = int(Tool.HAMMER)
        world.ram[0x0923] = 0x00
        task = EnsureCarryToolTask(tool_id=int(Tool.HAMMER))
        task.reset(world)
        result = task.step(world)
        self.assertEqual(result.status, TaskStatus.SUCCESS)
        self.assertIn("0x03", result.reason or "")

    def test_ensure_hammer_on_shed_door_walks_to_loaded_stand(self) -> None:
        from harvest.core.tile_catalog import ADDR_MAP, MAP_WIDTH, TILE_SIZE
        from harvest.tasks.farm_ops import LOADED_FARM_STAND, SHED_DOOR_TILE

        world = make_world(0x00)
        world.ram[0x0921] = int(Tool.HAMMER)
        world.ram[0x0923] = 0x00
        px = SHED_DOOR_TILE[0] * TILE_SIZE + 8
        py = SHED_DOOR_TILE[1] * TILE_SIZE + 8
        set_player_pos(world.ram, px, py)
        world.ram[ADDR_MAP + SHED_DOOR_TILE[1] * MAP_WIDTH + SHED_DOOR_TILE[0]] = 0xFF
        task = EnsureCarryToolTask(tool_id=int(Tool.HAMMER))
        task.reset(world)
        result = task.step(world)
        self.assertEqual(result.status, TaskStatus.RUNNING)
        self.assertEqual(task._phase, "leave_door")
        self.assertIsInstance(task._task, NavTask)
        stand = LOADED_FARM_STAND
        self.assertEqual(
            (task._task.target_px.x, task._task.target_px.y),
            (stand[0] * TILE_SIZE + 8, stand[1] * TILE_SIZE + 8),
        )

    def test_ensure_hammer_in_shed_uses_ram_shelf_not_recording(self) -> None:
        world = make_world(0x26)
        task = EnsureCarryToolTask(tool_id=int(Tool.HAMMER))
        task.reset(world)
        result = task.step(world)
        self.assertEqual(result.status, TaskStatus.RUNNING)
        self.assertEqual(task._phase, "inside")
        self.assertIsInstance(task._task, ShedShelfToolTask)
        self.assertEqual(task._task.stand_px, (176, 168))
        self.assertEqual(task._task.tool_id, int(Tool.HAMMER))

    def test_ensure_can_from_crop_pocket_with_stage2_potatoes(self) -> None:
        """rr-w7t9: can-recovery at pos=(216,456) must not be sealed by 0x58 crops."""
        from harvest.core.tile_catalog import ADDR_MAP, MAP_WIDTH
        from harvest.maps.map_config import WEST_POCKET_PLANT_CENTER
        world = make_world(0x00)
        set_player_pos(world.ram, 216, 456)  # tile (13, 28)
        # Surround (13, 28) with stage 2 dry potatoes (0x58)
        cx, cy = WEST_POCKET_PLANT_CENTER
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                if dx == 0 and dy == 0:
                    continue
                world.ram[ADDR_MAP + (cy + dy) * MAP_WIDTH + (cx + dx)] = 0x58

        task = EnsureCarryToolTask(tool_id=int(Tool.WATERING_CAN))
        task.reset(world)
        result = task.step(world)

        # Must start routing, NOT fail with no_path sealed!
        self.assertEqual(result.status, TaskStatus.RUNNING)
        self.assertNotIn("no_path sealed", result.reason or "")
        self.assertEqual(task._phase, "route")


class LeftoverEnsureNotRecordedTests(unittest.TestCase):
    def test_leftover_uses_ensure_tool_not_get_hammer_macro(self) -> None:
        hammer = ensure_hammer_phase()
        axe = ensure_axe_phase()
        self.assertEqual(hammer.phase, "ENSURE_HAMMER")
        self.assertEqual(axe.phase, "ENSURE_AXE")
        self.assertEqual(hammer.kind, PhaseKind.ENSURE_TOOL)
        self.assertEqual(axe.kind, PhaseKind.ENSURE_TOOL)
        self.assertEqual(hammer.params["tool_id"], int(Tool.HAMMER))
        self.assertEqual(axe.params["tool_id"], int(Tool.AXE))
        self.assertNotEqual(hammer.kind, PhaseKind.RECORDED)
        self.assertNotEqual(axe.kind, PhaseKind.RECORDED)

    def test_low_stam_leftover_spa_is_full_restore_then_return(self) -> None:
        from harvest.planner.day_phase_stamina import full_restore_spa_phase

        spa = full_restore_spa_phase()
        self.assertEqual(spa.phase, "HOT_SPRING_STAMINA")
        self.assertEqual(spa.params["min_stamina"], "full")
        self.assertTrue(spa.params["return_to_farm"])


if __name__ == "__main__":
    unittest.main()
