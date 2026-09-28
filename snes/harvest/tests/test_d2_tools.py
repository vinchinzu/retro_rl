"""D2 leftover hammer/axe RAM shelf — EnsureCarryToolTask, not GET_* recordings."""

from __future__ import annotations

from pathlib import Path
import sys
import unittest

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


class ShedHammerAxeSpecTests(unittest.TestCase):
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
