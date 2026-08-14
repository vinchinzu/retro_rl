"""Reactive hoe/plant/water/carry skills — no tape replay."""

from __future__ import annotations

import unittest

import numpy as np

from harvest.core.carry import ADDR_TOOL_BACKPACK, SEED_ITEM
from harvest.core.tile_catalog import (
    ADDR_MAP,
    ADDR_TOOL,
    ADDR_X,
    ADDR_Y,
    MAP_WIDTH,
    TILE_SIZE,
    Tool,
)
from harvest.tasks.crop_skills import (
    PLANTED_DRY,
    SelectCarrySkill,
    hoe_until_tilled_skill,
    plant_until_crop_skill,
)
from harvest.tasks.skills import farm_pocket_plant_skill
from retro_harness import TaskStatus, WorldState


def _ram(*, tile=(13, 28), tid=0x01, selected=0, backpack=0) -> np.ndarray:
    ram = np.zeros(ADDR_MAP + MAP_WIDTH * MAP_WIDTH, dtype=np.uint8)
    px = tile[0] * TILE_SIZE + 8
    py = tile[1] * TILE_SIZE + 8
    ram[ADDR_X] = px & 0xFF
    ram[ADDR_X + 1] = (px >> 8) & 0xFF
    ram[ADDR_Y] = py & 0xFF
    ram[ADDR_Y + 1] = (py >> 8) & 0xFF
    ram[ADDR_TOOL] = selected
    ram[ADDR_TOOL_BACKPACK] = backpack
    ram[ADDR_MAP + tile[1] * MAP_WIDTH + tile[0]] = tid
    return ram


class SelectCarryTests(unittest.TestCase):
    def test_already_selected_is_success(self) -> None:
        world = WorldState(frame=0, ram=_ram(selected=int(Tool.HOE)), info={}, obs=None)
        skill = SelectCarrySkill(wanted=int(Tool.HOE))
        skill.reset(world)
        self.assertEqual(skill.step(world).status, TaskStatus.SUCCESS)

    def test_swaps_when_wanted_in_backpack(self) -> None:
        ram = _ram(selected=int(Tool.HOE), backpack=SEED_ITEM["potato"])
        world = WorldState(frame=0, ram=ram, info={}, obs=None)
        skill = SelectCarrySkill(wanted=SEED_ITEM["potato"])
        skill.reset(world)
        result = skill.step(world)
        self.assertEqual(result.status, TaskStatus.RUNNING)
        self.assertIsNotNone(result.action)

    def test_missing_tool_fails(self) -> None:
        world = WorldState(frame=0, ram=_ram(selected=0, backpack=0), info={}, obs=None)
        skill = SelectCarrySkill(wanted=int(Tool.HOE))
        skill.reset(world)
        self.assertEqual(skill.step(world).status, TaskStatus.FAILURE)


class UseToolUntilTileTests(unittest.TestCase):
    def test_hoe_already_tilled_is_success(self) -> None:
        world = WorldState(
            frame=0,
            ram=_ram(tid=0x07, selected=int(Tool.HOE)),
            info={},
            obs=None,
        )
        skill = hoe_until_tilled_skill()
        skill.reset(world)
        self.assertEqual(skill.step(world).status, TaskStatus.SUCCESS)

    def test_hoe_on_weed_fails_fast(self) -> None:
        world = WorldState(
            frame=0,
            ram=_ram(tid=0x03, selected=int(Tool.HOE)),
            info={},
            obs=None,
        )
        skill = hoe_until_tilled_skill()
        skill.reset(world)
        result = skill.step(world)
        self.assertEqual(result.status, TaskStatus.FAILURE)
        self.assertIn("blocked", result.reason or "")

    def test_hoe_watches_faced_target_instead_of_stand_tile(self) -> None:
        ram = _ram(tile=(13, 29), tid=0x01, selected=int(Tool.HOE))
        ram[ADDR_MAP + 28 * MAP_WIDTH + 13] = 0x07
        world = WorldState(frame=0, ram=ram, info={}, obs=None)
        skill = hoe_until_tilled_skill(target_tile=(13, 28), face="up")
        skill.reset(world)

        result = skill.step(world)

        self.assertEqual(result.status, TaskStatus.SUCCESS)

    def test_plant_already_crop_is_success(self) -> None:
        world = WorldState(
            frame=0,
            ram=_ram(tid=PLANTED_DRY, selected=SEED_ITEM["potato"]),
            info={},
            obs=None,
        )
        skill = plant_until_crop_skill()
        skill.reset(world)
        self.assertEqual(skill.step(world).status, TaskStatus.SUCCESS)


class PocketPlantComposeTests(unittest.TestCase):
    def test_establish_sequence_has_no_water_and_no_recording(self) -> None:
        seq = farm_pocket_plant_skill(include_water=False)
        self.assertEqual(seq.name, "pocket_plant_cell")
        names = [t.name for t in seq.tasks]
        self.assertEqual(names[0], "fence_jump_toss")
        self.assertLess(
            names.index("nav_pocket_hoe_stand"), names.index("hoe_until_tilled")
        )
        self.assertLess(
            names.index("hoe_until_tilled"), names.index("nav_pocket_plant")
        )
        self.assertIn("hoe_until_tilled", names)
        self.assertIn("plant_until_crop", names)
        self.assertNotIn("water_until_wet", names)
        self.assertTrue(all("record" not in n for n in names))

    def test_include_water_appends_one_cell(self) -> None:
        seq = farm_pocket_plant_skill(include_water=True)
        names = [t.name for t in seq.tasks]
        self.assertIn("water_until_wet", names)


if __name__ == "__main__":
    unittest.main()
