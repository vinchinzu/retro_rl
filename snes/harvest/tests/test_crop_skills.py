"""Reactive hoe/plant/water/carry skills — no tape replay."""

from __future__ import annotations

import unittest

import numpy as np

from harvest.core.carry import ADDR_TOOL_BACKPACK, SEED_ITEM
from harvest.core.ram_catalog import field_spec
from harvest.core.tile_catalog import (
    ADDR_INPUT_LOCK,
    ADDR_MAP,
    ADDR_TILEMAP,
    ADDR_TOOL,
    ADDR_X,
    ADDR_Y,
    MAP_WIDTH,
    TILE_SIZE,
    Tool,
)
from harvest.maps.farm_pond import FARM_NO_GO_TILES, WEST_POCKET_PLANT_CENTER
from harvest.tasks.crop_geometry import WATER_PLAN_CENTER, hoe_plan, plot_tiles
from harvest.tasks.crop_skills import (
    PLANTED_DRY,
    PLANTED_WET,
    PLOT_RING_SIZE,
    PlantPlotSkill,
    SelectCarrySkill,
    count_ring_planted,
    count_ring_wet,
    hoe_stand_px,
    hoe_until_tilled_skill,
    plant_until_crop_skill,
    plant_until_plot_skill,
    pocket_hoe_ring_skills,
    pocket_water_ring_skills,
    remap_pocket_hoe_stand,
)
from harvest.tasks.skills import (
    farm_nav_pocket_hoe_stand_skill,
    farm_pocket_plant_skill,
    farm_pocket_water_skill,
)
from retro_harness import TaskStatus, WorldState
from retro_harness.actions import action_names

ADDR_DIR = field_spec("player_direction").address


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
    ram[ADDR_INPUT_LOCK] = 1
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

    def test_swaps_wait_when_input_locked(self) -> None:
        ram = _ram(selected=int(Tool.HOE), backpack=SEED_ITEM["potato"])
        ram[ADDR_INPUT_LOCK] = 0
        world = WorldState(frame=0, ram=ram, info={}, obs=None)
        skill = SelectCarrySkill(wanted=SEED_ITEM["potato"])
        skill.reset(world)
        result = skill.step(world)
        self.assertEqual(result.status, TaskStatus.RUNNING)
        self.assertNotIn("X", set(action_names(result.action.action)))


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

    def test_hoe_holds_face_until_ram_direction_then_y(self) -> None:
        ram = _ram(tile=(13, 29), tid=0x01, selected=int(Tool.HOE))
        ram[ADDR_MAP + 28 * MAP_WIDTH + 13] = 0x01
        ram[ADDR_DIR] = 0
        world = WorldState(frame=0, ram=ram, info={}, obs=None)
        skill = hoe_until_tilled_skill(target_tile=(13, 28), face="up")
        skill.reset(world)
        first = skill.step(world)
        self.assertIn("UP", set(action_names(first.action.action)))
        self.assertNotIn("Y", set(action_names(first.action.action)))
        ram[ADDR_DIR] = 1
        for _ in range(6):
            settle = skill.step(world)
            self.assertNotIn("Y", set(action_names(settle.action.action)))
            self.assertNotIn("UP", set(action_names(settle.action.action)))
        y_press = skill.step(world)
        self.assertIn("Y", set(action_names(y_press.action.action)))
        self.assertNotIn("UP", set(action_names(y_press.action.action)))

    def test_hoe_waits_when_input_locked(self) -> None:
        ram = _ram(tile=(13, 29), tid=0x01, selected=int(Tool.HOE))
        ram[ADDR_MAP + 28 * MAP_WIDTH + 13] = 0x01
        ram[ADDR_INPUT_LOCK] = 0
        world = WorldState(frame=0, ram=ram, info={}, obs=None)
        skill = hoe_until_tilled_skill(target_tile=(13, 28), face="up")
        skill.reset(world)
        result = skill.step(world)
        self.assertEqual(result.status, TaskStatus.RUNNING)
        self.assertNotIn("Y", set(action_names(result.action.action)))

    def test_plant_succeeds_when_bag_spent_before_tile_updates(self) -> None:
        ram = _ram(
            tile=(13, 28),
            tid=0x07,
            selected=SEED_ITEM["potato"],
            backpack=int(Tool.HOE),
        )
        world = WorldState(frame=0, ram=ram, info={}, obs=None)
        skill = plant_until_crop_skill(target_tile=(13, 28))
        skill.reset(world)
        first = skill.step(world)
        self.assertEqual(first.status, TaskStatus.RUNNING)
        ram[ADDR_TOOL] = 0
        done = skill.step(world)
        self.assertEqual(done.status, TaskStatus.SUCCESS)
        self.assertIn("bag spent", done.reason or "")

    def test_plant_on_weed_fails_fast(self) -> None:
        world = WorldState(
            frame=0,
            ram=_ram(tid=0x03, selected=SEED_ITEM["potato"]),
            info={},
            obs=None,
        )
        skill = plant_until_crop_skill()
        skill.reset(world)
        result = skill.step(world)
        self.assertEqual(result.status, TaskStatus.FAILURE)
        self.assertIn("blocked", result.reason or "")

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
    def test_establish_sequence_hoes_ring_then_plants_plot(self) -> None:
        seq = farm_pocket_plant_skill(include_water=False)
        self.assertEqual(seq.name, "pocket_plant_plot")
        names = [t.name for t in seq.tasks]
        self.assertEqual(names[0], "fence_jump_toss")
        self.assertLess(
            names.index("nav_pocket_hoe_stand"), names.index("select_carry_0x02")
        )
        self.assertLess(
            names.index("select_carry_0x02"), names.index("hoe_until_tilled")
        )
        hoe_steps = [t for t in seq.tasks if t.name == "hoe_until_tilled"]
        self.assertEqual(len(hoe_steps), PLOT_RING_SIZE)
        self.assertEqual(len(hoe_plan(WEST_POCKET_PLANT_CENTER)), PLOT_RING_SIZE)
        hoe_targets = {t.target_tile for t in hoe_steps}
        self.assertNotIn(WEST_POCKET_PLANT_CENTER, hoe_targets)
        self.assertEqual(
            hoe_targets,
            {target for target, _stand, _face in hoe_plan(WEST_POCKET_PLANT_CENTER)},
        )
        self.assertIn("plant_until_plot", names)
        self.assertNotIn("plant_until_crop", names)
        self.assertLess(names.index("hoe_until_tilled"), names.index("nav_pocket_plant"))
        self.assertLess(names.index("nav_pocket_plant"), names.index("plant_until_plot"))
        self.assertNotIn("water_until_wet", names)
        self.assertTrue(all("record" not in n for n in names))

    def test_hoe_only_skips_plant(self) -> None:
        seq = farm_pocket_plant_skill(include_plant=False)
        names = [t.name for t in seq.tasks]
        self.assertEqual(sum(1 for n in names if n == "hoe_until_tilled"), PLOT_RING_SIZE)
        self.assertNotIn("plant_until_plot", names)
        self.assertNotIn("nav_pocket_plant", names)

    def test_hoe_stand_px_nudges_away_from_target(self) -> None:
        cx, cy = 13 * TILE_SIZE + 8, 30 * TILE_SIZE + 8
        self.assertEqual(hoe_stand_px((13, 30), "up"), (cx, cy))
        self.assertEqual(hoe_stand_px((13, 29), "up"), (13 * TILE_SIZE + 8, 29 * TILE_SIZE + 8 + 5))
        self.assertEqual(hoe_stand_px((15, 27), "left"), (15 * TILE_SIZE + 8 + 5, 27 * TILE_SIZE + 8))

    def test_hoe_ring_skills_match_hoe_plan(self) -> None:
        skills = pocket_hoe_ring_skills(WEST_POCKET_PLANT_CENTER)
        self.assertEqual(len(skills), PLOT_RING_SIZE * 2)
        self.assertTrue(all(s.name.startswith("nav_hoe_ring_") for s in skills[0::2]))
        self.assertEqual(skills[0].name, "nav_hoe_ring_0_right")
        self.assertEqual([s.name for s in skills[1::2]], ["hoe_until_tilled"] * PLOT_RING_SIZE)

    def test_hoe_stand_nav_leaves_shed_door_south_then_west(self) -> None:
        ram = _ram(tile=(26, 30), tid=0x01)
        ram[ADDR_TILEMAP] = 0x00
        py = 30 * TILE_SIZE + 4
        ram[ADDR_Y] = py & 0xFF
        ram[ADDR_Y + 1] = (py >> 8) & 0xFF
        world = WorldState(frame=0, ram=ram, info={}, obs=None)
        skill = farm_nav_pocket_hoe_stand_skill()
        skill.reset(world)
        first = skill.step(world)
        self.assertEqual(first.status, TaskStatus.RUNNING)
        self.assertIn("DOWN", set(action_names(first.action.action)))
        py = 30 * TILE_SIZE + 8
        ram[ADDR_Y] = py & 0xFF
        ram[ADDR_Y + 1] = (py >> 8) & 0xFF
        second = skill.step(world)
        self.assertEqual(second.status, TaskStatus.RUNNING)
        self.assertIn("LEFT", set(action_names(second.action.action)))
        self.assertNotIn("UP", set(action_names(second.action.action)))

    def test_hoe_stand_nav_fails_inside_shed(self) -> None:
        ram = _ram()
        ram[ADDR_TILEMAP] = 0x26
        world = WorldState(frame=0, ram=ram, info={}, obs=None)
        skill = farm_nav_pocket_hoe_stand_skill()
        skill.reset(world)
        result = skill.step(world)
        self.assertEqual(result.status, TaskStatus.FAILURE)
        self.assertIn("left map", result.reason or "")

    def test_include_water_uses_eight_ring(self) -> None:
        seq = farm_pocket_plant_skill(include_water=True)
        names = [t.name for t in seq.tasks]
        self.assertIn("plant_until_plot", names)
        self.assertIn("pocket_water_ring", names)
        self.assertEqual(names.count("water_until_wet"), 0)
        water = next(t for t in seq.tasks if t.name == "pocket_water_ring")
        water_names = [t.name for t in water.tasks]
        self.assertEqual(
            sum(1 for n in water_names if n == "water_until_wet"), PLOT_RING_SIZE
        )

    def test_hoe_ring_never_stands_on_well_no_go(self) -> None:
        cx, cy = WEST_POCKET_PLANT_CENTER
        skills = pocket_hoe_ring_skills((cx, cy))
        hoes = skills[1::2]
        navs = skills[0::2]
        stands = []
        for nav in navs:
            px, py = nav.target_px
            stands.append((px // TILE_SIZE, py // TILE_SIZE))
        self.assertNotIn((15, 27), stands)
        for stand in stands:
            self.assertNotIn(stand, FARM_NO_GO_TILES)
        self.assertEqual(
            {h.target_tile for h in hoes},
            {target for target, _stand, _face in hoe_plan((cx, cy))},
        )
        well_target = (cx + 1, cy - 1)
        well_hoe = next(h for h in hoes if h.target_tile == well_target)
        self.assertEqual(well_hoe.face, "up")
        well_nav = skills[list(skills).index(well_hoe) - 1]
        self.assertEqual(
            (well_nav.target_px[0] // TILE_SIZE, well_nav.target_px[1] // TILE_SIZE),
            (14, 28),
        )
        stand, face = remap_pocket_hoe_stand((cx, cy), well_target, (15, 27), "left")
        self.assertEqual((stand, face), ((14, 28), "up"))
        fence_stand, fence_face = remap_pocket_hoe_stand(
            (cx, cy), (cx, cy + 1), (cx, cy + 2), "up"
        )
        self.assertEqual((fence_stand, fence_face), ((cx - 1, cy + 1), "right"))

    def test_pocket_water_ring_waters_eight_skips_center(self) -> None:
        cx, cy = WEST_POCKET_PLANT_CENTER
        skills = pocket_water_ring_skills((cx, cy))
        waters = [s for s in skills if s.name == "water_until_wet"]
        self.assertEqual(len(waters), PLOT_RING_SIZE)
        targets = {s.target_tile for s in waters}
        self.assertNotIn((cx, cy), targets)
        self.assertEqual(targets, set(plot_tiles((cx, cy), include_center=False)))
        expected = {
            ((cx + tdx, cy + tdy), face)
            for tdx, tdy, _sdx, _sdy, face in WATER_PLAN_CENTER
            if (tdx, tdy) != (0, 0)
        }
        self.assertEqual({(s.target_tile, s.face) for s in waters}, expected)
        seq = farm_pocket_water_skill()
        self.assertEqual(seq.name, "pocket_water_ring")
        names = [t.name for t in seq.tasks]
        self.assertEqual(sum(1 for n in names if n == "water_until_wet"), PLOT_RING_SIZE)
        self.assertIn("nav_pocket_plant", names)
        self.assertIn(f"select_carry_0x{int(Tool.WATERING_CAN):02X}", names)

    def test_water_until_wet_on_target_succeeds(self) -> None:
        ram = _ram(tile=(13, 28), tid=0x01, selected=int(Tool.WATERING_CAN))
        ram[ADDR_MAP + 27 * MAP_WIDTH + 13] = PLANTED_WET
        world = WorldState(frame=0, ram=ram, info={}, obs=None)
        from harvest.tasks.crop_skills import water_until_wet_skill

        skill = water_until_wet_skill(target_tile=(13, 27), face="up")
        skill.reset(world)
        self.assertEqual(skill.step(world).status, TaskStatus.SUCCESS)
        self.assertEqual(count_ring_wet(ram, WEST_POCKET_PLANT_CENTER), 1)


class PlantPlotSkillTests(unittest.TestCase):
    def test_counts_ring_not_center(self) -> None:
        ram = _ram(tile=(13, 28), tid=0x01, selected=SEED_ITEM["potato"])
        cx, cy = WEST_POCKET_PLANT_CENTER
        for tx, ty in plot_tiles((cx, cy), include_center=False):
            ram[ADDR_MAP + ty * MAP_WIDTH + tx] = PLANTED_DRY
        ram[ADDR_MAP + cy * MAP_WIDTH + cx] = 0x01
        self.assertEqual(count_ring_planted(ram, (cx, cy)), PLOT_RING_SIZE)

    def test_eight_ring_tiles_is_success(self) -> None:
        ram = _ram(tile=(13, 28), tid=0x01, selected=SEED_ITEM["potato"])
        cx, cy = WEST_POCKET_PLANT_CENTER
        for tx, ty in plot_tiles((cx, cy), include_center=False):
            ram[ADDR_MAP + ty * MAP_WIDTH + tx] = PLANTED_DRY
        world = WorldState(frame=0, ram=ram, info={}, obs=None)
        skill = plant_until_plot_skill()
        skill.reset(world)
        result = skill.step(world)
        self.assertEqual(result.status, TaskStatus.SUCCESS)
        self.assertIn("planted=8", result.reason or "")

    def test_bag_spent_with_two_tiles_is_failure(self) -> None:
        ram = _ram(
            tile=(13, 28),
            tid=0x01,
            selected=SEED_ITEM["potato"],
            backpack=int(Tool.HOE),
        )
        cx, cy = WEST_POCKET_PLANT_CENTER
        ram[ADDR_MAP + (cy + 1) * MAP_WIDTH + cx] = PLANTED_DRY
        ram[ADDR_MAP + cy * MAP_WIDTH + cx] = PLANTED_DRY
        world = WorldState(frame=0, ram=ram, info={}, obs=None)
        skill = PlantPlotSkill()
        skill.reset(world)
        first = skill.step(world)
        self.assertEqual(first.status, TaskStatus.RUNNING)
        ram[ADDR_TOOL] = 0
        ram[ADDR_TOOL_BACKPACK] = int(Tool.HOE)
        done = skill.step(world)
        self.assertEqual(done.status, TaskStatus.FAILURE)
        self.assertIn("bag spent planted=1", done.reason or "")


if __name__ == "__main__":
    unittest.main()
