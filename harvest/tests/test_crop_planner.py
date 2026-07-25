from __future__ import annotations

import unittest

import numpy as np

from harvest.planner.crop_planner import (
    CROP_LAYOUTS,
    CROP_SPECS,
    DEFAULT_SHIPPING_TILE,
    CropPlanningConfig,
    build_planting_steps,
    choose_crop_for_date,
    extract_planting_template_from_recording,
    plan_crop_field,
    watering_access_for_layout,
)
from harvest.core.tile_catalog import ADDR_MAP, MAP_WIDTH, STONE, UNTILLED


def _blank_ram(fill: int = UNTILLED) -> np.ndarray:
    ram = np.zeros(ADDR_MAP + MAP_WIDTH * MAP_WIDTH, dtype=np.uint8)
    for idx in range(MAP_WIDTH * MAP_WIDTH):
        ram[ADDR_MAP + idx] = fill
    return ram


def _set_tile(ram: np.ndarray, tx: int, ty: int, tile_id: int) -> None:
    ram[ADDR_MAP + ty * MAP_WIDTH + tx] = tile_id


class CropPlannerTests(unittest.TestCase):
    def test_layouts_model_eight_tile_and_sprinkler_access_seven_tile(self) -> None:
        eight = CROP_LAYOUTS["eight_tile_ring"]
        seven = CROP_LAYOUTS["seven_south_access"]

        self.assertEqual(eight.crop_count, 8)
        self.assertFalse(eight.has_center_access_opening)
        self.assertFalse(eight.sprinkler_ready)

        self.assertEqual(seven.crop_count, 7)
        self.assertTrue(seven.has_center_access_opening)
        self.assertTrue(seven.sprinkler_ready)
        self.assertNotIn((0, 1), seven.crop_offsets)
        self.assertIn((0, 1), seven.access_offsets)

    def test_watering_access_requires_every_tile_to_have_a_stand(self) -> None:
        ram = _blank_ram()
        center = (20, 35)
        eight = CROP_LAYOUTS["eight_tile_ring"]

        access = watering_access_for_layout(ram, center, eight, mode="manual")

        self.assertEqual(len(access), 8)
        crop_tiles = set(eight.crop_tiles(center))
        for item in access:
            self.assertTrue(item.stand_tiles)
            self.assertNotIn(item.stand_tiles[0], crop_tiles)

    def test_sprinkler_mode_uses_center_stand_for_seven_tile_layout(self) -> None:
        ram = _blank_ram()
        center = (20, 35)
        seven = CROP_LAYOUTS["seven_south_access"]

        access = watering_access_for_layout(ram, center, seven, mode="sprinkler")

        self.assertEqual(len(access), 7)
        self.assertEqual({item.stand_tiles[0] for item in access}, {center})

    def test_planner_avoids_shipping_stand_and_prefers_nearby_plots(self) -> None:
        ram = _blank_ram()
        config = CropPlanningConfig(
            seed_type="potato",
            day=1,
            max_seed_bags=3,
            bounds=(10, 28, 25, 40),
            shipping_tile=DEFAULT_SHIPPING_TILE,
        )

        plan = plan_crop_field(ram, config)

        self.assertEqual(plan.seed_bags_needed, 3)
        self.assertEqual(plan.crop_name, "potato")
        self.assertEqual(plan.layout_name, "eight_tile_ring")
        self.assertEqual(plan.plots[0].center, (13, 29))
        for plot in plan.plots:
            self.assertNotIn(DEFAULT_SHIPPING_TILE, plot.crop_tiles)
            self.assertNotIn(DEFAULT_SHIPPING_TILE, plot.water_stands)

    def test_summer_sprinkler_plan_uses_seven_tile_regrow_layout(self) -> None:
        ram = _blank_ram()
        config = CropPlanningConfig(
            seed_type="corn",
            season="summer",
            day=12,
            max_seed_bags=2,
            bounds=(10, 28, 25, 40),
            sprinkler_available=True,
        )

        plan = plan_crop_field(ram, config)

        self.assertEqual(plan.crop_name, "corn")
        self.assertEqual(plan.layout_name, "seven_south_access")
        self.assertEqual([len(plot.crop_tiles) for plot in plan.plots], [7, 7])
        self.assertTrue(all(plot.watering_mode == "sprinkler" for plot in plan.plots))
        self.assertGreater(CROP_SPECS["corn"].harvests_from_planting_day(12), 1)

    def test_candidate_rejects_obstacle_inside_crop_footprint(self) -> None:
        ram = _blank_ram()
        _set_tile(ram, 13, 29, STONE)
        config = CropPlanningConfig(
            seed_type="potato",
            day=1,
            max_seed_bags=1,
            bounds=(12, 28, 14, 30),
        )

        plan = plan_crop_field(ram, config)

        self.assertEqual(plan.seed_bags_needed, 0)

    def test_late_spring_refuses_crop_that_cannot_harvest_before_summer(self) -> None:
        ram = _blank_ram()
        config = CropPlanningConfig(
            seed_type="potato",
            season="spring",
            day=28,
            max_seed_bags=1,
            bounds=(10, 28, 25, 40),
        )

        plan = plan_crop_field(ram, config)

        self.assertEqual(plan.seed_bags_needed, 0)

    def test_choose_crop_for_date_accounts_for_summer_regrow(self) -> None:
        crop = choose_crop_for_date("summer", 12, layout_tiles=7)

        self.assertEqual(crop.name, "corn")
        self.assertGreater(crop.harvests_from_planting_day(12), 1)

    def test_fall_and_winter_have_no_plantable_crops(self) -> None:
        from harvest.planner.crop_planner import (
            is_crop_planting_season,
            resolve_seed_type_for_date,
            should_buy_seeds_for_date,
        )

        self.assertFalse(is_crop_planting_season("fall"))
        self.assertFalse(is_crop_planting_season("winter"))
        self.assertIsNone(choose_crop_for_date("fall", 1))
        self.assertIsNone(resolve_seed_type_for_date("winter", 10))
        self.assertFalse(should_buy_seeds_for_date("fall", 5))

    def test_resolve_seed_type_ignores_potato_stock_in_summer(self) -> None:
        from harvest.planner.crop_planner import resolve_seed_type_for_date

        seed = resolve_seed_type_for_date(
            "summer",
            7,
            inventory={"potato": 58, "corn": 0, "tomato": 0},
        )
        self.assertEqual(seed, "corn")

        seed = resolve_seed_type_for_date(
            "summer",
            7,
            inventory={"potato": 58, "corn": 3, "tomato": 3},
            shipped={"corn": 0, "tomato": 100},
        )
        self.assertEqual(seed, "corn")

        seed = resolve_seed_type_for_date(
            "summer",
            7,
            inventory={"potato": 0, "corn": 3, "tomato": 3},
            shipped={"corn": 400, "tomato": 10},
        )
        self.assertEqual(seed, "tomato")

    def test_late_summer_stops_buying_when_no_harvest_fits(self) -> None:
        from harvest.planner.crop_planner import should_buy_seeds_for_date

        self.assertTrue(should_buy_seeds_for_date("summer", 12))
        self.assertFalse(should_buy_seeds_for_date("summer", 28))

    def test_build_planting_steps_keeps_planting_separate_from_watering(self) -> None:
        ram = _blank_ram()
        plan = plan_crop_field(
            ram,
            CropPlanningConfig(seed_type="potato", max_seed_bags=1, bounds=(10, 28, 25, 40)),
        )

        steps = build_planting_steps(plan)

        self.assertEqual(len([step for step in steps if step.action == "hoe"]), 8)
        self.assertEqual(steps[-1].action, "plant_seed")
        self.assertEqual(steps[-1].stand_tile, plan.plots[0].center)
        self.assertFalse(any(step.tool == "watering_can" for step in steps))

    def test_extracts_seed_template_from_plan_new_crops_recording(self) -> None:
        template = extract_planting_template_from_recording("tasks/plan_new_crops.json")

        self.assertEqual(template.name, "plan_new_crops")
        self.assertEqual(template.frame_count, 10070)
        self.assertIn((23, 35), template.seed_action_tiles)
        self.assertIn((19, 34), template.seed_action_tiles)
        self.assertGreaterEqual(len(template.hoe_action_tiles), 20)

    def test_extracts_seed_template_from_summer_repair_recording(self) -> None:
        template = extract_planting_template_from_recording("tasks/repair_crops.json")

        self.assertEqual(template.name, "repair_crops")
        self.assertEqual(template.frame_count, 10346)
        self.assertGreater(len(template.visited_farm_tiles), 100)
        self.assertTrue(
            {
                (13, 25),
                (13, 29),
                (7, 35),
                (3, 35),
                (3, 39),
                (3, 23),
            }.issubset(set(template.seed_action_tiles))
        )
        self.assertGreaterEqual(len(template.hoe_action_tiles), 20)


if __name__ == "__main__":
    unittest.main()
