from __future__ import annotations

import unittest

import numpy as np

from harvest.planner.crop_planner import (
    CROP_LAYOUTS,
    CROP_SPECS,
    DEFAULT_SHIPPING_TILE,
    POND_CORRIDOR_TILES,
    CropPlanningConfig,
    build_planting_steps,
    choose_crop_for_date,
    evaluate_plot_candidate,
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

    def test_planner_avoids_shipping_stand_and_plants_potato_away_from_bin(self) -> None:
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
        near_bin = (13, 29)
        potato_dist = abs(plan.plots[0].center[0] - DEFAULT_SHIPPING_TILE[0]) + abs(
            plan.plots[0].center[1] - DEFAULT_SHIPPING_TILE[1]
        )
        near_dist = abs(near_bin[0] - DEFAULT_SHIPPING_TILE[0]) + abs(
            near_bin[1] - DEFAULT_SHIPPING_TILE[1]
        )
        self.assertGreater(potato_dist, near_dist)
        for plot in plan.plots:
            self.assertNotIn(DEFAULT_SHIPPING_TILE, plot.crop_tiles)
            self.assertNotIn(DEFAULT_SHIPPING_TILE, plot.water_stands)

    def test_corn_prefers_plots_closer_to_the_bin_than_potato(self) -> None:
        ram = _blank_ram()
        bounds = (10, 28, 25, 40)
        potato = plan_crop_field(
            ram,
            CropPlanningConfig(
                seed_type="potato",
                day=1,
                max_seed_bags=1,
                bounds=bounds,
                shipping_tile=DEFAULT_SHIPPING_TILE,
            ),
        )
        corn = plan_crop_field(
            ram,
            CropPlanningConfig(
                seed_type="corn",
                season="summer",
                day=1,
                max_seed_bags=1,
                bounds=bounds,
                shipping_tile=DEFAULT_SHIPPING_TILE,
            ),
        )
        def _dist(plan):
            c = plan.plots[0].center
            return abs(c[0] - DEFAULT_SHIPPING_TILE[0]) + abs(c[1] - DEFAULT_SHIPPING_TILE[1])

        self.assertGreater(_dist(potato), _dist(corn))

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

    def test_late_spring_potato_still_plants_because_it_harvests_in_summer(self) -> None:
        ram = _blank_ram()
        config = CropPlanningConfig(
            seed_type="potato",
            season="spring",
            day=28,
            max_seed_bags=1,
            bounds=(10, 28, 25, 40),
        )

        plan = plan_crop_field(ram, config)

        self.assertEqual(plan.seed_bags_needed, 1)
        self.assertEqual(CROP_SPECS["potato"].harvests_from_planting_day(28), 1)

    def test_choose_crop_for_date_accounts_for_summer_regrow(self) -> None:
        crop = choose_crop_for_date("summer", 12, layout_tiles=7)

        self.assertEqual(crop.name, "corn")
        self.assertGreater(crop.harvests_from_planting_day(12), 1)

    def test_fall_has_eggplant_and_winter_has_no_plantable_crops(self) -> None:
        from harvest.planner.crop_planner import (
            is_crop_planting_season,
            resolve_seed_type_for_date,
            should_buy_seeds_for_date,
            choose_crop_for_date,
        )

        self.assertTrue(is_crop_planting_season("fall"))
        self.assertFalse(is_crop_planting_season("winter"))
        crop = choose_crop_for_date("fall", 1)
        self.assertIsNotNone(crop)
        assert crop is not None
        self.assertEqual(crop.name, "eggplant")
        self.assertEqual(resolve_seed_type_for_date("fall", 5), "eggplant")
        self.assertTrue(should_buy_seeds_for_date("fall", 5))
        self.assertIsNone(resolve_seed_type_for_date("winter", 10))
        self.assertFalse(should_buy_seeds_for_date("winter", 5))

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
        from harvest.paths import PROJECT_DIR

        template = extract_planting_template_from_recording(
            str(PROJECT_DIR / "tasks" / "plan_new_crops.json")
        )

        self.assertEqual(template.name, "plan_new_crops")
        self.assertEqual(template.frame_count, 10070)
        self.assertIn((23, 35), template.seed_action_tiles)
        self.assertIn((19, 34), template.seed_action_tiles)
        self.assertGreaterEqual(len(template.hoe_action_tiles), 20)

    def test_extracts_seed_template_from_summer_repair_recording(self) -> None:
        from harvest.paths import PROJECT_DIR

        template = extract_planting_template_from_recording(
            str(PROJECT_DIR / "tasks" / "repair_crops.json")
        )

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


class SecondPlotPlacementTests(unittest.TestCase):
    """rr-20w.3 D3: score a second potato ring with the D2 ring protected."""

    def _d2_ring(self):
        return tuple(
            (13 + dx, 28 + dy) for dy in (-1, 0, 1) for dx in (-1, 0, 1)
        )

    def test_protected_d2_ring_is_never_reused(self) -> None:
        ram = _blank_ram()
        protected = self._d2_ring()
        config = CropPlanningConfig(
            seed_type="potato",
            day=3,
            max_seed_bags=1,
            bounds=(10, 24, 24, 30),
            protected_tiles=protected,
        )

        plan = plan_crop_field(ram, config)

        self.assertEqual(plan.seed_bags_needed, 1)
        chosen = plan.plots[0]
        self.assertFalse(set(chosen.crop_tiles) & set(protected))
        self.assertNotIn((13, 28), chosen.crop_tiles)
        # Potato goes farther from the bin than the D2 ring so summer 3-day
        # corn/tomato can occupy the close remainder.
        d2 = (13, 28)
        chosen_dist = abs(chosen.center[0] - DEFAULT_SHIPPING_TILE[0]) + abs(
            chosen.center[1] - DEFAULT_SHIPPING_TILE[1]
        )
        d2_dist = abs(d2[0] - DEFAULT_SHIPPING_TILE[0]) + abs(
            d2[1] - DEFAULT_SHIPPING_TILE[1]
        )
        self.assertGreater(chosen_dist, d2_dist)

    def test_candidate_overlapping_the_d2_ring_is_rejected(self) -> None:
        ram = _blank_ram()
        eight = CROP_LAYOUTS["eight_tile_ring"]
        config = CropPlanningConfig(
            seed_type="potato", day=3, protected_tiles=self._d2_ring()
        )
        # (14,28) ring overlaps the protected D2 tiles → rejected.
        self.assertIsNone(
            evaluate_plot_candidate(
                ram, (14, 28), CROP_SPECS["potato"], eight, config
            )
        )
        # A clear neighbour to the east still scores.
        self.assertIsNotNone(
            evaluate_plot_candidate(
                ram, (19, 28), CROP_SPECS["potato"], eight, config
            )
        )


class PlotSitingPolicyTests(unittest.TestCase):
    """Corridor reject, one-shot vs regrow distance, clutter penalty."""

    def test_ring_on_pond_corridor_is_not_a_candidate(self) -> None:
        ram = _blank_ram()
        eight = CROP_LAYOUTS["eight_tile_ring"]
        config = CropPlanningConfig(seed_type="potato", day=1)
        corridor_center = (13, 33)
        self.assertTrue(
            (set(eight.crop_tiles(corridor_center)) | {corridor_center})
            & POND_CORRIDOR_TILES
        )
        self.assertIsNone(
            evaluate_plot_candidate(
                ram, corridor_center, CROP_SPECS["potato"], eight, config
            )
        )
        plan = plan_crop_field(
            ram,
            CropPlanningConfig(
                seed_type="potato",
                day=1,
                max_seed_bags=3,
                bounds=(10, 30, 20, 36),
            ),
        )
        for plot in plan.plots:
            footprint = set(plot.crop_tiles) | {plot.center} | set(plot.access_tiles)
            self.assertFalse(footprint & POND_CORRIDOR_TILES)

    def test_one_shot_crops_prefer_far_regrow_prefer_close(self) -> None:
        ram = _blank_ram()
        eight = CROP_LAYOUTS["eight_tile_ring"]
        near = (40, 20)
        far = (50, 45)
        potato_cfg = CropPlanningConfig(seed_type="potato", day=1)
        corn_cfg = CropPlanningConfig(seed_type="corn", season="summer", day=1)
        potato_near = evaluate_plot_candidate(
            ram, near, CROP_SPECS["potato"], eight, potato_cfg
        )
        potato_far = evaluate_plot_candidate(
            ram, far, CROP_SPECS["potato"], eight, potato_cfg
        )
        corn_near = evaluate_plot_candidate(
            ram, near, CROP_SPECS["corn"], eight, corn_cfg
        )
        corn_far = evaluate_plot_candidate(
            ram, far, CROP_SPECS["corn"], eight, corn_cfg
        )
        self.assertIsNotNone(potato_near)
        self.assertIsNotNone(potato_far)
        self.assertIsNotNone(corn_near)
        self.assertIsNotNone(corn_far)
        assert potato_near is not None and potato_far is not None
        assert corn_near is not None and corn_far is not None
        self.assertGreater(potato_far.score, potato_near.score)
        self.assertLess(potato_far.route_cost, potato_near.route_cost)
        self.assertGreater(corn_near.score, corn_far.score)
        self.assertLess(corn_near.route_cost, corn_far.route_cost)

        tomato_cfg = CropPlanningConfig(seed_type="tomato", season="summer", day=1)
        tomato_near = evaluate_plot_candidate(
            ram, near, CROP_SPECS["tomato"], eight, tomato_cfg
        )
        tomato_far = evaluate_plot_candidate(
            ram, far, CROP_SPECS["tomato"], eight, tomato_cfg
        )
        assert tomato_near is not None and tomato_far is not None
        self.assertGreater(tomato_near.score, tomato_far.score)

    def test_high_clutter_ring_scores_worse_than_open_field(self) -> None:
        ram_open = _blank_ram()
        ram_boxed = _blank_ram()
        center = (40, 20)
        # Corners of the 5x5 frame: Moore-adjacent to crop tiles, not cardinal
        # watering stands, so the ring stays legal.
        for tx, ty in ((38, 18), (42, 18), (38, 22), (42, 22)):
            _set_tile(ram_boxed, tx, ty, 0xA1)
        eight = CROP_LAYOUTS["eight_tile_ring"]
        config = CropPlanningConfig(seed_type="potato", day=1)
        open_c = evaluate_plot_candidate(
            ram_open, center, CROP_SPECS["potato"], eight, config
        )
        boxed_c = evaluate_plot_candidate(
            ram_boxed, center, CROP_SPECS["potato"], eight, config
        )
        self.assertIsNotNone(open_c)
        self.assertIsNotNone(boxed_c)
        assert open_c is not None and boxed_c is not None
        self.assertGreater(open_c.score, boxed_c.score)
        self.assertEqual(open_c.expected_profit_g, boxed_c.expected_profit_g)
        self.assertEqual(open_c.route_cost, boxed_c.route_cost)
        self.assertGreaterEqual(open_c.score - boxed_c.score, 25)


if __name__ == "__main__":
    unittest.main()
