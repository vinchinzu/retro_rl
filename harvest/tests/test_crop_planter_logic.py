from __future__ import annotations

import unittest
from types import SimpleNamespace

import numpy as np

from harvest.tasks.crop_planter import (
    CropWaterTask,
    ADDR_WEATHER,
    ADDR_WEATHER_FLAGS,
    build_water_steps,
    carry_pair_items,
    detect_crop_resume_plots,
    seed_item_in_carry_pair,
    tile_can_be_water_target,
    tile_is_watered,
    tile_needs_watering,
    watering_can_in_carry_pair,
    _merge_plot_centers,
)
from harvest.tasks.farm_clearer import ADDR_INPUT_LOCK, ADDR_MAP, ADDR_TILEMAP, ADDR_TOOL, ADDR_X, ADDR_Y, MAP_WIDTH
from retro_harness import TaskStatus


def _blank_ram() -> np.ndarray:
    return np.zeros(ADDR_MAP + MAP_WIDTH * MAP_WIDTH, dtype=np.uint8)


def _set_tile(ram: np.ndarray, tx: int, ty: int, tile_id: int) -> None:
    ram[ADDR_MAP + ty * MAP_WIDTH + tx] = tile_id


class CropPlanterLogicTests(unittest.TestCase):
    def test_watering_can_check_uses_current_carry_pair(self) -> None:
        ram = _blank_ram()
        ram[ADDR_TOOL] = 0x07
        ram[0x0923] = 0x10

        self.assertEqual(carry_pair_items(ram), {0x07, 0x10})
        self.assertTrue(watering_can_in_carry_pair(ram))
        self.assertTrue(seed_item_in_carry_pair(ram, "potato"))

    def test_crop_task_continues_when_watering_can_not_in_carry_pair(self) -> None:
        """Seeds-only carry pair must still allow hoe/plant before watering."""
        ram = _blank_ram()
        ram[ADDR_TOOL] = 0x07
        ram[0x0923] = 0x0C
        ram[ADDR_TILEMAP] = 0x00
        world = SimpleNamespace(ram=ram, info={}, obs=None)
        task = CropWaterTask()
        task.reset(world)

        result = task.step(world)

        # Must not hard-fail for missing watering can while seeds are in hand.
        self.assertNotEqual(result.status, TaskStatus.FAILURE)
        self.assertNotEqual(result.reason, "watering can not in carry pair")

    def test_crop_task_succeeds_without_seed_tool_when_raining(self) -> None:
        ram = _blank_ram()
        ram[ADDR_WEATHER_FLAGS] = 0x02
        ram[ADDR_TOOL] = 0x10
        ram[0x0923] = 0x01
        world = SimpleNamespace(ram=ram, info={}, obs=None)
        task = CropWaterTask(seed_type="potato")
        task.reset(world)

        result = task.step(world)

        self.assertEqual(result.status, TaskStatus.SUCCESS)
        self.assertEqual(result.reason, "rain; seed tool 0x07 not in carry pair")

    def test_establish_mode_skips_water_after_no_plots(self) -> None:
        ram = _blank_ram()
        # No seeds in carry or stock → establish pass has nothing to plant.
        ram[ADDR_TOOL] = 0x02  # hoe only
        ram[0x0923] = 0x00
        ram[ADDR_TILEMAP] = 0x00
        ram[ADDR_INPUT_LOCK] = 1
        world = SimpleNamespace(ram=ram, info={}, obs=None)
        task = CropWaterTask(work_mode="establish", seed_type="potato")
        task.reset(world)

        result = task.step(world)

        self.assertEqual(result.status, TaskStatus.SUCCESS)
        self.assertIn("no plots", result.reason or "")

    def test_water_only_mode_does_not_plan_new_plots(self) -> None:
        ram = _blank_ram()
        ram[ADDR_TOOL] = 0x10  # watering can
        ram[0x0923] = 0x00
        ram[ADDR_TILEMAP] = 0x00
        ram[ADDR_INPUT_LOCK] = 1
        world = SimpleNamespace(ram=ram, info={}, obs=None)
        task = CropWaterTask(work_mode="water")
        task.reset(world)

        result = task.step(world)

        self.assertEqual(result.status, TaskStatus.SUCCESS)
        self.assertIn("water-only", result.reason or "")

    def test_water_only_never_enters_plant_or_hoe(self) -> None:
        ram = _blank_ram()
        # Dry potato crop tiles around a center so detect finds a plot.
        center = (10, 40)
        for dy in range(-1, 2):
            for dx in range(-1, 2):
                _set_tile(ram, center[0] + dx, center[1] + dy, 0x54)
        ram[ADDR_TOOL] = 0x07  # seed, not can
        ram[0x0923] = 0x02
        ram[ADDR_TILEMAP] = 0x00
        px = center[0] * 16 + 8
        py = center[1] * 16 + 8
        ram[ADDR_X] = px & 0xFF
        ram[ADDR_X + 1] = (px >> 8) & 0xFF
        ram[ADDR_Y] = py & 0xFF
        ram[ADDR_Y + 1] = (py >> 8) & 0xFF
        ram[ADDR_INPUT_LOCK] = 1
        world = SimpleNamespace(ram=ram, info={}, obs=None)
        task = CropWaterTask(work_mode="water", bounds=(3, 34, 20, 50))
        task.reset(world)

        for _ in range(40):
            result = task.step(world)
            if result.status != TaskStatus.RUNNING:
                break
        # Water-only must not start planting even with seeds in hand.
        self.assertNotEqual(task._plot_phase, "plant")
        self.assertNotEqual(task._plot_phase, "hoe")

    def test_dry_wet_crop_pairs_are_classified_correctly(self) -> None:
        self.assertFalse(tile_needs_watering(0x07))
        self.assertTrue(tile_needs_watering(0x07, include_fresh_tilled=True))
        self.assertTrue(tile_needs_watering(0x54))
        self.assertTrue(tile_needs_watering(0x56))
        self.assertTrue(tile_needs_watering(0x58))
        self.assertTrue(tile_needs_watering(0x5A))
        self.assertTrue(tile_needs_watering(0x5E))
        self.assertTrue(tile_needs_watering(0x66))
        self.assertTrue(tile_needs_watering(0x68))

        self.assertTrue(tile_is_watered(0x08))
        self.assertTrue(tile_is_watered(0x55))
        self.assertTrue(tile_is_watered(0x59))
        self.assertTrue(tile_is_watered(0x5B))

        self.assertFalse(tile_needs_watering(0x08))
        self.assertFalse(tile_needs_watering(0x55))
        self.assertFalse(tile_needs_watering(0x59))
        self.assertFalse(tile_needs_watering(0x5B))
        self.assertFalse(tile_needs_watering(0x60))
        self.assertFalse(tile_needs_watering(0x61))
        self.assertFalse(tile_needs_watering(0x6E))
        self.assertFalse(tile_needs_watering(0x6F))

    def test_unknown_tiles_only_allowed_when_explicitly_requested(self) -> None:
        self.assertFalse(tile_can_be_water_target(0x01))
        self.assertFalse(tile_can_be_water_target(0x01, allow_unknown=True))
        self.assertFalse(tile_can_be_water_target(0x07))
        self.assertTrue(tile_can_be_water_target(0x07, include_fresh_tilled=True))
        self.assertFalse(tile_can_be_water_target(0x55, allow_unknown=True))
        self.assertFalse(tile_can_be_water_target(0x60, allow_unknown=True))
        self.assertTrue(tile_can_be_water_target(0x68, allow_unknown=True))
        self.assertFalse(tile_can_be_water_target(0x6E, allow_unknown=True))

    def test_detect_crop_resume_plots_suppresses_overlapping_centers(self) -> None:
        ram = _blank_ram()

        plot_tiles = {
            (13, 29): {
                (12, 28): 0x54,
                (13, 28): 0x54,
                (14, 28): 0x54,
                (12, 29): 0x54,
                (14, 29): 0x55,
                (12, 30): 0x54,
                (13, 30): 0x54,
                (14, 30): 0x55,
            },
            (17, 29): {
                (16, 28): 0x58,
                (17, 28): 0x58,
                (18, 28): 0x59,
                (16, 29): 0x58,
                (18, 29): 0x58,
                (16, 30): 0x58,
                (17, 30): 0x58,
                (18, 30): 0x59,
            },
            (3, 35): {
                (2, 34): 0x5A,
                (3, 34): 0x5A,
                (4, 34): 0x5B,
                (2, 35): 0x5A,
                (3, 35): 0x5A,
                (2, 36): 0x5A,
                (3, 36): 0x5A,
                (4, 36): 0x5B,
            },
        }
        for tiles in plot_tiles.values():
            for (tx, ty), tile_id in tiles.items():
                _set_tile(ram, tx, ty, tile_id)

        plots = detect_crop_resume_plots(ram, bounds=(1, 27, 20, 38))

        self.assertEqual(plots, [(13, 29), (17, 29), (3, 35)])

    def test_detect_crop_resume_plots_keeps_diagonal_plot_members(self) -> None:
        ram = _blank_ram()
        for tile in [(14, 34), (15, 34), (16, 34), (15, 35)]:
            _set_tile(ram, *tile, 0x57)
        _set_tile(ram, 14, 36, 0x56)

        plots = detect_crop_resume_plots(ram, bounds=(13, 33, 17, 37))

        self.assertEqual(plots, [(15, 35)])

    def test_merge_plot_centers_keeps_far_supplemental_plots(self) -> None:
        merged = _merge_plot_centers(
            primary=[(13, 29), (17, 29)],
            secondary=[(13, 25), (15, 29), (17, 29), (3, 35)],
        )

        self.assertEqual(merged, [(13, 29), (17, 29), (13, 25), (3, 35)])

    def test_build_water_steps_prefers_perimeter_stands_for_resume_plot(self) -> None:
        ram = _blank_ram()

        # Plot at x=2..4, y=34..36 with a notch/open stand at (4,35) and path
        # tiles on the left edge. This mirrors the left-side field near the
        # shipping area where walking through grown crops is invalid.
        tiles = {
            (1, 34): 0xA8,
            (1, 35): 0xA8,
            (1, 36): 0xA8,
            (2, 34): 0x5A,
            (3, 34): 0x5A,
            (4, 34): 0x5B,
            (2, 35): 0x5A,
            (3, 35): 0x5B,
            (4, 35): 0x02,
            (2, 36): 0x5A,
            (3, 36): 0x5A,
            (4, 36): 0x5B,
        }
        for (tx, ty), tile_id in tiles.items():
            _set_tile(ram, tx, ty, tile_id)

        steps = build_water_steps(ram, center=(3, 35), allow_crop_walkable=False)
        step_map = {target: (stand, face) for target, stand, face in steps}

        self.assertEqual(step_map[(2, 35)], ((1, 35), "right"))
        stand_234, face_234 = step_map[(2, 34)]
        self.assertIn(face_234, {"right", "down"})
        self.assertEqual(abs(stand_234[0] - 2) + abs(stand_234[1] - 34), 1)
        self.assertNotIn(stand_234, {(2, 34), (3, 34), (2, 35), (2, 36), (3, 36)})

        stand_334, face_334 = step_map[(3, 34)]
        self.assertIn(face_334, {"down", "left"})
        self.assertEqual(abs(stand_334[0] - 3) + abs(stand_334[1] - 34), 1)
        self.assertNotIn(stand_334, {(2, 34), (3, 34), (4, 34), (3, 35)})
        self.assertNotIn((3, 35), step_map)  # already watered (0x5B)

    def test_build_water_steps_prefers_nearest_remaining_steps_from_start_tile(self) -> None:
        ram = _blank_ram()

        for ty in range(23, 28):
            for tx in range(11, 16):
                _set_tile(ram, tx, ty, 0xA1)

        for tx in range(12, 15):
            for ty in range(24, 27):
                _set_tile(ram, tx, ty, 0x5A)

        steps = build_water_steps(
            ram,
            center=(13, 25),
            allow_crop_walkable=False,
            start_tile=(19, 30),
        )

        self.assertEqual(steps[0][0][1], 26)
        self.assertEqual(steps[0][1][1], 27)

    def test_build_water_steps_skips_known_harvestable_tiles(self) -> None:
        ram = _blank_ram()
        for ty in range(34, 37):
            for tx in range(2, 5):
                _set_tile(ram, tx, ty, 0x5A)
        for tile in [(1, 35), (3, 33), (5, 35), (3, 37)]:
            _set_tile(ram, *tile, 0xA1)

        steps = build_water_steps(ram, center=(3, 35), skip_tiles={(3, 35), (4, 35)})
        targets = {target for target, _stand, _face in steps}

        self.assertNotIn((3, 35), targets)
        self.assertNotIn((4, 35), targets)

    def test_build_water_steps_skips_unplanted_and_mature_tiles(self) -> None:
        ram = _blank_ram()
        for ty in range(33, 38):
            for tx in range(1, 6):
                _set_tile(ram, tx, ty, 0xA1)
        _set_tile(ram, 2, 34, 0x07)  # hoed but not necessarily planted
        _set_tile(ram, 3, 34, 0x01)  # untilled
        _set_tile(ram, 4, 34, 0x60)  # mature potato
        _set_tile(ram, 2, 35, 0x61)  # watered mature potato
        _set_tile(ram, 3, 35, 0x5A)  # unripe dry crop

        steps = build_water_steps(ram, center=(3, 35))

        self.assertEqual({target for target, _stand, _face in steps}, {(3, 35)})

    def test_crop_task_chooses_reachable_alternate_water_stand(self) -> None:
        ram = _blank_ram()
        ram[ADDR_TILEMAP] = 0x00
        for ty in range(64):
            for tx in range(64):
                _set_tile(ram, tx, ty, 0xFF)

        center = (17, 29)
        target = (16, 28)
        _set_tile(ram, *target, 0x56)
        _set_tile(ram, 16, 27, 0x02)  # preferred by face order, but isolated
        for tile in [(20, 27), (19, 27), (18, 27), (18, 28), (17, 28)]:
            _set_tile(ram, *tile, 0x02)

        world = SimpleNamespace(ram=ram, info={}, obs=None)
        task = CropWaterTask()
        task.reset(world)
        task._plots = [center]
        task._plot_index = 0
        task._plot_phase = "water"

        best = task._best_water_variant(ram, target, current_tile=(20, 27))

        self.assertIsNotNone(best)
        stand, face, _score = best
        self.assertEqual((stand, face), ((17, 28), "left"))

    def test_crop_task_does_not_seed_partial_live_crop_plot(self) -> None:
        ram = _blank_ram()
        for ty in range(64):
            for tx in range(64):
                _set_tile(ram, tx, ty, 0xA1)
        _set_tile(ram, 4, 4, 0x07)
        _set_tile(ram, 5, 4, 0x5A)
        _set_tile(ram, 4, 5, 0x5A)

        world = SimpleNamespace(ram=ram, info={}, obs=None)
        task = CropWaterTask()
        task.reset(world)
        task._plots = [(5, 5)]
        task._plot_index = 0

        task._start_plot(ram)

        self.assertEqual(task._plot_phase, "water")
        self.assertNotEqual(task._target_tile, (5, 5))

    def test_crop_task_accepts_alternate_adjacent_water_stand(self) -> None:
        ram = _blank_ram()
        for ty in range(64):
            for tx in range(64):
                _set_tile(ram, tx, ty, 0xA1)
        _set_tile(ram, 5, 5, 0x5A)
        ram[ADDR_TOOL] = int(0x10)
        ram[0x0923] = 0x07
        ram[ADDR_INPUT_LOCK] = 1
        ram[ADDR_MAP + 5 * MAP_WIDTH + 4] = 0xA1
        ram[ADDR_MAP + 5 * MAP_WIDTH + 6] = 0xA1
        ram[ADDR_MAP + 4 * MAP_WIDTH + 5] = 0xA1

        ram[ADDR_X] = 4 * 16 + 8
        ram[ADDR_X + 1] = 0
        ram[ADDR_Y] = 5 * 16 + 8
        ram[ADDR_Y + 1] = 0

        world = SimpleNamespace(ram=ram, info={}, obs=None)
        task = CropWaterTask()
        task.reset(world)
        task._plots = [(5, 5)]
        task._plot_index = 0
        task._plot_phase = "water"
        task._state = "navigate"
        task._water_steps = [((5, 5), (5, 4), "down")]
        task._water_index = 0
        task._target_tile = (5, 5)
        task._approach_tile = (5, 4)
        task._face_direction = "down"
        task._navigator.update(ram)

        task.step(world)

        self.assertEqual(task._approach_tile, (4, 5))
        self.assertEqual(task._face_direction, "right")
        self.assertEqual(task._state, "center")

    def test_crop_task_keeps_existing_water_path_until_repath_is_needed(self) -> None:
        ram = _blank_ram()
        for ty in range(64):
            for tx in range(64):
                _set_tile(ram, tx, ty, 0xA1)
        _set_tile(ram, 5, 5, 0x5A)
        ram[ADDR_TOOL] = int(0x10)
        ram[0x0923] = 0x07
        ram[ADDR_INPUT_LOCK] = 1
        ram[ADDR_MAP + 5 * MAP_WIDTH + 4] = 0xA1
        ram[ADDR_MAP + 5 * MAP_WIDTH + 6] = 0xA1
        ram[ADDR_MAP + 4 * MAP_WIDTH + 5] = 0xA1

        ram[ADDR_X] = 3 * 16 + 8
        ram[ADDR_X + 1] = 0
        ram[ADDR_Y] = 5 * 16 + 8
        ram[ADDR_Y + 1] = 0

        world = SimpleNamespace(ram=ram, info={}, obs=None)
        task = CropWaterTask()
        task.reset(world)
        task._plots = [(5, 5)]
        task._plot_index = 0
        task._plot_phase = "water"
        task._state = "navigate"
        task._water_steps = [((5, 5), (5, 4), "down")]
        task._water_index = 0
        task._target_tile = (5, 5)
        task._approach_tile = (5, 4)
        task._face_direction = "down"
        task._navigator.update(ram)
        task._navigator.path = [(4, 5), (5, 5), (5, 4)]

        task.step(world)

        self.assertEqual(task._approach_tile, (5, 4))
        self.assertEqual(task._face_direction, "down")

    def test_resume_water_phase_does_not_enable_crop_walkability(self) -> None:
        ram = _blank_ram()
        for ty in range(64):
            for tx in range(64):
                _set_tile(ram, tx, ty, 0xA1)
        for tx in range(12, 15):
            for ty in range(24, 27):
                _set_tile(ram, tx, ty, 0x5A)
        _set_tile(ram, 13, 25, 0x5B)

        world = SimpleNamespace(ram=ram, info={}, obs=None)
        task = CropWaterTask()
        task.reset(world)
        task._plots = [(13, 25)]
        task._plot_index = 0

        task._begin_water_phase(ram, allow_unknown_tiles=False)

        self.assertFalse(task._allow_crop_walkable)
        self.assertEqual(task._pathfinder.extra_walkable, set())
        self.assertNotIn(task._approach_tile, task._current_plot_tiles())

    def test_crop_task_hotswap_resume_rescans_after_refill_timeout(self) -> None:
        ram = _blank_ram()
        for ty in range(64):
            for tx in range(64):
                _set_tile(ram, tx, ty, 0xA1)
        for tx in range(12, 15):
            for ty in range(24, 27):
                _set_tile(ram, tx, ty, 0x5A)
        ram[ADDR_TILEMAP] = 0x00
        ram[ADDR_TOOL] = 0x10
        ram[0x0923] = 0x07
        ram[0x0926] = 0
        world = SimpleNamespace(ram=ram, info={}, obs=None)
        task = CropWaterTask()
        task.reset(world)
        task._plots = [(13, 25)]
        task._plot_index = 0
        task._plot_phase = "refill"
        task._state = "navigate"
        task._water_steps = [((12, 24), (12, 23), "down")]
        task._water_index = 0
        task._target_tile = (9, 28)
        task._approach_tile = (9, 28)
        task._refill_pond_tile = (9, 28)
        task._bad_refill_tiles = {(9, 28)}
        task._refill_exhausted = True
        task._steps_on_target = task.max_steps_per_target + 1

        task.resume_after_hotswap(world)

        self.assertEqual(task._state, "detect")
        self.assertEqual(task._plot_phase, "plant")
        self.assertEqual(task._bad_refill_tiles, set())
        self.assertFalse(task._refill_exhausted)
        self.assertEqual(task._steps_on_target, 0)
        self.assertEqual(task._water_steps, [])


if __name__ == "__main__":
    unittest.main()
