"""Unit tests for named pond corridor refill selection + crop completion."""

from __future__ import annotations

import unittest
from types import SimpleNamespace

import numpy as np

from harvest.maps.map_config import (
    FARM_MAIN_POND_STANDS,
    FARM_POND_ACCESS_STAGING_TILES,
    FARM_POND_REFILL_CORRIDOR,
    farm_pond_refill_primary_stand,
    player_in_west_plant_pocket,
)
from harvest.tasks.crop_planter import CropWaterTask
from harvest.tasks.farm_clearer import ADDR_INPUT_LOCK, ADDR_MAP, ADDR_TILEMAP, ADDR_TOOL, ADDR_X, ADDR_Y, MAP_WIDTH
from harvest.tasks.skills import farm_nav_to_pond_refill_skill, farm_pond_refill_face
from harvest.tasks.water_refill import (
    corridor_needs_fence_open,
    crop_completion_status,
    is_no_work_reason,
    select_main_pond_refill,
    select_staging_stand,
)
from retro_harness import TaskStatus


def _blank_ram() -> np.ndarray:
    return np.zeros(ADDR_MAP + MAP_WIDTH * MAP_WIDTH, dtype=np.uint8)


def _set_tile(ram: np.ndarray, tx: int, ty: int, tile_id: int) -> None:
    ram[ADDR_MAP + ty * MAP_WIDTH + tx] = tile_id


def _set_player_tile(ram: np.ndarray, tile: tuple[int, int]) -> None:
    px = tile[0] * 16 + 8
    py = tile[1] * 16 + 8
    ram[ADDR_X] = px & 0xFF
    ram[ADDR_X + 1] = (px >> 8) & 0xFF
    ram[ADDR_Y] = py & 0xFF
    ram[ADDR_Y + 1] = (py >> 8) & 0xFF


class PondCorridorConfigTests(unittest.TestCase):
    def test_primary_stand_is_south_lip(self) -> None:
        stand, face = farm_pond_refill_primary_stand()
        self.assertEqual(stand, (32, 34))
        self.assertEqual(face, "up")
        self.assertEqual(FARM_MAIN_POND_STANDS[0], (stand, face))

    def test_corridor_steps_named(self) -> None:
        self.assertIn("stage_west_of_fence", FARM_POND_REFILL_CORRIDOR)
        self.assertIn("open_fence_row_y31", FARM_POND_REFILL_CORRIDOR)
        self.assertIn("fill_at_main_pond", FARM_POND_REFILL_CORRIDOR)

    def test_west_pocket_predicate(self) -> None:
        self.assertTrue(player_in_west_plant_pocket((13, 27)))
        self.assertTrue(player_in_west_plant_pocket((12, 29)))
        self.assertFalse(player_in_west_plant_pocket((32, 34)))
        self.assertFalse(player_in_west_plant_pocket((10, 40)))

    def test_pond_nav_skill_targets_primary_stand(self) -> None:
        skill = farm_nav_to_pond_refill_skill()
        self.assertEqual(skill.target_px, (32 * 16 + 8, 34 * 16 + 8))
        self.assertEqual(farm_pond_refill_face(), "up")


class RefillSelectionTests(unittest.TestCase):
    def test_select_main_pond_prefers_pathable_stand(self) -> None:
        def find_path(start, goal):
            if goal == (33, 30):
                return [start, goal]
            return None

        hit = select_main_pond_refill((20, 25), find_path)
        self.assertIsNotNone(hit)
        assert hit is not None
        self.assertEqual(hit.stand, (33, 30))
        self.assertEqual(hit.face, "down")
        self.assertEqual(hit.source, "main_pond_corridor")

    def test_select_main_pond_skips_bad_stands(self) -> None:
        def find_path(start, goal):
            return [start, goal]

        hit = select_main_pond_refill(
            (20, 25),
            find_path,
            bad_stands={(32, 34), (33, 34)},
        )
        self.assertIsNotNone(hit)
        assert hit is not None
        self.assertEqual(hit.stand, (33, 30))

    def test_select_staging_from_west_pocket(self) -> None:
        def find_path(start, goal):
            if goal == (12, 29):
                return [start, goal]
            return None

        hit = select_staging_stand((13, 27), find_path)
        self.assertIsNotNone(hit)
        assert hit is not None
        self.assertEqual(hit.stand, (12, 29))
        self.assertEqual(hit.source, "staging")

    def test_corridor_needs_fence_when_pond_blocked(self) -> None:
        def no_path(start, goal):
            return None

        self.assertTrue(
            corridor_needs_fence_open(
                (13, 27),
                no_path,
                blocking_fences=[(15, 31), (16, 31)],
            )
        )
        self.assertFalse(
            corridor_needs_fence_open(
                (13, 27),
                no_path,
                blocking_fences=[],
            )
        )

    def test_start_refill_commits_main_pond_when_pathable(self) -> None:
        ram = _blank_ram()
        ram[ADDR_TILEMAP] = 0x00
        ram[ADDR_INPUT_LOCK] = 1
        ram[ADDR_TOOL] = 0x10
        # Open farm dirt so pathfinder can walk west pocket → pond.
        for ty in range(20, 40):
            for tx in range(10, 40):
                _set_tile(ram, tx, ty, 0x01)
        # F0 pond water cells that stands face (south lip face up → y=33).
        for ty in range(31, 34):
            for tx in range(31, 35):
                _set_tile(ram, tx, ty, 0xF0)
        _set_player_tile(ram, (20, 28))
        world = SimpleNamespace(ram=ram, info={}, obs=None)
        task = CropWaterTask(work_mode="water", refill_bounds=(3, 14, 62, 60))
        task.reset(world)
        # Seed a water step so refill has remaining work.
        task._water_steps = [((12, 25), (12, 26), "up")]
        task._water_index = 0
        task._plot_phase = "water"
        task._plots = [(12, 25)]
        task._plot_index = 0

        task._start_refill(ram)

        self.assertEqual(task._plot_phase, "refill")
        self.assertIn(task._refill_pond_tile, {s for s, _ in FARM_MAIN_POND_STANDS})
        self.assertEqual(task._state, "navigate")
        # Stand must face preferred fill water on the live map.
        from harvest.tasks.crop_planter import edge_water_tile_id, REFILL_PREFERRED_WATER_TILES

        wid = edge_water_tile_id(ram, task._refill_pond_tile, task._refill_pond_face)
        self.assertIn(wid, REFILL_PREFERRED_WATER_TILES)


class CropCompletionTests(unittest.TestCase):
    def test_no_work_reason_helper(self) -> None:
        self.assertTrue(is_no_work_reason("no_work: water-only; no dry crop tiles"))
        self.assertTrue(is_no_work_reason("no_work"))
        self.assertFalse(is_no_work_reason("planted=1 watered=3"))
        self.assertFalse(is_no_work_reason(None))

    def test_water_mode_fails_when_dry_crops_unwatered(self) -> None:
        status, reason = crop_completion_status(
            work_mode="water",
            planted=0,
            watered=0,
            dry_at_start=3,
            refill_exhausted=False,
            had_seed_stock=False,
        )
        self.assertEqual(status, "failure")
        self.assertIn("dry_crops=3", reason)

    def test_water_mode_fails_on_refill_exhausted(self) -> None:
        status, reason = crop_completion_status(
            work_mode="water",
            planted=0,
            watered=0,
            dry_at_start=2,
            refill_exhausted=True,
            had_seed_stock=False,
        )
        self.assertEqual(status, "failure")
        self.assertIn("refill exhausted", reason)

    def test_water_mode_no_work_when_nothing_dry(self) -> None:
        status, reason = crop_completion_status(
            work_mode="water",
            planted=0,
            watered=0,
            dry_at_start=0,
            refill_exhausted=False,
            had_seed_stock=False,
        )
        self.assertEqual(status, "no_work")
        self.assertTrue(is_no_work_reason(reason))

    def test_water_mode_success_when_watered(self) -> None:
        status, reason = crop_completion_status(
            work_mode="water",
            planted=0,
            watered=3,
            dry_at_start=3,
            refill_exhausted=False,
            had_seed_stock=False,
        )
        self.assertEqual(status, "success")
        self.assertIn("watered=3", reason)

    def test_establish_fails_with_seed_but_no_plant(self) -> None:
        status, reason = crop_completion_status(
            work_mode="establish",
            planted=0,
            watered=0,
            dry_at_start=0,
            refill_exhausted=False,
            had_seed_stock=True,
        )
        self.assertEqual(status, "failure")
        self.assertIn("planted=0", reason)

    def test_crop_task_fails_water_only_with_dry_tiles_and_no_progress(self) -> None:
        ram = _blank_ram()
        center = (12, 25)
        for dy in range(-1, 2):
            for dx in range(-1, 2):
                _set_tile(ram, center[0] + dx, center[1] + dy, 0x54)
        # Wall off everything so water steps cannot progress.
        for ty in range(14, 40):
            for tx in range(3, 40):
                if abs(tx - center[0]) > 1 or abs(ty - center[1]) > 1:
                    _set_tile(ram, tx, ty, 0x05)
        ram[ADDR_TILEMAP] = 0x00
        ram[ADDR_INPUT_LOCK] = 1
        ram[ADDR_TOOL] = 0x10
        _set_player_tile(ram, center)
        world = SimpleNamespace(ram=ram, info={}, obs=None)
        task = CropWaterTask(
            work_mode="water",
            bounds=(3, 14, 30, 40),
            max_steps_per_target=5,
            max_failures=3,
        )
        task.reset(world)

        result = None
        for _ in range(80):
            result = task.step(world)
            if result.status != TaskStatus.RUNNING:
                break

        self.assertIsNotNone(result)
        assert result is not None
        # Either failure (dry remain) or timeout-ish failure — not silent SUCCESS.
        if result.status == TaskStatus.SUCCESS:
            self.assertTrue(
                is_no_work_reason(result.reason),
                msg=f"unexpected success reason: {result.reason}",
            )
            # no_work only valid if we never saw dry tiles
            self.assertEqual(task._dry_crop_tiles_at_start, 0)
        else:
            self.assertEqual(result.status, TaskStatus.FAILURE)


class SameDayEstablishWaterOrderTests(unittest.TestCase):
    """Day-plan crop pass order for same-day plant then water (rr-e1p)."""

    def test_crop_establish_then_water_phases_order(self) -> None:
        from harvest.planner.day_plan_phases import crop_establish_phases, crop_water_phases

        establish = [p.phase for p in crop_establish_phases()]
        water = [p.phase for p in crop_water_phases()]
        self.assertIn("CROP_ESTABLISH", establish)
        self.assertIn("ENSURE_CROP_SEEDS", establish)
        self.assertNotIn("ENSURE_WATERING_CAN", establish)
        self.assertIn("ENSURE_WATERING_CAN", water)
        self.assertIn("CROP_WATER", water)
        # Plant pass before water pass when both are scheduled.
        from harvest.planner.day_plan_phases import _crop_work_phases

        from harvest.planner.day_phase_types import DayPlannerPolicy

        phases = _crop_work_phases(
            has_harvest=False,
            has_waterable=True,
            has_seeds=True,
            is_rainy=False,
            late_day=False,
            policy=DayPlannerPolicy(),
        )
        names = [p.phase for p in phases]
        if "CROP_ESTABLISH" in names and "CROP_WATER" in names:
            self.assertLess(names.index("CROP_ESTABLISH"), names.index("CROP_WATER"))
            self.assertLess(
                names.index("CROP_ESTABLISH"),
                names.index("ENSURE_WATERING_CAN"),
            )


class KeepAliveClearOrderTests(unittest.TestCase):
    """CLEAR_FIELD must not starve crop keep-alive water (rr-3v9)."""

    def test_outdoor_water_before_clear_when_dry_crops(self) -> None:
        from harvest.planner.day_plan_phases import build_outdoor_day_phases

        phases = build_outdoor_day_phases(
            weekday=3,
            hour=6,
            has_harvest=False,
            has_waterable=True,
            has_seeds=False,
            has_debris=True,
            is_rainy=False,
        )
        names = [p.phase for p in phases]
        self.assertIn("CROP_WATER", names)
        self.assertIn("CLEAR_FIELD", names)
        self.assertLess(names.index("CROP_WATER"), names.index("CLEAR_FIELD"))
        self.assertLess(names.index("ENSURE_WATERING_CAN"), names.index("CLEAR_FIELD"))

    def test_outdoor_clear_before_water_when_no_crops(self) -> None:
        from harvest.planner.day_plan_phases import build_outdoor_day_phases

        phases = build_outdoor_day_phases(
            weekday=3,
            hour=6,
            has_harvest=False,
            has_waterable=False,
            has_seeds=False,
            has_debris=True,
            is_rainy=False,
        )
        names = [p.phase for p in phases]
        self.assertIn("CLEAR_FIELD", names)
        self.assertNotIn("CROP_WATER", names)

    def test_full_day_water_before_clear_when_dry_crops(self) -> None:
        from harvest.planner.day_plan_phases import build_day_phases
        from harvest.planner.day_phase_types import DayPlannerPolicy

        phases = build_day_phases(
            None,
            weekday=3,
            hour=6,
            has_chickens=False,
            has_cows=False,
            has_harvest=False,
            has_waterable=True,
            has_seeds=False,
            has_debris=True,
            policy=DayPlannerPolicy(
                include_chickens=False,
                include_cows=False,
                include_shop_run=False,
                include_berry_run=False,
                include_end_day=False,
            ),
        )
        names = [p.phase for p in phases]
        self.assertIn("CROP_WATER", names)
        self.assertIn("CLEAR_FIELD", names)
        self.assertLess(names.index("CROP_WATER"), names.index("CLEAR_FIELD"))


if __name__ == "__main__":
    unittest.main()
