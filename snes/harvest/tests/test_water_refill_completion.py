"""Unit tests for crop watering completion status."""

from __future__ import annotations

from pathlib import Path
import sys
import unittest
from types import SimpleNamespace

# Path-stable import of sibling helpers (works under unittest and pytest importlib mode).
_TESTS_DIR = Path(__file__).resolve().parent
if str(_TESTS_DIR) not in sys.path:
    sys.path.insert(0, str(_TESTS_DIR))
from water_refill_helpers import (
    _blank_ram,
    _set_player_tile,
    _set_tile,
)

from harvest.tasks.crop_planter import CropWaterTask
from harvest.core.tile_catalog import (
    ADDR_INPUT_LOCK,
    ADDR_TILEMAP,
    ADDR_TOOL,
)
from harvest.tasks.water_refill import (
    crop_completion_status,
    is_no_work_reason,
)
from retro_harness import TaskStatus


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
        # The blank RAM never turns the farmer, so wall verdicts wait out the
        # dropped-input cap (MAX_IGNORED_HOLD_FRAMES) before they land.
        for _ in range(900):
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

    def test_summer_farm_arms_sixteen_dry_tiles_without_left_map(self) -> None:
        """summer1: ``water fail: dry_crops=16 watered=0`` was not a skipped stand.

        The first armed stand was ``nav_water_13_28_0`` / ``nav_water_19_28_0``
        with ``require_tilemap=0x00``. Summer's farm is tilemap 0x01, so a
        strict compare returned ``left map 0x00 → 0x01`` and aborted the
        sequence before any tile was watered. ``tilemaps_match`` treats
        0x00–0x03 as one farm; the stand must stay running and face a dry crop.
        """
        from harvest.core.tile_catalog import ADDR_MAP
        from harvest.tasks.crop_planter import tile_needs_watering
        from harvest.tasks.nav import MAP_WIDTH

        ram = _blank_ram()
        for cx, cy in ((13, 28), (19, 28)):
            for dy in (-1, 0, 1):
                for dx in (-1, 0, 1):
                    tid = 0x01 if dx == 0 and dy == 0 else 0x54
                    _set_tile(ram, cx + dx, cy + dy, tid)
        ram[ADDR_TILEMAP] = 0x01
        ram[ADDR_INPUT_LOCK] = 1
        ram[ADDR_TOOL] = 0x10
        ram[0x0926] = 20
        _set_player_tile(ram, (8, 26))
        world = SimpleNamespace(ram=ram, info={}, obs=None)
        task = CropWaterTask(work_mode="water", bounds=(2, 3, 62, 60))
        task.reset(world)

        first = task.step(world)
        self.assertEqual(first.status, TaskStatus.RUNNING, first.reason)
        self.assertEqual(task._dry_crop_tiles_at_start, 16)
        self.assertEqual(task._plots, [(13, 28), (19, 28)])
        self.assertEqual(task._armed_water_tiles, 16)
        self.assertEqual(task.watered_count, 0)

        pair = task._water_pairs[0]
        nav, wet = pair.tasks
        self.assertEqual(nav.require_tilemap, 0x00)
        tx, ty = wet.target_tile
        self.assertTrue(tile_needs_watering(int(ram[ADDR_MAP + ty * MAP_WIDTH + tx])))
        sx, sy = nav.target_px[0] // 16, nav.target_px[1] // 16
        self.assertEqual(abs(sx - tx) + abs(sy - ty), 1)

        second = task.step(world)
        self.assertEqual(second.status, TaskStatus.RUNNING, second.reason)
        self.assertNotIn("left map", second.reason or "")
        self.assertFalse(getattr(pair, "skipped", False))

    def test_crop_task_reports_no_work_in_fall_and_winter(self) -> None:
        """No fall seed exists and Fall/Winter never grow a crop (ROM nightly check)."""
        from day_plan_test_helpers import make_date_world

        for season in (2, 3):
            world = make_date_world(0x00, season=season, day=3)
            _set_tile(world.ram, 12, 25, 0x4C)  # dry, unripe corn
            task = CropWaterTask(work_mode="full", bounds=(3, 14, 30, 40))
            task.reset(world)
            result = task.step(world)
            self.assertEqual(result.status, TaskStatus.SUCCESS)
            self.assertTrue(is_no_work_reason(result.reason), result.reason)


if __name__ == "__main__":
    unittest.main()
