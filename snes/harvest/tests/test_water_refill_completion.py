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


if __name__ == "__main__":
    unittest.main()
