"""Unit tests for pond refill stand selection + start_refill commit."""

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

from harvest.maps.map_config import FARM_MAIN_POND_STANDS
from harvest.tasks.crop_planter import CropWaterTask
from harvest.core.tile_catalog import (
    ADDR_INPUT_LOCK,
    ADDR_TILEMAP,
    ADDR_TOOL,
)
from harvest.tasks.water_refill import (
    corridor_needs_fence_open,
    select_main_pond_refill,
    select_staging_stand,
)


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
        # Nearest remaining north-lip stand from (20,25) among corridor list.
        self.assertIn(hit.stand, {(32, 30), (33, 30), (34, 30)})
        self.assertNotIn(hit.stand, {(32, 34), (33, 34)})

    def test_select_main_pond_prefers_nearest_pathable(self) -> None:
        """North of pond: north lip must beat far south lip when both pathable."""

        def find_path(start, goal):
            return [start, goal]

        hit = select_main_pond_refill((33, 28), find_path)
        self.assertIsNotNone(hit)
        assert hit is not None
        self.assertEqual(hit.stand, (33, 30))
        self.assertEqual(hit.face, "down")

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
        task = CropWaterTask(work_mode="water", refill_bounds=(3, 10, 62, 60))
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


if __name__ == "__main__":
    unittest.main()
