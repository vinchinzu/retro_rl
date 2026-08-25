"""rr-20w.2.2 / rr-20w.2.10: travel BFS never enters weeds, stumps, or rocks."""
from __future__ import annotations

from pathlib import Path
import sys

_TESTS_DIR = Path(__file__).resolve().parent
if str(_TESTS_DIR) not in sys.path:
    sys.path.insert(0, str(_TESTS_DIR))
from day_plan_test_helpers import make_navigation_ram

import unittest

from harvest.core.tile_catalog import (
    ADDR_MAP,
    FENCE,
    LARGE_ROCK_DAMAGE_TILES,
    LARGE_ROCK_TILES,
    LARGE_ROCK_TL,
    MAP_WIDTH,
    ROCK,
    STONE,
    STUMP_TILES,
    TRAVEL_SOLID_TILES,
    WEED,
    debris_footprint,
)
from harvest.tasks.nav import Pathfinder, Point, TILE_SIZE
from harvest.tasks.travel_walk import is_travel_occupied, is_travel_solid


DIRT = 0xA1
ROCK_QUAD = ((12, 10), (13, 10), (12, 11), (13, 11))


def _set_tile(ram, tx: int, ty: int, tile_id: int) -> None:
    ram[ADDR_MAP + ty * MAP_WIDTH + tx] = tile_id


def _place_large_rock(ram, tx: int, ty: int) -> None:
    for (x, y), tid in zip(
        ((tx, ty), (tx + 1, ty), (tx, ty + 1), (tx + 1, ty + 1)),
        (0x0D, 0x0E, 0x0F, 0x10),
    ):
        _set_tile(ram, x, y, tid)


class TravelSolidDenylistTests(unittest.TestCase):
    def test_travel_solid_ids(self) -> None:
        expected = (
            {WEED, STONE, FENCE, ROCK}
            | STUMP_TILES
            | LARGE_ROCK_TILES
            | LARGE_ROCK_DAMAGE_TILES
        )
        self.assertEqual(TRAVEL_SOLID_TILES, expected)
        for tid in expected:
            self.assertTrue(is_travel_solid(tid))
        self.assertFalse(is_travel_solid(DIRT))

    def test_is_walkable_false_for_each_travel_solid(self) -> None:
        ram = make_navigation_ram(current_tile=(5, 5), blocked_tile=(63, 63))
        pf = Pathfinder()
        cell = (8, 8)
        for tid in sorted(TRAVEL_SOLID_TILES):
            with self.subTest(tile_id=hex(tid)):
                _set_tile(ram, *cell, tid)
                self.assertFalse(pf.is_walkable(ram, *cell))

    def test_extra_walkable_and_override_cannot_enter_solids(self) -> None:
        ram = make_navigation_ram(current_tile=(5, 5), blocked_tile=(63, 63))
        pf = Pathfinder()
        cell = (8, 8)
        _set_tile(ram, *cell, WEED)
        pf.extra_walkable.add(cell)
        self.assertFalse(pf.is_walkable(ram, *cell))
        self.assertFalse(pf.is_walkable(ram, *cell, walkable_override={cell}))

    def test_tl_only_large_rock_occupies_the_whole_quad(self) -> None:
        ram = make_navigation_ram(current_tile=(11, 10), blocked_tile=(63, 63))
        _set_tile(ram, 12, 10, LARGE_ROCK_TL)
        pf = Pathfinder()
        for cell in ROCK_QUAD:
            self.assertFalse(
                pf.is_walkable(ram, *cell),
                msg=f"quad sibling {cell} must be occupied",
            )
            self.assertTrue(is_travel_occupied(ram, *cell))
        path = pf.find_path(ram, (11, 10), (14, 10))
        self.assertIsNotNone(path)
        assert path is not None
        self.assertTrue(set(ROCK_QUAD).isdisjoint(path))
        self.assertEqual(path[-1], (14, 10))

    def test_find_path_goes_around_large_rock(self) -> None:
        ram = make_navigation_ram(current_tile=(11, 10), blocked_tile=(63, 63))
        _place_large_rock(ram, 12, 10)
        pf = Pathfinder()
        path = pf.find_path(ram, (11, 10), (14, 10))
        self.assertIsNotNone(path)
        assert path is not None
        occupied = set(ROCK_QUAD)
        self.assertTrue(occupied.isdisjoint(path))
        self.assertEqual(path[-1], (14, 10))

    def test_find_approach_large_rock_tl_is_adjacent_dirt(self) -> None:
        ram = make_navigation_ram(current_tile=(8, 10), blocked_tile=(63, 63))
        _place_large_rock(ram, 12, 10)
        pf = Pathfinder()
        player = Point(8 * TILE_SIZE + 8, 10 * TILE_SIZE + 8)
        approach = pf.find_approach(
            ram,
            (12, 10),
            player,
            footprint=debris_footprint((12, 10), LARGE_ROCK_TL),
        )
        self.assertIsNotNone(approach)
        assert approach is not None
        self.assertNotIn(approach, ROCK_QUAD)
        ax, ay = approach
        self.assertEqual(int(ram[ADDR_MAP + ay * MAP_WIDTH + ax]), DIRT)
        self.assertTrue(
            any(abs(ax - rx) + abs(ay - ry) == 1 for rx, ry in ROCK_QUAD)
        )

    def test_find_path_skips_weed_despite_farm_walkable(self) -> None:
        ram = make_navigation_ram(current_tile=(10, 10), blocked_tile=(63, 63))
        _set_tile(ram, 11, 10, WEED)
        pf = Pathfinder()
        path = pf.find_path(ram, (10, 10), (12, 10))
        self.assertIsNotNone(path)
        assert path is not None
        self.assertNotIn((11, 10), path)
        self.assertEqual(path[-1], (12, 10))

    def test_standing_on_weed_is_walkable_to_leave(self) -> None:
        ram = make_navigation_ram(current_tile=(10, 10), blocked_tile=(63, 63))
        _set_tile(ram, 10, 10, WEED)
        pf = Pathfinder()
        self.assertFalse(pf.is_walkable(ram, 10, 10))
        self.assertTrue(pf.is_walkable(ram, 10, 10, current_pos=(10, 10)))


if __name__ == "__main__":
    unittest.main()
