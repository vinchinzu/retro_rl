"""Town shop doorstep tiles read 0xFF but are the real walkway (Y1 D13)."""

import unittest

import numpy as np

from harvest.core.tile_catalog import ADDR_TILEMAP, set_tile_at
from harvest.tasks.nav import Pathfinder

TOWN = 0x04


def _town_ram() -> np.ndarray:
    ram = np.zeros(0x20000, dtype=np.uint8)
    ram[ADDR_TILEMAP] = TOWN
    for ty in range(10, 24):
        for tx in range(33, 42):
            set_tile_at(ram, tx, ty, 0xA0)
    for ty in (14, 15, 16):
        set_tile_at(ram, 37, ty, 0xFF)
    return ram


class TownDoorstepWalkwayTests(unittest.TestCase):
    def test_doorstep_column_is_walkable_and_pathable(self) -> None:
        ram = _town_ram()
        pf = Pathfinder()
        for ty in (14, 15, 16):
            self.assertTrue(pf.is_walkable(ram, 37, ty), ty)
        path = pf.find_path(ram, (37, 14), (37, 20))
        self.assertTrue(path)
        self.assertEqual(path[-1], (37, 20))

    def test_stale_ff_elsewhere_stays_blocked(self) -> None:
        ram = _town_ram()
        set_tile_at(ram, 35, 18, 0xFF)
        self.assertFalse(Pathfinder().is_walkable(ram, 35, 18))


if __name__ == "__main__":
    unittest.main()
