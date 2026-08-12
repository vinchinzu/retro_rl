from __future__ import annotations

import unittest

import numpy as np

from harvest.core.animal_probe import BARN_TILEMAP, animal_blocker_tiles_from_slots, cow_slot_snapshots, cow_tiles_from_slots
from harvest.core.tile_catalog import ADDR_TILEMAP
from harvest.core.ram_catalog import COW_SLOT_BASE, COW_SLOT_SIZE


def _set_cow_slot(ram: np.ndarray, slot: int, tile: tuple[int, int], *, status: int = 0x05) -> None:
    offset = COW_SLOT_BASE + slot * COW_SLOT_SIZE
    ram[offset] = status
    ram[offset + 2] = BARN_TILEMAP
    ram[offset + 3] = 0
    ram[offset + 4] = 120
    px = tile[0] * 16 + 8
    py = tile[1] * 16 + 8
    ram[offset + 8] = px & 0xFF
    ram[offset + 9] = px >> 8
    ram[offset + 10] = py & 0xFF
    ram[offset + 11] = py >> 8


class AnimalProbeTests(unittest.TestCase):
    def test_cow_slot_snapshots_decode_barn_positions(self) -> None:
        ram = np.zeros(0x20000, dtype=np.uint8)
        ram[ADDR_TILEMAP] = BARN_TILEMAP
        _set_cow_slot(ram, 0, (10, 17))

        rows = cow_slot_snapshots(ram, require_barn=True)

        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["tile"], [10, 17])
        self.assertEqual(rows[0]["home_map_raw"], BARN_TILEMAP)
        self.assertEqual(rows[0]["happiness"], 120)

    def test_cow_tiles_feed_barn_blocker_helper(self) -> None:
        ram = np.zeros(0x20000, dtype=np.uint8)
        ram[ADDR_TILEMAP] = BARN_TILEMAP
        _set_cow_slot(ram, 0, (10, 17))
        _set_cow_slot(ram, 1, (12, 17))

        self.assertEqual(cow_tiles_from_slots(ram, require_barn=True), {(10, 17), (12, 17)})
        self.assertEqual(animal_blocker_tiles_from_slots(ram), {(10, 17), (12, 17)})


if __name__ == "__main__":
    unittest.main()
