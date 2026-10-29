"""N-ring capacity: stamina gate, harvest/water per-tile failure memory."""

from __future__ import annotations

import unittest
from unittest import mock

import numpy as np

from harvest.core.tile_catalog import ADDR_MAP
from harvest.maps.farm_pond import (
    POCKET_PLANT_CENTERS,
    pocket_water_reserve,
    ring_establish_affordable,
)
from harvest.tasks import crop_planter, harvest_task
from harvest.tasks.crop_geometry import plot_tiles
from tests.crop_planter_test_helpers import set_stamina, set_tile

PLANTED_DRY = 0x54


def _field() -> np.ndarray:
    ram = np.zeros(0x24000, dtype=np.uint8)
    ram[ADDR_MAP : ADDR_MAP + 64 * 64] = 0x01  # open soil everywhere
    return ram


def _plant(ram, center, tid=PLANTED_DRY) -> None:
    for tx, ty in plot_tiles(center, include_center=False):
        set_tile(ram, tx, ty, tid)


class StaminaGateTests(unittest.TestCase):
    def test_full_stamina_opens_a_bare_ring(self) -> None:
        ram = _field()
        set_stamina(ram, 100)
        self.assertTrue(ring_establish_affordable(ram, POCKET_PLANT_CENTERS[0]))

    def test_exhausted_farmer_opens_nothing(self) -> None:
        ram = _field()
        set_stamina(ram, 10)
        self.assertFalse(ring_establish_affordable(ram, POCKET_PLANT_CENTERS[0]))

    def test_standing_crops_are_watered_before_a_new_ring_is_hoed(self) -> None:
        ram = _field()
        for center in POCKET_PLANT_CENTERS[:3]:
            _plant(ram, center)
        self.assertEqual(pocket_water_reserve(ram), 3 * 8 * 2)
        set_stamina(ram, 60)  # hoe 16 + owed watering 48 = 64 > 60
        self.assertFalse(ring_establish_affordable(ram, POCKET_PLANT_CENTERS[3]))
        set_stamina(ram, 64)
        self.assertTrue(ring_establish_affordable(ram, POCKET_PLANT_CENTERS[3]))

    def test_replanting_hoed_soil_costs_no_hoe_stamina(self) -> None:
        ram = _field()
        for tx, ty in plot_tiles(POCKET_PLANT_CENTERS[0], include_center=False):
            set_tile(ram, tx, ty, 0x08)  # watered tilled soil left by a harvest
        set_stamina(ram, 0)
        self.assertTrue(ring_establish_affordable(ram, POCKET_PLANT_CENTERS[0]))


class HarvestFailureMemoryTests(unittest.TestCase):
    def setUp(self) -> None:
        harvest_task.reset_harvest_failure_memory()

    def tearDown(self) -> None:
        harvest_task.reset_harvest_failure_memory()

    def test_tile_is_parked_after_the_limit_and_forgotten_when_no_longer_ripe(self) -> None:
        tile = (30, 40)
        live = [tile, (5, 5)]
        kept, parked = harvest_task.drop_failed_harvest_tiles(live)
        self.assertEqual((kept, parked), (live, []))
        for _ in range(harvest_task.HARVEST_TILE_FAIL_LIMIT):
            harvest_task.note_harvest_tile_failure(tile)
        kept, parked = harvest_task.drop_failed_harvest_tiles(live)
        self.assertEqual((kept, parked), ([(5, 5)], [tile]))
        # The tile is picked / replaced: it stops being a target, memory clears.
        harvest_task.drop_failed_harvest_tiles([(5, 5)])
        kept, parked = harvest_task.drop_failed_harvest_tiles(live)
        self.assertEqual(parked, [])


class WaterFailureMemoryTests(unittest.TestCase):
    def setUp(self) -> None:
        crop_planter.reset_water_failure_memory()

    def tearDown(self) -> None:
        crop_planter.reset_water_failure_memory()

    def test_unwaterable_tile_is_parked_then_forgotten_once_wet(self) -> None:
        ram = _field()
        tile = (13, 29)
        set_tile(ram, *tile, PLANTED_DRY)
        bounds = (0, 0, 63, 63)
        for _ in range(crop_planter.WATER_TILE_FAIL_LIMIT):
            crop_planter.note_water_tile_failure(tile)
        self.assertEqual(crop_planter.parked_water_tiles(ram, bounds), {tile})
        set_tile(ram, *tile, PLANTED_DRY + 1)  # watered by other means
        self.assertEqual(crop_planter.parked_water_tiles(ram, bounds), set())
        self.assertNotIn(tile, crop_planter._WATER_TILE_FAILURES)

    def test_parked_tile_is_retried_after_the_park_days(self) -> None:
        # A parked tile is never watered, so it never stops being dry: without a
        # retry, two bad days parked a whole field for the rest of the season.
        ram = _field()
        tile = (13, 29)
        set_tile(ram, *tile, PLANTED_DRY)
        bounds = (0, 0, 63, 63)
        crop_planter._WATER_CLOCK["day"] = 40
        for _ in range(crop_planter.WATER_TILE_FAIL_LIMIT):
            crop_planter.note_water_tile_failure(tile)
        with mock.patch.object(crop_planter, "_water_day", return_value=40 + crop_planter.WATER_PARK_DAYS - 1):
            self.assertEqual(crop_planter.parked_water_tiles(ram, bounds), {tile})
        with mock.patch.object(crop_planter, "_water_day", return_value=40 + crop_planter.WATER_PARK_DAYS):
            self.assertEqual(crop_planter.parked_water_tiles(ram, bounds), set())
        self.assertNotIn(tile, crop_planter._WATER_TILE_FAILURES)
        # Forgotten: one new failure does not park it again, two do.
        crop_planter.note_water_tile_failure(tile)
        with mock.patch.object(crop_planter, "_water_day", return_value=43):
            self.assertEqual(crop_planter.parked_water_tiles(ram, bounds), set())
        crop_planter.note_water_tile_failure(tile)
        with mock.patch.object(crop_planter, "_water_day", return_value=43):
            self.assertEqual(crop_planter.parked_water_tiles(ram, bounds), {tile})

    def test_failures_far_apart_do_not_add_up(self) -> None:
        ram = _field()
        tile = (13, 29)
        set_tile(ram, *tile, PLANTED_DRY)
        crop_planter._WATER_CLOCK["day"] = 5
        crop_planter.note_water_tile_failure(tile)
        with mock.patch.object(crop_planter, "_water_day", return_value=5 + crop_planter.WATER_PARK_DAYS):
            self.assertEqual(crop_planter.parked_water_tiles(ram, (0, 0, 63, 63)), set())
        crop_planter.note_water_tile_failure(tile)
        with mock.patch.object(crop_planter, "_water_day", return_value=5 + crop_planter.WATER_PARK_DAYS):
            self.assertEqual(crop_planter.parked_water_tiles(ram, (0, 0, 63, 63)), set())

    def test_one_failure_is_not_enough(self) -> None:
        ram = _field()
        set_tile(ram, 13, 29, PLANTED_DRY)
        crop_planter.note_water_tile_failure((13, 29))
        self.assertEqual(crop_planter.parked_water_tiles(ram, (0, 0, 63, 63)), set())


if __name__ == "__main__":
    unittest.main()
