"""One shop trip per harvest day, not one per ring.

The shop round trip is ~2480 f whether it carries one bag or two, and a
harvest day empties both pocket rings at once. Before this, BUY_SEEDS always
read "bought potato_seeds 0->1" and CROP_ESTABLISH resolved exactly one ring,
so the second ring idled until the next day's trip.
"""
from __future__ import annotations

import unittest

import numpy as np

from harvest.core.ram_catalog import field_spec, ram_index
from harvest.maps.farm_pond import (
    POCKET_PLANT_CENTERS,
    pocket_plant_target,
    pocket_plant_targets,
)
from harvest.planner.day_phase_catalog import BUY_SEEDS_PHASE
from harvest.planner.day_phase_registry import _shop_bag_count
from harvest.tasks.buy_seeds import (
    BUY_ATTEMPT_LIMIT,
    POTATO_BAG_PRICE,
    BuySeedsTask,
)
from harvest.tasks.crop_geometry import plot_tiles

from tests.crop_planter_test_helpers import set_tile as _set_tile
from tests.day_plan_test_helpers import make_world

_MONEY = field_spec("money")
_STOCK = field_spec("potato_seeds")
PLANTED_DRY = 0x54


def _write_money(ram, value: int) -> None:
    """money is u24, stored as gold/10 at a live-WRAM offset."""
    raw = int(value) // int(_MONEY.display_multiplier)
    idx = ram_index(ram, _MONEY.address, live_offset=_MONEY.live_offset)
    for i in range(3):
        ram[idx + i] = (raw >> (8 * i)) & 0xFF


def _write_stock(ram, value: int) -> None:
    idx = ram_index(ram, _STOCK.address, live_offset=_STOCK.live_offset)
    ram[idx] = int(value)


def _plant_ring(ram, center):
    for tx, ty in plot_tiles(center, include_center=False):
        _set_tile(ram, tx, ty, PLANTED_DRY)


class PocketPlantTargetsTests(unittest.TestCase):
    def setUp(self) -> None:
        self.ram = np.zeros(0x20000, dtype=np.uint8)

    def test_both_rings_bare_want_two_bags(self) -> None:
        self.assertEqual(len(pocket_plant_targets(self.ram)), len(POCKET_PLANT_CENTERS))

    def test_one_ring_planted_wants_one_bag(self) -> None:
        _plant_ring(self.ram, POCKET_PLANT_CENTERS[0])
        self.assertEqual(pocket_plant_targets(self.ram), (POCKET_PLANT_CENTERS[1],))

    def test_singular_target_still_agrees_with_the_plural_form(self) -> None:
        _plant_ring(self.ram, POCKET_PLANT_CENTERS[0])
        self.assertEqual(pocket_plant_target(self.ram), pocket_plant_targets(self.ram)[0])

    def test_all_planted_wants_nothing(self) -> None:
        for center in POCKET_PLANT_CENTERS:
            _plant_ring(self.ram, center)
        self.assertEqual(pocket_plant_targets(self.ram), ())


class ShopBagCountTests(unittest.TestCase):
    def _world(self, stock: int = 0):
        world = make_world(0x00)
        _write_stock(world.ram, stock)
        return world

    def test_two_empty_rings_buy_two_bags(self) -> None:
        world = self._world()
        self.assertEqual(_shop_bag_count(BUY_SEEDS_PHASE, world, "potato_seeds"), 2)

    def test_a_bag_already_in_the_pocket_only_needs_one_more(self) -> None:
        world = self._world(stock=1)
        self.assertEqual(_shop_bag_count(BUY_SEEDS_PHASE, world, "potato_seeds"), 1)

    def test_one_waiting_ring_buys_one_bag(self) -> None:
        world = self._world()
        _plant_ring(world.ram, POCKET_PLANT_CENTERS[0])
        self.assertEqual(_shop_bag_count(BUY_SEEDS_PHASE, world, "potato_seeds"), 1)

    def test_never_returns_zero(self) -> None:
        world = self._world(stock=5)
        self.assertEqual(_shop_bag_count(BUY_SEEDS_PHASE, world, "potato_seeds"), 1)


class BuySeedsBagClampTests(unittest.TestCase):
    def _task(self, *, bags: int, money: int) -> BuySeedsTask:
        task = BuySeedsTask(bags=bags)
        task._money_before = money
        task._stock_before = 0
        return task

    def test_wallet_clamps_the_bag_count(self) -> None:
        self.assertEqual(self._task(bags=2, money=250)._bags_target(), 1)

    def test_a_full_wallet_buys_what_was_asked(self) -> None:
        self.assertEqual(self._task(bags=2, money=2 * POTATO_BAG_PRICE)._bags_target(), 2)

    def test_target_is_never_below_one(self) -> None:
        self.assertEqual(self._task(bags=2, money=0)._bags_target(), 1)

    def test_bags_done_reads_the_wallet_when_stock_lags(self) -> None:
        task = self._task(bags=2, money=400)
        ram = np.zeros(0x20000, dtype=np.uint8)
        _write_money(ram, 400 - POTATO_BAG_PRICE)
        self.assertEqual(task._bags_done(ram), 1)


class SecondBagFallbackTests(unittest.TestCase):
    """A second bag that will not close must not cost us the first one."""

    def _world_after_one_bag(self):
        world = make_world(0x1C)  # still in the shop
        _write_money(world.ram, 400 - POTATO_BAG_PRICE)
        _write_stock(world.ram, 1)
        return world

    def _task(self) -> BuySeedsTask:
        task = BuySeedsTask(bags=2)
        task._money_before = 400
        task._stock_before = 0
        task._phase = "buy"
        return task

    def test_one_bag_of_two_is_not_yet_done(self) -> None:
        task = self._task()
        world = self._world_after_one_bag()
        task.step(world)
        self.assertFalse(task._bought)

    def test_stalled_second_bag_banks_the_first_and_leaves(self) -> None:
        task = self._task()
        world = self._world_after_one_bag()
        task.step(world)  # records bag 1 at the current attempt count
        task._buy_attempts = task._attempts_at_last_bag + BUY_ATTEMPT_LIMIT
        task.step(world)
        self.assertTrue(task._bought)

    def test_a_stall_with_no_bag_at_all_never_banks(self) -> None:
        task = self._task()
        world = make_world(0x1C)
        _write_money(world.ram, 400)
        task._buy_attempts = BUY_ATTEMPT_LIMIT * 2
        task.step(world)
        self.assertFalse(task._bought)


if __name__ == "__main__":
    unittest.main()
