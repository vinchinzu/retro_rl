"""Unit tests for the two-slot carry pair helpers."""

from __future__ import annotations

import unittest

import numpy as np

from harvest.core.carry import (
    ADDR_TOOL_BACKPACK,
    ADDR_TOOL_SELECTED,
    SEED_ITEM,
    backpack_tool,
    carry_pair_items,
    carry_pair_nonempty,
    seed_in_carry_pair,
    seed_item_id,
    selected_tool,
    tool_in_carry_pair,
    watering_can_in_carry_pair,
)
from harvest.core.tile_catalog import Tool


def _ram(selected: int = 0, backpack: int = 0) -> np.ndarray:
    ram = np.zeros(0x1000, dtype=np.uint8)
    ram[ADDR_TOOL_SELECTED] = selected
    ram[ADDR_TOOL_BACKPACK] = backpack
    return ram


class CarryPairTests(unittest.TestCase):
    def test_pair_reads_both_slots(self) -> None:
        ram = _ram(int(Tool.HOE), SEED_ITEM["potato"])
        self.assertEqual(carry_pair_items(ram), {int(Tool.HOE), SEED_ITEM["potato"]})
        self.assertTrue(tool_in_carry_pair(ram, int(Tool.HOE)))
        self.assertTrue(seed_in_carry_pair(ram, "potato"))
        self.assertFalse(watering_can_in_carry_pair(ram))
        self.assertEqual(selected_tool(ram), int(Tool.HOE))
        self.assertEqual(backpack_tool(ram), SEED_ITEM["potato"])
        self.assertTrue(carry_pair_nonempty(ram))

    def test_empty_pair(self) -> None:
        ram = _ram(0, 0)
        self.assertEqual(carry_pair_items(ram), {0})
        self.assertFalse(carry_pair_nonempty(ram))
        self.assertFalse(seed_in_carry_pair(ram, "potato"))

    def test_seed_item_ids(self) -> None:
        self.assertEqual(seed_item_id("potato"), 0x07)
        self.assertEqual(seed_item_id("unknown"), 0x07)


if __name__ == "__main__":
    unittest.main()
