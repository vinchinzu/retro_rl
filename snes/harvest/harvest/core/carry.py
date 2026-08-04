"""Two-slot carry pair (selected + backpack) helpers.

Harvest Moon SNES only holds two tools/items at a time. Shop seed bags and
shed tools live on shelves until taken with A; X only swaps the carry pair.
"""

from __future__ import annotations

from typing import Dict, Set

import numpy as np

from harvest.core.ram_catalog import field_spec, read_ram_u8
from harvest.core.tile_catalog import Tool

ADDR_TOOL_SELECTED = field_spec("tool_selected").address
ADDR_TOOL_BACKPACK = field_spec("tool_backpack").address

# Item IDs that occupy the tool/item slots (0x0921 / 0x0923).
SEED_ITEM: Dict[str, int] = {
    "corn": 0x05,
    "tomato": 0x06,
    "potato": 0x07,
    "turnip": 0x08,
    "grass": 0x0C,
}


def carry_pair_items(ram: np.ndarray) -> Set[int]:
    """Return the set of item ids currently in the two carry slots."""
    items: Set[int] = set()
    if ADDR_TOOL_SELECTED < len(ram):
        items.add(int(ram[ADDR_TOOL_SELECTED]))
    if ADDR_TOOL_BACKPACK < len(ram):
        items.add(int(ram[ADDR_TOOL_BACKPACK]))
    return items


def tool_in_carry_pair(ram: np.ndarray, tool_id: int) -> bool:
    return int(tool_id) in carry_pair_items(ram)


def seed_item_id(seed_type: str = "potato") -> int:
    return SEED_ITEM.get(seed_type, SEED_ITEM["potato"])


def seed_in_carry_pair(ram: np.ndarray, seed_type: str = "potato") -> bool:
    return tool_in_carry_pair(ram, seed_item_id(seed_type))


def watering_can_in_carry_pair(ram: np.ndarray) -> bool:
    return tool_in_carry_pair(ram, int(Tool.WATERING_CAN))


def selected_tool(ram: np.ndarray) -> int:
    return read_ram_u8(ram, ADDR_TOOL_SELECTED)


def backpack_tool(ram: np.ndarray) -> int:
    return read_ram_u8(ram, ADDR_TOOL_BACKPACK)


def carry_pair_nonempty(ram: np.ndarray) -> bool:
    return selected_tool(ram) != 0 or backpack_tool(ram) != 0


__all__ = [
    "ADDR_TOOL_SELECTED",
    "ADDR_TOOL_BACKPACK",
    "SEED_ITEM",
    "carry_pair_items",
    "tool_in_carry_pair",
    "seed_item_id",
    "seed_in_carry_pair",
    "watering_can_in_carry_pair",
    "selected_tool",
    "backpack_tool",
    "carry_pair_nonempty",
]
