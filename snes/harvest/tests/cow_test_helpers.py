"""Fake barn RAM for CowChoresTask unit tests."""
from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from harvest.core.animal_status import ADDR_FED_COWS_FLAGS
from harvest.core.ram_catalog import COW_SLOT_BASE, COW_SLOT_SIZE
from harvest.core.tile_catalog import ADDR_INPUT_LOCK, ADDR_STAMINA, ADDR_TILEMAP, ADDR_X, ADDR_Y
from harvest.tasks.cow_task import (
    ADDR_FED_COWS_N,
    ADDR_HELD_ITEM,
    ADDR_NUM_COWS,
    ADDR_STORED_GRASS,
    ADDR_TOOL_BACKPACK,
    ADDR_TOOL_SELECTED,
)

ADULT, CHILD, BABY, SICK, PREGNANT = 0x09, 0x05, 0x03, 0x29, 0x49
# Twelve stalls, ROM ``Table_CowsSpawnPos``: right column top to bottom is read here bottom up by slot.
STALL_PX = (
    (168, 278), (168, 246), (168, 214), (168, 150), (168, 118), (168, 86),
    (56, 278), (56, 246), (56, 214), (56, 150), (56, 118), (56, 86),
)


def write_u16(ram: np.ndarray, addr: int, value: int) -> None:
    ram[addr] = value & 0xFF
    ram[addr + 1] = (value >> 8) & 0xFF


def write_u24(ram: np.ndarray, addr: int, value: int) -> None:
    write_u16(ram, addr, value)
    ram[addr + 2] = (value >> 16) & 0xFF


def set_player_px(ram: np.ndarray, pos: tuple[int, int]) -> None:
    write_u16(ram, ADDR_X, pos[0])
    write_u16(ram, ADDR_Y, pos[1])


def set_cow(ram: np.ndarray, slot: int, pixel: tuple[int, int] | None = None, *, status: int = ADULT, flags: int = 0, happiness: int = 0) -> None:
    offset = COW_SLOT_BASE + slot * COW_SLOT_SIZE
    ram[offset] = status
    ram[offset + 1] = flags
    ram[offset + 2] = 0x27
    ram[offset + 4] = happiness
    x, y = pixel or STALL_PX[slot]
    write_u16(ram, offset + 8, x)
    write_u16(ram, offset + 10, y)


def set_cow_slot(ram: np.ndarray, slot: int, tile: tuple[int, int], *, status: int = CHILD) -> None:
    """A cow on a tile centre."""
    set_cow(ram, slot, (tile[0] * 16 + 8, tile[1] * 16 + 8), status=status)


def set_cow_daily(ram: np.ndarray, slot: int, *, flags: int, happiness: int) -> None:
    offset = COW_SLOT_BASE + slot * COW_SLOT_SIZE
    ram[offset + 1] = flags
    ram[offset + 4] = happiness


def make_barn_ram(
    *,
    cows: int = 1,
    status: int = ADULT,
    fed_flags: int = 0,
    hay: int = 10,
    held_item: int = 0,
    tool_selected: int = 0,
    tool_backpack: int = 0,
    player_px: tuple[int, int] = (128, 356),
) -> np.ndarray:
    ram = np.zeros(0x20000, dtype=np.uint8)
    ram[ADDR_TILEMAP] = 0x27
    ram[ADDR_INPUT_LOCK] = 1
    ram[0x0025] = 15  # faded in
    ram[ADDR_STAMINA] = 100
    set_player_px(ram, player_px)
    ram[ADDR_NUM_COWS] = cows
    ram[ADDR_FED_COWS_N] = bin(fed_flags).count("1")
    write_u16(ram, ADDR_FED_COWS_FLAGS, fed_flags)
    write_u16(ram, ADDR_STORED_GRASS, hay)
    ram[ADDR_HELD_ITEM] = held_item
    ram[ADDR_TOOL_SELECTED] = tool_selected
    ram[ADDR_TOOL_BACKPACK] = tool_backpack
    for slot in range(cows):
        set_cow(ram, slot, status=status)
    return ram


def make_world(ram: np.ndarray):
    return SimpleNamespace(ram=ram, info={}, obs=None, frame=0)


_make_barn_ram = make_barn_ram
_make_world = make_world
