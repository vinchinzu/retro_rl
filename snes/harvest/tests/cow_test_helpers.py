"""Shared fixtures/helpers for CowChoresTask unit tests (split from monofile)."""
from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from harvest.tasks.cow_task import (
    ADDR_TOOL_BACKPACK,
    ADDR_TOOL_SELECTED,
    ADDR_FED_COWS_N,
    ADDR_HELD_ITEM,
    ADDR_NUM_COWS,
    ADDR_STORED_GRASS,
    COW_TALK_STAND,
)
from harvest.core.tile_catalog import (
    ADDR_INPUT_LOCK,
    ADDR_MAP,
    ADDR_TILEMAP,
    ADDR_X,
    ADDR_Y,
)
from harvest.core.ram_catalog import COW_SLOT_BASE, COW_SLOT_SIZE


def write_u16(ram: np.ndarray, addr: int, value: int) -> None:
    ram[addr] = value & 0xFF
    ram[addr + 1] = (value >> 8) & 0xFF


def write_u24(ram: np.ndarray, addr: int, value: int) -> None:
    ram[addr] = value & 0xFF
    ram[addr + 1] = (value >> 8) & 0xFF
    ram[addr + 2] = (value >> 16) & 0xFF


def set_player_tile(ram: np.ndarray, tile: tuple[int, int]) -> None:
    x = tile[0] * 16 + 8
    y = tile[1] * 16 + 8
    write_u16(ram, ADDR_X, x)
    write_u16(ram, ADDR_Y, y)


def set_player_px(ram: np.ndarray, pos: tuple[int, int]) -> None:
    write_u16(ram, ADDR_X, pos[0])
    write_u16(ram, ADDR_Y, pos[1])


def set_cow_slot(ram: np.ndarray, slot: int, tile: tuple[int, int], *, status: int = 0x05) -> None:
    offset = COW_SLOT_BASE + slot * COW_SLOT_SIZE
    ram[offset] = status
    ram[offset + 2] = 0x27
    x = tile[0] * 16 + 8
    y = tile[1] * 16 + 8
    write_u16(ram, offset + 8, x)
    write_u16(ram, offset + 10, y)


def set_cow_daily(ram: np.ndarray, slot: int, *, flags: int, happiness: int) -> None:
    offset = COW_SLOT_BASE + slot * COW_SLOT_SIZE
    ram[offset + 1] = flags
    ram[offset + 4] = happiness


def make_barn_ram(
    *,
    cows: int = 1,
    fed: int = 0,
    hay: int = 10,
    held_item: int = 0,
    tool_selected: int = 0,
    tool_backpack: int = 0,
    player_tile: tuple[int, int] = COW_TALK_STAND,
) -> np.ndarray:
    ram = np.zeros(0x20000, dtype=np.uint8)
    ram[ADDR_TILEMAP] = 0x27
    ram[ADDR_INPUT_LOCK] = 1
    set_player_tile(ram, player_tile)
    for i in range(64 * 64):
        ram[ADDR_MAP + i] = 0xA1
    ram[ADDR_NUM_COWS] = cows
    ram[ADDR_FED_COWS_N] = fed
    write_u16(ram, ADDR_STORED_GRASS, hay)
    ram[ADDR_HELD_ITEM] = held_item
    ram[ADDR_TOOL_SELECTED] = tool_selected
    ram[ADDR_TOOL_BACKPACK] = tool_backpack
    for slot in range(cows):
        set_cow_slot(ram, slot, (9 + slot, 17))
    return ram


def make_world(ram: np.ndarray):
    return SimpleNamespace(ram=ram, info={}, obs=None)


# Underscore aliases matching original test monofile names.
_write_u16 = write_u16
_write_u24 = write_u24
_set_player_tile = set_player_tile
_set_player_px = set_player_px
_set_cow_slot = set_cow_slot
_set_cow_daily = set_cow_daily
_make_barn_ram = make_barn_ram
_make_world = make_world
