"""Shared fixtures/helpers for CoopChoresTask unit tests (split from monofile)."""
from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from harvest.tasks.coop_task import (
    ADDR_EGG_AVAILABLE,
    ADDR_HAY_COUNT,
    ADDR_INCUBATOR_FLAGS,
    ADDR_ITEM_ON_HAND,
    ADDR_FED_CHICKENS_FLAGS,
    ADDR_FED_CHICKENS_N,
    CHICKEN_SLOT_BASE,
    CHICKEN_SLOT_SIZE,
    FEED_BIN_STAND,
    INCUBATOR_BIT,
    ITEM_CHICKEN_FEED,
    ITEM_EGG,
    VISIBLE_EGG_SPRITE,
)
from harvest.planner.day_plan import ADDR_CHICKEN_COUNT
from harvest.core.tile_catalog import (
    ADDR_INPUT_LOCK,
    ADDR_MAP,
    ADDR_TILEMAP,
    ADDR_X,
    ADDR_Y,
)
from harvest.tasks.nav import MAP_WIDTH
from harvest.tasks.harvest_task import ADDR_SHIPPING_MONEY
from harvest.core.npc_catalog import GOBJ_INITIALIZED, GOBJ_STRUCT_BASE, GOBJ_STRUCT_STRIDE


def block_tiles(ram: np.ndarray, tiles: list[tuple[int, int]]) -> None:
    """Mark tiles unwalkable in the fake coop map."""
    for tx, ty in tiles:
        ram[ADDR_MAP + ty * MAP_WIDTH + tx] = 0x00


def make_coop_ram(
    *,
    adults: int = 1,
    chicks: int = 0,
    slot_eggs: int = 0,
    hay: int = 50,
    egg_available: bool | int = True,
    incubating: bool = False,
    holding_egg: bool = False,
    holding_feed: bool = False,
    fed_chickens: int = 0,
    fed_chicken_flags: int = 0,
    shipping_money: int = 0,
    player_tile: tuple = FEED_BIN_STAND,
    live_offset: bool = False,
) -> np.ndarray:
    """Build a fake RAM snapshot inside the chicken coop."""
    ram = np.zeros(0x24000 if live_offset else 0x20000, dtype=np.uint8)
    base = 0x4000 if live_offset else 0
    ram[ADDR_TILEMAP] = 0x28
    ram[ADDR_INPUT_LOCK] = 1

    # Player position (pixel = tile * 16 + 8)
    px = player_tile[0] * 16 + 8
    py = player_tile[1] * 16 + 8
    ram[ADDR_X] = px & 0xFF
    ram[ADDR_X + 1] = (px >> 8) & 0xFF
    ram[ADDR_Y] = py & 0xFF
    ram[ADDR_Y + 1] = (py >> 8) & 0xFF

    # Fill coop tilemap with walkable tiles
    for i in range(64 * 64):
        ram[ADDR_MAP + i] = 0xA1

    # Chicken count
    ram[ADDR_CHICKEN_COUNT + base] = adults + chicks + slot_eggs

    # Chicken slots: adults first, then chicks, then eggs
    slot = 0
    for _ in range(adults):
        ram[CHICKEN_SLOT_BASE + base + slot * CHICKEN_SLOT_SIZE] = 0x09  # exists + adult age
        slot += 1
    for _ in range(chicks):
        ram[CHICKEN_SLOT_BASE + base + slot * CHICKEN_SLOT_SIZE] = 0x05  # exists + chick age
        slot += 1
    for _ in range(slot_eggs):
        ram[CHICKEN_SLOT_BASE + base + slot * CHICKEN_SLOT_SIZE] = 0x03  # exists + egg age
        slot += 1

    # Hay
    ram[ADDR_HAY_COUNT + base] = hay & 0xFF
    ram[ADDR_HAY_COUNT + base + 1] = (hay >> 8) & 0xFF
    ram[ADDR_FED_CHICKENS_N + base] = fed_chickens & 0xFF
    ram[ADDR_FED_CHICKENS_FLAGS + base] = fed_chicken_flags & 0xFF
    ram[ADDR_FED_CHICKENS_FLAGS + base + 1] = (fed_chicken_flags >> 8) & 0xFF

    # Egg available bitfield
    egg_flags = int(egg_available) if not isinstance(egg_available, bool) else (1 if egg_available else 0)
    ram[ADDR_EGG_AVAILABLE + base] = egg_flags & 0xFF
    ram[ADDR_EGG_AVAILABLE + base + 1] = (egg_flags >> 8) & 0xFF

    # Incubator
    if incubating:
        flags = INCUBATOR_BIT | 0x0400
    else:
        flags = 0x0400
    ram[ADDR_INCUBATOR_FLAGS + base] = flags & 0xFF
    ram[ADDR_INCUBATOR_FLAGS + base + 1] = (flags >> 8) & 0xFF

    # Held item
    if holding_egg:
        ram[ADDR_ITEM_ON_HAND + base] = ITEM_EGG
    elif holding_feed:
        ram[ADDR_ITEM_ON_HAND + base] = ITEM_CHICKEN_FEED

    # Shipping money
    ram[ADDR_SHIPPING_MONEY + base] = shipping_money & 0xFF
    ram[ADDR_SHIPPING_MONEY + base + 1] = (shipping_money >> 8) & 0xFF

    return ram


def make_world(ram: np.ndarray):
    return SimpleNamespace(ram=ram, info={}, obs=None)


def add_chicken_object(ram: np.ndarray, tile: tuple[int, int], *, slot: int = 1) -> None:
    offset = GOBJ_STRUCT_BASE + slot * GOBJ_STRUCT_STRIDE
    ram[offset] = GOBJ_INITIALIZED & 0xFF
    ram[offset + 1] = (GOBJ_INITIALIZED >> 8) & 0xFF
    ram[offset + 2] = 0xE1
    ram[offset + 3] = 0x01
    px = tile[0] * 16 + 8
    py = tile[1] * 16 + 8
    ram[offset + 0x08] = px & 0xFF
    ram[offset + 0x09] = (px >> 8) & 0xFF
    ram[offset + 0x0A] = py & 0xFF
    ram[offset + 0x0B] = (py >> 8) & 0xFF


def add_egg_object(ram: np.ndarray, tile: tuple[int, int], *, slot: int = 2) -> None:
    offset = GOBJ_STRUCT_BASE + slot * GOBJ_STRUCT_STRIDE
    ram[offset] = GOBJ_INITIALIZED & 0xFF
    ram[offset + 1] = (GOBJ_INITIALIZED >> 8) & 0xFF
    ram[offset + 2] = VISIBLE_EGG_SPRITE & 0xFF
    ram[offset + 3] = (VISIBLE_EGG_SPRITE >> 8) & 0xFF
    px = tile[0] * 16 + 8
    py = tile[1] * 16 + 8
    ram[offset + 0x08] = px & 0xFF
    ram[offset + 0x09] = (px >> 8) & 0xFF
    ram[offset + 0x0A] = py & 0xFF
    ram[offset + 0x0B] = (py >> 8) & 0xFF


def set_chicken_slot_position(
    ram: np.ndarray, slot: int, tile: tuple[int, int], *, live_offset: bool = False
) -> None:
    base = 0x4000 if live_offset else 0
    offset = CHICKEN_SLOT_BASE + base + slot * CHICKEN_SLOT_SIZE
    ram[offset + 1] = 0x28
    px = tile[0] * 16 + 8
    py = tile[1] * 16 + 8
    ram[offset + 4] = px & 0xFF
    ram[offset + 5] = (px >> 8) & 0xFF
    ram[offset + 6] = py & 0xFF
    ram[offset + 7] = (py >> 8) & 0xFF
