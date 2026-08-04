"""Shared animal and livestock RAM helpers."""

from __future__ import annotations

from typing import Tuple

import numpy as np

from harvest.core.ram_catalog import field_spec, live_wram_base, read_animal_slot_field, read_ram_u8, read_ram_u16

ADDR_CHICKEN_COUNT = field_spec("num_chickens").address
ADDR_COW_COUNT = field_spec("num_cows").address
ADDR_NUM_COWS = ADDR_COW_COUNT
ADDR_HAY_COUNT = field_spec("stored_grass").address
ADDR_STORED_GRASS = ADDR_HAY_COUNT
ADDR_FED_CHICKENS_N = field_spec("fed_chickens_n").address
ADDR_FED_CHICKENS_FLAGS = field_spec("fed_chickens_flags").address
ADDR_FED_COWS_N = field_spec("fed_cows_n").address
ADDR_FED_COWS_FLAGS = field_spec("fed_cows_flags").address
ADDR_ITEM_ON_HAND = field_spec("held_item").address
ADDR_HELD_ITEM = ADDR_ITEM_ON_HAND
ADDR_INCUBATOR_FLAGS = field_spec("incubator_flags").address
ADDR_EGG_AVAILABLE = field_spec("egg_available").address

ITEM_EGG = 0x14
ITEM_FODDER = 0x1A
ITEM_CHICKEN_FEED = ITEM_FODDER
CHICKEN_SLOT_BASE = 0xC286
CHICKEN_SLOT_SIZE = 8
CHICKEN_SLOT_COUNT = 12
CHICKEN_SLOT_POSITION_X_OFFSET = 0x04
CHICKEN_SLOT_POSITION_Y_OFFSET = 0x06
TILE_SIZE = 16
INCUBATOR_EGG_TILES = frozenset({(14, 11)})
COW_SLOT_COUNT = 12
INCUBATOR_BIT = 0x2000
COW_DAILY_BRUSHED_FLAG = 0x01
COW_DAILY_MILKED_FLAG = 0x02
COW_DAILY_TALKED_FLAG = 0x04
COW_DAILY_ATTENTION_FLAGS = COW_DAILY_BRUSHED_FLAG | COW_DAILY_TALKED_FLAG
COW_STATUS_BABY_FLAG = 0x02
COW_STATUS_YOUNG_FLAG = 0x04
COW_STATUS_ADULT_FLAG = 0x08
COW_STATUS_SICK_FLAG = 0x20
COW_STATUS_PREGNANT_FLAG = 0x40


def read_hay_count(ram: np.ndarray) -> int:
    """Read 16-bit hay/stored grass count."""
    return read_ram_u16(ram, ADDR_HAY_COUNT)


def read_stored_grass(ram: np.ndarray) -> int:
    return read_hay_count(ram)


def read_item_on_hand(ram: np.ndarray) -> int:
    return read_ram_u8(ram, ADDR_ITEM_ON_HAND)


def read_held_item(ram: np.ndarray) -> int:
    return read_item_on_hand(ram)


def read_num_cows(ram: np.ndarray) -> int:
    return read_ram_u8(ram, ADDR_COW_COUNT)


def read_fed_chickens_n(ram: np.ndarray) -> int:
    return read_ram_u8(ram, ADDR_FED_CHICKENS_N)


def read_fed_chickens_flags(ram: np.ndarray) -> int:
    return read_ram_u16(ram, ADDR_FED_CHICKENS_FLAGS)


def read_fed_cows_n(ram: np.ndarray) -> int:
    return read_ram_u8(ram, ADDR_FED_COWS_N)


def read_fed_cows_flags(ram: np.ndarray) -> int:
    return read_ram_u16(ram, ADDR_FED_COWS_FLAGS)


def read_cow_status(ram: np.ndarray, slot: int) -> int:
    return read_animal_slot_field(ram, "cow", slot, "status_raw")


def read_cow_daily_flags(ram: np.ndarray, slot: int) -> int:
    return read_animal_slot_field(ram, "cow", slot, "raw_1")


def read_cow_happiness(ram: np.ndarray, slot: int) -> int:
    return read_animal_slot_field(ram, "cow", slot, "happiness")


def cow_slot_exists(ram: np.ndarray, slot: int) -> bool:
    return bool(read_cow_status(ram, slot) & 0x01)


def existing_cow_slots(ram: np.ndarray) -> list[int]:
    slots: list[int] = []
    for slot in range(COW_SLOT_COUNT):
        try:
            if cow_slot_exists(ram, slot):
                slots.append(slot)
        except Exception:
            break
    return slots


def cow_needs_daily_attention(ram: np.ndarray, slot: int) -> bool:
    flags = read_cow_daily_flags(ram, slot)
    return (flags & COW_DAILY_ATTENTION_FLAGS) != COW_DAILY_ATTENTION_FLAGS


def cow_is_milk_ready(ram: np.ndarray, slot: int) -> bool:
    status = read_cow_status(ram, slot)
    if not (status & COW_STATUS_ADULT_FLAG):
        return False
    return not bool(status & (COW_STATUS_SICK_FLAG | COW_STATUS_PREGNANT_FLAG))


def cow_needs_milking(ram: np.ndarray, slot: int) -> bool:
    return cow_is_milk_ready(ram, slot) and not bool(read_cow_daily_flags(ram, slot) & COW_DAILY_MILKED_FLAG)


def is_holding_egg(ram: np.ndarray) -> bool:
    """True when the player is currently holding an egg."""
    return read_item_on_hand(ram) == ITEM_EGG


def is_incubating(ram: np.ndarray) -> bool:
    """True when an egg is currently in the incubator."""
    return bool(read_ram_u16(ram, ADDR_INCUBATOR_FLAGS) & INCUBATOR_BIT)


def egg_available_today(ram: np.ndarray) -> bool:
    """True when the chicken has laid an egg that has not been picked up yet."""
    return read_egg_available_flags(ram) > 0


def read_egg_available_flags(ram: np.ndarray) -> int:
    """Read the bitfield of floor eggs available in the coop."""
    return read_ram_u16(ram, ADDR_EGG_AVAILABLE)


def count_chicken_slots(ram: np.ndarray) -> Tuple[int, int, int]:
    """Count (adults, chicks, eggs) across the 12 chicken slots."""
    adults = chicks = eggs = 0
    base = live_wram_base(ram)
    for i in range(CHICKEN_SLOT_COUNT):
        addr = base + CHICKEN_SLOT_BASE + i * CHICKEN_SLOT_SIZE
        if addr >= len(ram):
            break
        b = int(ram[addr])
        if not (b & 0x01):
            continue
        age = (b >> 1) & 0x07
        if age >= 4:
            adults += 1
        elif age >= 2:
            chicks += 1
        else:
            eggs += 1
    return adults, chicks, eggs


def chicken_slot_eggs_available(ram: np.ndarray) -> bool:
    """True when persistent chicken slots show an uncollected egg object."""
    base = live_wram_base(ram)
    incubating = bool(read_ram_u16(ram, ADDR_INCUBATOR_FLAGS) & INCUBATOR_BIT)
    for i in range(CHICKEN_SLOT_COUNT):
        addr = base + CHICKEN_SLOT_BASE + i * CHICKEN_SLOT_SIZE
        if addr >= len(ram):
            break
        status = int(ram[addr])
        if not (status & 0x01):
            continue
        age = (status >> 1) & 0x07
        if age >= 2:
            continue
        x_addr = addr + CHICKEN_SLOT_POSITION_X_OFFSET
        y_addr = addr + CHICKEN_SLOT_POSITION_Y_OFFSET
        if y_addr + 1 >= len(ram):
            return True
        x = int(ram[x_addr]) | (int(ram[x_addr + 1]) << 8)
        y = int(ram[y_addr]) | (int(ram[y_addr + 1]) << 8)
        tile = (x // TILE_SIZE, y // TILE_SIZE)
        if incubating and tile in INCUBATOR_EGG_TILES:
            continue
        return True
    return False


def count_cow_slots(ram: np.ndarray) -> int:
    return len(existing_cow_slots(ram))


def ram_has_chickens(ram: np.ndarray) -> bool:
    """Return True when live/save RAM has at least one chicken."""
    return read_ram_u8(ram, ADDR_CHICKEN_COUNT) > 0


def ram_has_cows(ram: np.ndarray) -> bool:
    """Return True when live/save RAM has at least one cow."""
    return read_ram_u8(ram, ADDR_COW_COUNT) > 0


def ram_needs_chicken_chores(ram: np.ndarray) -> bool:
    """Return True when any chicken chore remains for the current day."""
    adults, chicks, eggs = count_chicken_slots(ram)
    if adults <= 0 and chicks == 0 and eggs == 0:
        adults = read_ram_u8(ram, ADDR_CHICKEN_COUNT)
    if adults <= 0:
        return egg_available_today(ram)
    fed = read_fed_chickens_n(ram)
    needs_feed = fed < adults and (
        read_hay_count(ram) > 0 or read_ram_u8(ram, ADDR_ITEM_ON_HAND) == ITEM_CHICKEN_FEED
    )
    return needs_feed or egg_available_today(ram) or chicken_slot_eggs_available(ram) or is_holding_egg(ram)


def ram_needs_cow_chores(ram: np.ndarray) -> bool:
    """Return True when any cow still needs feed, talk, or brushing today."""
    slots = existing_cow_slots(ram)
    cows = max(read_ram_u8(ram, ADDR_COW_COUNT), len(slots))
    if cows <= 0:
        return read_ram_u8(ram, ADDR_ITEM_ON_HAND) == ITEM_FODDER
    fed = read_ram_u8(ram, ADDR_FED_COWS_N)
    needs_feed = fed < cows and (read_hay_count(ram) > 0 or read_ram_u8(ram, ADDR_ITEM_ON_HAND) == ITEM_FODDER)
    needs_attention = any(cow_needs_daily_attention(ram, slot) for slot in slots)
    needs_milk = any(cow_needs_milking(ram, slot) for slot in slots)
    return needs_feed or needs_attention or needs_milk


__all__ = [
    "ADDR_CHICKEN_COUNT",
    "ADDR_COW_COUNT",
    "ADDR_NUM_COWS",
    "ADDR_HAY_COUNT",
    "ADDR_STORED_GRASS",
    "ADDR_FED_CHICKENS_N",
    "ADDR_FED_CHICKENS_FLAGS",
    "ADDR_FED_COWS_N",
    "ADDR_FED_COWS_FLAGS",
    "ADDR_ITEM_ON_HAND",
    "ADDR_HELD_ITEM",
    "ADDR_INCUBATOR_FLAGS",
    "ADDR_EGG_AVAILABLE",
    "ITEM_EGG",
    "ITEM_FODDER",
    "ITEM_CHICKEN_FEED",
    "CHICKEN_SLOT_BASE",
    "CHICKEN_SLOT_SIZE",
    "CHICKEN_SLOT_COUNT",
    "INCUBATOR_EGG_TILES",
    "COW_SLOT_COUNT",
    "INCUBATOR_BIT",
    "COW_DAILY_BRUSHED_FLAG",
    "COW_DAILY_MILKED_FLAG",
    "COW_DAILY_TALKED_FLAG",
    "COW_DAILY_ATTENTION_FLAGS",
    "COW_STATUS_BABY_FLAG",
    "COW_STATUS_YOUNG_FLAG",
    "COW_STATUS_ADULT_FLAG",
    "COW_STATUS_SICK_FLAG",
    "COW_STATUS_PREGNANT_FLAG",
    "read_hay_count",
    "read_stored_grass",
    "read_item_on_hand",
    "read_held_item",
    "read_num_cows",
    "read_fed_chickens_n",
    "read_fed_chickens_flags",
    "read_fed_cows_n",
    "read_fed_cows_flags",
    "read_cow_status",
    "read_cow_daily_flags",
    "read_cow_happiness",
    "cow_slot_exists",
    "existing_cow_slots",
    "cow_needs_daily_attention",
    "cow_is_milk_ready",
    "cow_needs_milking",
    "is_holding_egg",
    "is_incubating",
    "egg_available_today",
    "read_egg_available_flags",
    "chicken_slot_eggs_available",
    "count_chicken_slots",
    "count_cow_slots",
    "ram_has_chickens",
    "ram_has_cows",
    "ram_needs_chicken_chores",
    "ram_needs_cow_chores",
]
