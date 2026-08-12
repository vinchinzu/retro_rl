"""Shared fixtures/helpers for day plan sequence tests (split from monofile)."""
from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import unittest

from harvest.core.animal_status import CHICKEN_SLOT_BASE, CHICKEN_SLOT_SIZE
from harvest.planner.day_plan import (
    ADDR_DAY,
    ADDR_HOUR,
    ADDR_MINUTE,
    ADDR_MONEY,
    ADDR_SEASON,
    ADDR_TILEMAP,
    is_house_tilemap,
)
from harvest.core.tile_catalog import (
    ADDR_INPUT_LOCK,
    ADDR_MAP,
    ADDR_X,
    ADDR_Y,
)
from harvest.core.ram_catalog import COW_SLOT_BASE, COW_SLOT_SIZE, field_spec


class DayPlanPhaseHelpers(unittest.TestCase):
    """Mixin-style base with shared phase-name helper used by build-phase tests."""

    def _phase_names(self, phases):
        return [p.phase for p in phases]


def make_world(tilemap: int):
    ram = np.zeros(0x20000, dtype=np.uint8)
    ram[ADDR_TILEMAP] = tilemap
    ram[ADDR_INPUT_LOCK] = 1
    return SimpleNamespace(ram=ram, info={}, obs=None)


def make_time_world(tilemap: int, *, day: int, hour: int, minute: int, live_offset: bool = False):
    ram = np.zeros(0x24000, dtype=np.uint8) if live_offset else np.zeros(0x20000, dtype=np.uint8)
    ram[ADDR_TILEMAP] = tilemap
    ram[ADDR_INPUT_LOCK] = 1
    base = 0x4000 if live_offset else 0
    ram[ADDR_DAY + base] = day
    ram[ADDR_HOUR + base] = hour
    ram[ADDR_MINUTE + base] = minute
    return SimpleNamespace(ram=ram, info={}, obs=None)


def make_date_world(tilemap: int, *, season: int, day: int, hour: int = 6, minute: int = 0):
    ram = np.zeros(0x24000, dtype=np.uint8)
    ram[ADDR_TILEMAP] = tilemap
    ram[ADDR_INPUT_LOCK] = 1
    base = 0x4000
    ram[ADDR_SEASON + base] = season
    ram[ADDR_DAY + base] = day
    ram[ADDR_HOUR + base] = hour
    ram[ADDR_MINUTE + base] = minute
    # Settled stand points so scene classification is not wake/invalid.
    if is_house_tilemap(tilemap):
        set_player_pos(ram, 136, 120)
    else:
        set_player_pos(ram, 136, 424)
    return SimpleNamespace(ram=ram, info={}, obs=None)


def set_player_pos(ram: np.ndarray, x: int, y: int) -> None:
    ram[ADDR_X] = x & 0xFF
    ram[ADDR_X + 1] = (x >> 8) & 0xFF
    ram[ADDR_Y] = y & 0xFF
    ram[ADDR_Y + 1] = (y >> 8) & 0xFF


def set_money(ram: np.ndarray, amount: int, *, live_offset: bool = True) -> None:
    base = 0x4000 if live_offset else 0
    raw = amount // 10
    ram[ADDR_MONEY + base] = raw & 0xFF
    ram[ADDR_MONEY + base + 1] = (raw >> 8) & 0xFF
    ram[ADDR_MONEY + base + 2] = (raw >> 16) & 0xFF


def set_live_u16(ram: np.ndarray, field: str, value: int) -> None:
    addr = field_spec(field).address
    base = 0x4000 if len(ram) > 0x20000 and addr + 0x4001 < len(ram) else 0
    ram[addr + base] = value & 0xFF
    ram[addr + base + 1] = (value >> 8) & 0xFF


def set_live_cow_slot(
    ram: np.ndarray,
    slot: int,
    *,
    flags: int = 0,
    happiness: int = 0,
    tile: tuple[int, int] = (9, 17),
) -> None:
    offset = 0x4000 + COW_SLOT_BASE + slot * COW_SLOT_SIZE
    ram[offset] = 0x05
    ram[offset + 1] = flags
    ram[offset + 2] = 0x27
    ram[offset + 4] = happiness
    px = tile[0] * 16 + 8
    py = tile[1] * 16 + 8
    ram[offset + 8] = px & 0xFF
    ram[offset + 9] = px >> 8
    ram[offset + 10] = py & 0xFF
    ram[offset + 11] = py >> 8


def set_live_chicken_slots(ram: np.ndarray, *, adults: int = 0, chicks: int = 0, eggs: int = 0) -> None:
    base = 0x4000 if len(ram) > 0x20000 else 0
    slot = 0
    for count, status in ((adults, 0x09), (chicks, 0x05), (eggs, 0x01)):
        for _ in range(count):
            offset = base + CHICKEN_SLOT_BASE + slot * CHICKEN_SLOT_SIZE
            ram[offset] = status
            slot += 1


def make_navigation_ram(*, current_tile=(13, 8), blocked_tile=(14, 8), blocked_id=0x76):
    ram = np.zeros(0x20000, dtype=np.uint8)
    ram[ADDR_TILEMAP] = 0x00
    set_player_pos(ram, current_tile[0] * 16 + 8, current_tile[1] * 16 + 8)
    for ty in range(64):
        for tx in range(64):
            ram[ADDR_MAP + ty * 64 + tx] = 0xA1
    ram[ADDR_MAP + blocked_tile[1] * 64 + blocked_tile[0]] = blocked_id
    return ram


def make_transition_world(tilemap: int, *, current_tile=(13, 8)):
    ram = make_navigation_ram(current_tile=current_tile, blocked_tile=(63, 63), blocked_id=0xA1)
    ram[ADDR_TILEMAP] = tilemap
    ram[ADDR_INPUT_LOCK] = 1
    return SimpleNamespace(ram=ram, info={}, obs=None)


