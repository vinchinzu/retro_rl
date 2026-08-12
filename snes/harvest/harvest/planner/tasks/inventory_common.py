"""Shared outdoor free-move / morning-intro helpers for inventory tasks."""

from __future__ import annotations

import numpy as np

from harvest.tasks.nav import get_pos_from_ram
from harvest.core.tile_catalog import ADDR_TILEMAP
from harvest.planner.day_plan_status import is_farm_tilemap
from harvest.core.ram_catalog import read_ram_value

# High byte of game_state (u16 @ 0x00D2): free on-foot outdoor control.
# After pure D1 truck→sleep, ExitToFarm often clears this bit and the player
# is auto-walked south into house-enter stand ~(133,425) then tilemap 0x5F.
GAME_STATE_FREE_MOVE_BIT = 0x4000
# Outdoor house door / enter stand soft-lock pocket after control loss.
HOUSE_FRONT_SOFTLOCK_Y_MIN = 400
HOUSE_FRONT_SOFTLOCK_X_MAX = 160
# event_flags_1f68 bits that must be set BEFORE first house→farm exit so the
# ROM does not fire CODE_83CEAE morning intro (which softlocks free-move).
# Causal A/B on town_day1_rest_end (2026-08-09): 0x00A1/0x00B1 keep free-move;
# 0x0011/0x0031/0x0091 lose it. 0x0080 = dog owned (HM-Decomp bank_84).
EVENT_1F68_TRUCK_BIT = 0x0001  # set by truck/day processing
EVENT_1F68_MORNING_INTRO_DONE = 0x0020  # first outdoor morning CC 0x0C/0
EVENT_1F68_DOG_OWNED = 0x0080
EVENT_1F68_OUTDOOR_INTRO_MASK = (
    EVENT_1F68_TRUCK_BIT | EVENT_1F68_MORNING_INTRO_DONE | EVENT_1F68_DOG_OWNED
)  # 0x00A1


def farm_free_move_ready(ram: np.ndarray) -> bool:
    """True when outdoor free-move bit is set (can steer to shed/field).

    Verified 2026-08-09 (rr-bhr): ``Y1_Inside_House`` / ``Y1_Front_House`` exit
    keep ``game_state & 0x4000``. Pure D1 truck→sleep then ExitToFarm clears it
    (gs→0x0001) and auto-walks into the house-enter soft-lock.
    """
    try:
        flags = int(read_ram_value(ram, "game_state", raw=True))
    except Exception:
        return True
    return bool(flags & GAME_STATE_FREE_MOVE_BIT)


def outdoor_intro_flags_ready(ram: np.ndarray) -> bool:
    """True when event_flags_1f68 already has morning-intro + dog-owned bits.

    House→farm with only truck prereqs (``0x0011``) fires ROM morning intro
    (``CODE_83CEAE``): sets ``0x0020``, clears free-move, auto-walks to
    house-front dialogue, never recovers (controller-only mash/name attempts
    still → tilemap ``0x5F``). Pre-set ``0x00A1`` / Y1 ``0x00B1`` skips intro
    and keeps free-move. ``house_size`` is not causal.
    """
    try:
        flags = int(read_ram_value(ram, "event_flags_1f68", raw=True))
    except Exception:
        return True
    return (flags & EVENT_1F68_OUTDOOR_INTRO_MASK) == EVENT_1F68_OUTDOOR_INTRO_MASK


def farm_house_front_softlock(ram: np.ndarray) -> bool:
    """True when stuck in the post-truck house-front pocket without free move."""
    if not is_farm_tilemap(int(ram[ADDR_TILEMAP]) if ADDR_TILEMAP < len(ram) else 0):
        return False
    if farm_free_move_ready(ram):
        return False
    pos = get_pos_from_ram(ram)
    return pos.y >= HOUSE_FRONT_SOFTLOCK_Y_MIN and pos.x <= HOUSE_FRONT_SOFTLOCK_X_MAX
