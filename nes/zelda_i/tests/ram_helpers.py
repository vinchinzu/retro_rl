"""Shared 0x800 RAM-array builder for zelda_i unit tests.

Not a test module itself (no ``test_*`` names) — a plain import. Each
level's ``_ram(**fields)`` used to hand-roll the same
``np.zeros(0x800) + per-address assignment`` block; this collapses that
to one canonical name -> address table plus a per-file defaults dict.
"""

from __future__ import annotations

import numpy as np

from zelda_i.ram import (
    ADDR_ARROWS,
    ADDR_BOMBS,
    ADDR_BOW,
    ADDR_CANDLE,
    ADDR_COMPASS,
    ADDR_CUR_OPENED_DOORS,
    ADDR_FOOD,
    ADDR_HEALTH,
    ADDR_HELP_DROP_COUNT,
    ADDR_HELP_DROP_VALUE,
    ADDR_IS_UPDATING_MODE,
    ADDR_KEYS,
    ADDR_LADDER,
    ADDR_LEVEL,
    ADDR_LINK_FACING,
    ADDR_LINK_X,
    ADDR_LINK_Y,
    ADDR_MAGIC_KEY,
    ADDR_MAP,
    ADDR_MODE,
    ADDR_OPEN_DOORWAY_MASK,
    ADDR_RAFT,
    ADDR_ROD,
    ADDR_ROOM_ALL_DEAD,
    ADDR_ROOM_ITEM_ID,
    ADDR_RUPEES,
    ADDR_SCREEN,
    ADDR_COLLIDING_TILE,
    ADDR_SELECTED_ITEM,
    ADDR_SWORD,
    ADDR_TRIFORCE,
    ADDR_WHISTLE,
    ADDR_WORLD_KILL_COUNT,
)

FIELD_ADDR: dict[str, int] = {
    "mode": ADDR_MODE,
    "level": ADDR_LEVEL,
    "screen": ADDR_SCREEN,
    "x": ADDR_LINK_X,
    "y": ADDR_LINK_Y,
    "triforce": ADDR_TRIFORCE,
    "keys": ADDR_KEYS,
    "bombs": ADDR_BOMBS,
    "rupees": ADDR_RUPEES,
    "rod": ADDR_ROD,
    "bow": ADDR_BOW,
    "arrows": ADDR_ARROWS,
    "raft": ADDR_RAFT,
    "ladder": ADDR_LADDER,
    "health": ADDR_HEALTH,
    "item": ADDR_ROOM_ITEM_ID,
    "doors": ADDR_CUR_OPENED_DOORS,
    "mask": ADDR_OPEN_DOORWAY_MASK,
    "tile": ADDR_COLLIDING_TILE,
    "sword": ADDR_SWORD,
    "updating": ADDR_IS_UPDATING_MODE,
    "selected": ADDR_SELECTED_ITEM,
    "food": ADDR_FOOD,
    "whistle": ADDR_WHISTLE,
    "candle": ADDR_CANDLE,
    "facing": ADDR_LINK_FACING,
    "compass": ADDR_COMPASS,
    "room_all_dead": ADDR_ROOM_ALL_DEAD,
    "map": ADDR_MAP,
    "magic_key": ADDR_MAGIC_KEY,
    "room_item": ADDR_ROOM_ITEM_ID,
    "world_kill": ADDR_WORLD_KILL_COUNT,
    "help_count": ADDR_HELP_DROP_COUNT,
    "help_value": ADDR_HELP_DROP_VALUE,
}


def make_ram(defaults: dict[str, int], **fields: int) -> np.ndarray:
    """0x800 zero RAM array with ``defaults`` overridden by ``fields``.

    Every key (in ``defaults`` or ``fields``) must be in ``FIELD_ADDR``.
    """
    ram = np.zeros(0x800, dtype=np.uint8)
    for name, value in {**defaults, **fields}.items():
        ram[FIELD_ADDR[name]] = value
    return ram


__all__ = ["FIELD_ADDR", "make_ram"]
