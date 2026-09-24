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
    ADDR_MAX_BOMBS,
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
    ADDR_LINK_IFRAMES,
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
    "max_bombs": ADDR_MAX_BOMBS,
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
    "link_iframes": ADDR_LINK_IFRAMES,
}


def make_ram(defaults: dict[str, int], **fields: int) -> np.ndarray:
    """0x800 zero RAM array with ``defaults`` overridden by ``fields``.

    Every key (in ``defaults`` or ``fields``) must be in ``FIELD_ADDR``.
    """
    ram = np.zeros(0x800, dtype=np.uint8)
    for name, value in {**defaults, **fields}.items():
        ram[FIELD_ADDR[name]] = value
    return ram


__all__ = [
    "FIELD_ADDR",
    "make_ram",
    "room_tile_env",
    "room_tile_ram",
    "tile_map_env",
]


def room_tile_ram(room: str, level: int = 1) -> np.ndarray:
    """A RAM window carrying the captured ``$6530`` map for one room."""
    import json
    from pathlib import Path

    from zelda_i.dungeon import tilemap as tm

    path = Path(__file__).parent / "fixtures" / f"room_tiles_l{level}_{room}.json"
    tiles = json.loads(path.read_text())["tiles"]
    ram = np.zeros(tm.WRAM_RAM_OFFSET + 0x2000, dtype=np.uint8)
    start = tm.WRAM_RAM_OFFSET + tm.ADDR_ROOM_TILE_MAP - tm.WRAM_BASE
    ram[start : start + len(tiles)] = np.asarray(tiles, dtype=np.uint8)
    return ram


def room_tile_env(room: str, level: int = 1):
    """A stub env whose ``get_ram()`` carries a captured ``$6530`` tile map.

    Controllers with ``occupancy_from_tilemap`` measure their walls from the
    live map via ``bind_env``. A unit test that skips the bind gives them an
    empty grid — no geometry at all — which is how a 0x23 test came to expect
    a walk straight up into the water bar. Bind this instead so the room under
    test has exactly the walls the ROM has.
    """
    ram = room_tile_ram(room, level)

    class _Env:
        def get_ram(self):
            return ram

    return _Env()


def tile_map_env(solid: "set[tuple[int, int]]" = frozenset()):
    """A stub env carrying a *synthetic* ``$6530`` map: floor plus ``solid``.

    ``solid`` is a set of 16x16 cell origins in screen coords, each filled
    with a block tile. Use this only for rules that are about the *shape* of
    an obstacle (does the walker route around a wall at all); a rule about a
    specific room's walls belongs on ``room_tile_env`` with a captured map,
    because a hand-built map is exactly the drift ``blocked_link_cells``
    exists to remove.
    """
    from zelda_i.dungeon import tilemap as tm

    ram = np.zeros(tm.WRAM_RAM_OFFSET + 0x2000, dtype=np.uint8)
    start = tm.WRAM_RAM_OFFSET + tm.ADDR_ROOM_TILE_MAP - tm.WRAM_BASE
    tiles = np.full((tm.TILE_ROWS, tm.TILE_COLS), 0x74, dtype=np.uint8)
    for cx, cy in solid:
        col = int(cx) // tm.TILE_PX
        row = (int(cy) - tm.PLAYFIELD_TOP_Y) // tm.TILE_PX
        tiles[row : row + 2, col : col + 2] = 0xB0
    ram[start : start + tm.TILE_COLS * tm.TILE_ROWS] = tiles.T.reshape(-1)

    class _Env:
        def get_ram(self):
            return ram

    return _Env()
