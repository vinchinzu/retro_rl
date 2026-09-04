"""Level 7 cellar / secret-staircase tables from first-quest ROM.

Data Crystal stairway list (PRG ``0x19A18`` / iNES ``0x19A28``):
``0x7B, 0x4A, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF``.
First 6 bytes are ``LevelInfo_CellarRoomIdArray``; ``0xFF`` slots are unused.

CheckWarps (aldonunez ``Z_05.asm``): scan that array; the first cellar whose
``LevelBlockAttrsA/B`` equals the play ``RoomId`` is the dest, mode 9. In
cellars those bytes are dest room ids, not door codes. InitMode9: source ==
AttrA -> left ladder ``x=$30``, else right ``x=$C0``. CheckSubroom: ``Y<$40``
and UP; ``X<$80`` -> AttrA else AttrB.

ROM-claimed. Fixture-live dest play 0x29 from the 0x7B B-side spawn
(rr-n91a 2026-09-04, 2/2). Walk-on from play 0x0D is still unobserved.
"""

from __future__ import annotations

LEVEL7_STAIR_LIST_PRG = 0x19A18
LEVEL7_STAIR_LIST_INES = 0x19A28
LEVEL7_STAIR_LIST: tuple[int, ...] = (0x7B, 0x4A, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF)
LEVEL7_CELLAR_ROOMS: tuple[int, ...] = (0x7B, 0x4A)
NOSE_CELLAR_ROM = 0x7B
CANDLE_CELLAR_ROM = 0x4A
TIP_OF_NOSE_ROM = 0x0D
PRE_BOSS_ROM = 0x29
AQUAMENTUS_ROM = 0x2A
TRIFORCE_ROM = 0x2B
LEVEL7_CELLAR_DEST_LEFT: dict[int, int] = {0x7B: 0x29, 0x4A: 0x1A}
LEVEL7_CELLAR_DEST_RIGHT: dict[int, int] = {0x7B: 0x0D, 0x4A: 0x1A}
CELLAR_LADDER_LEFT_X = 0x30
CELLAR_LADDER_RIGHT_X = 0xC0
CHECKSUBROOM_SPLIT_X = 0x80
CHECKSUBROOM_MAX_Y = 0x40
# L7-9 door codes: 0 open, 1 wall, 2/3 false, 4 bomb, 5/6 key, 7 shutter.
# Values are (north, south, west, east, secret=AttrE&7).
ROM_DOORS: dict[int, tuple[int, int, int, int, int]] = {
    0x0C: (1, 0, 1, 4, 7),
    0x0D: (1, 1, 4, 1, 5),
    0x1A: (1, 1, 4, 4, 0),
    0x29: (1, 1, 1, 4, 0),
    0x2A: (1, 1, 4, 7, 7),
    0x2B: (1, 1, 0, 1, 0),
}
ROM_DOOR_NAMES: dict[int, str] = {
    0: "open", 1: "wall", 2: "false", 3: "false2",
    4: "bomb", 5: "key", 6: "key2", 7: "shutter",
}
ROM_SECRET_NAMES: dict[int, str] = {
    0: "none", 1: "all_dead", 5: "block_stairs", 7: "foes_item",
}
_NSWE = "nswe"


def cellar_dest_for(room: int, *, side: str = "left") -> int | None:
    table = LEVEL7_CELLAR_DEST_LEFT if side == "left" else LEVEL7_CELLAR_DEST_RIGHT
    return table.get(int(room) & 0xFF)


def play_rooms_entering_cellar(cellar: int) -> tuple[tuple[int, str], ...]:
    value = int(cellar) & 0xFF
    rows: list[tuple[int, str]] = []
    left = LEVEL7_CELLAR_DEST_LEFT.get(value)
    right = LEVEL7_CELLAR_DEST_RIGHT.get(value)
    if left is not None:
        rows.append((left, "left"))
    if right is not None and right != left:
        rows.append((right, "right"))
    elif right is not None and left is None:
        rows.append((right, "right"))
    return tuple(rows)


def cellar_for_play_room(play: int) -> tuple[int, str] | None:
    value = int(play) & 0xFF
    for cellar in LEVEL7_CELLAR_ROOMS:
        if LEVEL7_CELLAR_DEST_LEFT.get(cellar) == value:
            return (cellar, "left")
        if LEVEL7_CELLAR_DEST_RIGHT.get(cellar) == value:
            return (cellar, "right")
    return None


def other_play_endpoint(play: int) -> int | None:
    """Far-side play room of the cellar that CheckWarps would enter from ``play``."""
    found = cellar_for_play_room(play)
    if found is None:
        return None
    cellar, side = found
    if side == "left":
        return LEVEL7_CELLAR_DEST_RIGHT[cellar]
    return LEVEL7_CELLAR_DEST_LEFT[cellar]


def spawn_ladder_for_source(cellar: int, source: int) -> str:
    """InitMode9 ladder side when entering ``cellar`` from play ``source``."""
    if LEVEL7_CELLAR_DEST_LEFT.get(int(cellar) & 0xFF) == (int(source) & 0xFF):
        return "left"
    return "right"


def rom_door_name(code: int) -> str:
    return ROM_DOOR_NAMES.get(int(code) & 7, f"code_{int(code) & 7}")


def rom_side(room: int, side: str) -> int:
    return ROM_DOORS[int(room)][_NSWE.index(side[0])]


def rom_secret(room: int) -> int:
    return ROM_DOORS[int(room)][4]


def rom_secret_name(room: int) -> str:
    secret = rom_secret(room)
    return ROM_SECRET_NAMES.get(secret, f"secret_{secret}")


__all__ = [
    "AQUAMENTUS_ROM",
    "CANDLE_CELLAR_ROM",
    "CELLAR_LADDER_LEFT_X",
    "CELLAR_LADDER_RIGHT_X",
    "CHECKSUBROOM_MAX_Y",
    "CHECKSUBROOM_SPLIT_X",
    "LEVEL7_CELLAR_DEST_LEFT",
    "LEVEL7_CELLAR_DEST_RIGHT",
    "LEVEL7_CELLAR_ROOMS",
    "LEVEL7_STAIR_LIST",
    "LEVEL7_STAIR_LIST_INES",
    "LEVEL7_STAIR_LIST_PRG",
    "NOSE_CELLAR_ROM",
    "PRE_BOSS_ROM",
    "ROM_DOORS",
    "TIP_OF_NOSE_ROM",
    "TRIFORCE_ROM",
    "cellar_dest_for",
    "cellar_for_play_room",
    "other_play_endpoint",
    "play_rooms_entering_cellar",
    "rom_door_name",
    "rom_secret",
    "rom_secret_name",
    "rom_side",
    "spawn_ladder_for_source",
]
