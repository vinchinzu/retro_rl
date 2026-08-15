"""Level 9 stair-source geometry, ROM pairing, and stair-taking nav.

First-quest L9 stairway list (Data Crystal 0x19C10 / iNES 0x19C20):
``0x60, 0x70, 0x72, 0x75, 0x67, 0x77, 0x00, 0x4F``.  Sequential pairing is a
ROM hypothesis only; live SCREEN after walking the stairs is the truth.

``0x52`` is not a stair source.  Room ``0x62`` is disproved as a cardinal
predecessor (rr-sz8.3).  The route predecessor is whichever source actually
drops into live final Patra.
"""

from __future__ import annotations

from dataclasses import dataclass

from retro_harness.input_script import FrameAction
from retro_harness.nes import nes_action, nes_idle_action
from zelda_i.bomb_wall_path import BOMB_N_WAIT_BLAST, BombWallController
from zelda_i.combat import in_sword_hitbox
from zelda_i.dungeon_ids import object_name
from zelda_i.level2_puzzles import BombWall, DOOR_RIGHT
from zelda_i.level9_ganon import LEVEL9, ROOM_BEFORE_GANON
from zelda_i.level9_patra import PATRA_EYE_COUNT, final_patra_live, patra_eyes
from zelda_i.level9_path import NORTH_DOOR_X, NORTH_DOOR_X_TOL
from zelda_i.level9_room62 import LEVEL9_STAIR_SOURCES, door_bits
from zelda_i.ram import PLAY_MODE, ZeldaObject, ZeldaSnapshot

# Data Crystal 0x19C10 is iNES 0x19C20 (add 16-byte header).
LEVEL9_STAIR_LIST_INES = 0x19C20
LEVEL9_STAIR_LIST_DC = 0x19C10

# Sequential pairing of the 8-byte stairway list. Hypothesis only.
LEVEL9_STAIR_PAIRS: tuple[tuple[int, int], ...] = (
    (0x60, 0x70),
    (0x72, 0x75),
    (0x67, 0x77),
    (0x00, 0x4F),
)

# First 6 bytes of ROM 0x19C10 = LevelInfo_CellarRoomIdArray.
# CheckSubroom (mode 9): X < 0x80 reads AttrsA, else AttrsB, as dest room.
LEVEL9_CELLAR_ROOMS: tuple[int, ...] = (0x60, 0x70, 0x72, 0x75, 0x67, 0x77)
LEVEL9_CELLAR_DEST_LEFT: dict[int, int] = {
    0x60: 0x14,
    0x70: 0x63,
    0x72: 0x71,
    0x75: 0x20,
    0x67: 0x30,
    0x77: 0x52,  # live: left mouth → final Patra
}
LEVEL9_CELLAR_DEST_RIGHT: dict[int, int] = {
    0x60: 0x55,
    0x70: 0x05,
    0x72: 0x74,
    0x75: 0x61,
    0x67: 0x04,
    0x77: 0x03,  # live: right mouth → 0x03
}
PATRA_STAIR_SOURCE = 0x77

# Engine UW stairs revealed by a push-block secret (disassembly).
BLOCK_STAIRS_X = 0xD0
BLOCK_STAIRS_Y = 0x60

# Tiles 0x70-0x73 are dungeon stairs; 0x24 is a black cave mouth.
STAIR_TILE_LO = 0x70
STAIR_TILE_HI = 0x73
BLACK_MOUTH_TILE = 0x24

# Play rooms whose CheckWarps can enter a cellar (dest table reverse).
# 0x03 / 0x52 are the live AttrsB / AttrsA of cellar 0x77.
PLAY_STAIR_CANDIDATES: tuple[int, ...] = (
    0x03,
    0x04,
    0x13,
    0x07,
    0x30,
    0x52,
    0x76,
    0x67,
)

CELLAR_MODE = 9
STAIRS_EXIT_MODE = 10
ITEM_CELLAR_MODE = 11
STAIRS_ENTER_MODE = 16

CELLAR_EXIT_Y = 0x3D
CELLAR_LEFT_X = 0x50
CELLAR_RIGHT_X = 0xB0
CELLAR_SPLIT_X = 0x80
# Natural 0x03→0x77 entry: go DOWN the opposite mouth to the pit, then across.
CELLAR_CORRIDOR_Y = 0xB4  # live 0x77 right spawn (192,165) cannot cross until lower
PUSHABLE_BLOCK = 0x68
# Play room 0x03: west 0x68 at (96,144); CheckWarps stand is exact (128,141).
ROOM03_STAIR_X = 0x80
ROOM03_STAIR_Y = 0x8D
ROOM03_PUSH_X = 0x60
ROOM03_PUSH_Y = 170
ROOM03_SLOT_Y = 133
ROOM03_BLOCK_REST_Y = 0x90
ROOM03_BLOCK_PUSHED_Y = 0x80

# Walk-onto candidates after settle / kill-clear / block push.
STAIR_STANDS: tuple[tuple[int, int], ...] = (
    (0x78, 0x90),  # visible center stairs (live 0x03 / 0x04 layout 0x1A)
    (0x80, 0x90),
    (0x70, 0x90),
    (0x78, 0xA0),
    (0x78, 0x80),
    (0x88, 0x90),
    (0x68, 0x90),
    (BLOCK_STAIRS_X, BLOCK_STAIRS_Y),
    (0x60, 0x90),
    (0x90, 0x90),
    (0x78, 0x8D),
    (0x50, 0x80),
    (0xA0, 0x80),
    (0x40, 0xA0),
    (0xB0, 0xA0),
    (0x78, 0x70),
    (0x78, 0xB0),
)

# Stand just right of a typical center block and push LEFT.
BLOCK_PUSH_STANDS: tuple[tuple[int, int], ...] = (
    (0x88, 0x90),
    (0x80, 0x90),
    (0x78, 0x90),
    (0x90, 0x90),
    (0x70, 0x90),
    (0x88, 0x80),
    (0x88, 0xA0),
)

_PLAY_SLOTS = range(1, 13)
ALIGN_TOL = 3


@dataclass(frozen=True)
class StairLoader:
    """Neighbor-scroll seed that materializes one L9 stair-source room."""

    room: int
    from_room: int
    direction: str
    link_x: int
    link_y: int
    label: str


def paired_stair_dest(room: int) -> int | None:
    """Return the sequential-pair dest, or None if ``room`` is not a source."""
    value = int(room) & 0xFF
    for left, right in LEVEL9_STAIR_PAIRS:
        if value == left:
            return right
        if value == right:
            return left
    return None


def cellar_dest_for(room: int, *, side: str = "left") -> int | None:
    """AttrsA/B dest when ``room`` is the current mode-9 cellar RoomId."""
    table = LEVEL9_CELLAR_DEST_LEFT if side == "left" else LEVEL9_CELLAR_DEST_RIGHT
    return table.get(int(room) & 0xFF)


def is_patra_cellar_source(room: int) -> bool:
    return (int(room) & 0xFF) == PATRA_STAIR_SOURCE


def play_rooms_entering_cellar(cellar: int) -> tuple[tuple[int, str], ...]:
    """Play rooms whose stairs CheckWarps into ``cellar`` (AttrsA/B reverse)."""
    value = int(cellar) & 0xFF
    rows: list[tuple[int, str]] = []
    left = LEVEL9_CELLAR_DEST_LEFT.get(value)
    right = LEVEL9_CELLAR_DEST_RIGHT.get(value)
    if left is not None:
        rows.append((left, "left"))
    if right is not None:
        rows.append((right, "right"))
    return tuple(rows)


def cellar_for_play_room(play: int) -> tuple[int, str] | None:
    """Return (cellar RoomId, mouth side) if ``play`` is an AttrsA/B dest."""
    value = int(play) & 0xFF
    for cellar in LEVEL9_CELLAR_ROOMS:
        if LEVEL9_CELLAR_DEST_LEFT.get(cellar) == value:
            return (cellar, "left")
        if LEVEL9_CELLAR_DEST_RIGHT.get(cellar) == value:
            return (cellar, "right")
    return None


def cellar_mouth_xy(*, side: str = "left") -> tuple[int, int]:
    """Stand that satisfies CheckSubroom (Y < 0x40, mouth X)."""
    x = CELLAR_LEFT_X if side == "left" else CELLAR_RIGHT_X
    return (x, CELLAR_EXIT_Y)


# Enter through an open or false-wall door so Link is not trapped in a
# key-door alcove against a center block column (live 0x60 south mouth).
# ROM L7-9 door codes: 0 open, 2/3 false wall, 4 bomb, 5/6 key, 7 shutter, 1 wall.
_PREFERRED_LOADERS: dict[int, tuple[int, str, int, int]] = {
    0x60: (0x50, "DOWN", 0x78, 0xDD),   # north open; south key traps Link
    0x70: (0x60, "DOWN", 0x78, 0xDD),   # north false-wall; side scroll failed
    0x72: (0x62, "DOWN", 0x78, 0xDD),   # north false-wall
    0x75: (0x65, "DOWN", 0x78, 0xDD),   # north neighbor; east scroll failed
    0x67: (0x77, "UP", 0x78, 0x58),     # south neighbor
    0x77: (0x67, "DOWN", 0x78, 0xDD),   # north neighbor
    0x00: (0x10, "UP", 0x78, 0x58),     # forced south (N/W are off-grid)
    0x4F: (0x5F, "UP", 0x78, 0x58),     # south bomb, door-staging scroll
    # Play rooms (not cellar RoomIds). 0x03 N/S/W are walls; east is bomb from 0x04.
    # South staging from 0x13 lands on the push band; east 0x04 also works.
    0x03: (0x13, "UP", 0x78, 0x58),
    0x04: (0x14, "UP", 0x78, 0x58),     # south wall staged; 0x03 RIGHT failed
    # Play 0x30: N/W wall, S key, E bomb, secret block_stairs. South 0x40
    # key-north + Magical Key. Do not stage from 0x04 (would poke 0x04 doors).
    0x30: (0x40, "UP", 0x78, 0x58),
    # Play 0x40: N/S key, W/E wall. Stage 0x50, never 0x30.
    0x40: (0x50, "UP", 0x78, 0x58),
    # Play 0x20 north of 0x30: S wall. Default south-from-0x30 would poke 0x30.
    0x20: (0x10, "UP", 0x78, 0x58),
    # Play 0x31: W bomb pairs 0x30 E bomb. Stage 0x41, never 0x30.
    0x31: (0x41, "UP", 0x78, 0x58),
    # Play 0x21: S shutter pairs 0x31 N open. Stage 0x11, never 0x31.
    0x21: (0x11, "DOWN", 0x78, 0xDD),
    # Play 0x41: N open / S shutter / W/E wall. Stage 0x51, never 0x31.
    0x41: (0x51, "UP", 0x78, 0x58),
    0x13: (0x23, "UP", 0x78, 0x58),
    0x07: (0x17, "UP", 0x78, 0x58),
    0x52: (0x62, "UP", 0x78, 0x58),
    0x76: (0x66, "DOWN", 0x78, 0xDD),
}

# First-quest L7-9 door codes (Data Crystal 0x18A00/0x18A80): 0 open, 1 wall,
# 2/3 false, 4 bomb, 5/6 key, 7 shutter.  0x13 north and 0x03 south are walls.
ROOM13 = 0x13
ROOM03 = 0x03
ROOM13_ROM_NORTH = 1  # wall
ROOM13_ROM_SOUTH = 5  # key
ROOM13_ROM_WEST = 5  # key
ROOM13_ROM_EAST = 1  # wall
ROOM03_ROM_NORTH = 1  # wall
ROOM03_ROM_SOUTH = 1  # wall
ROOM03_ROM_WEST = 1  # wall
ROOM03_ROM_EAST = 4  # bomb from 0x04
ROOM04 = 0x04
ROOM04_ROM_NORTH = 1  # wall
ROOM04_ROM_SOUTH = 1  # wall
ROOM04_ROM_WEST = 4  # bomb into 0x03
ROOM04_ROM_EAST = 1  # wall
ROOM02 = 0x02
ROOM02_ROM_NORTH = 1  # wall
ROOM02_ROM_SOUTH = 0  # open
ROOM02_ROM_WEST = 5  # key
ROOM02_ROM_EAST = 1  # wall
# Typical west bomb stand (left of center @ door y). Measure live.
ROOM04_BOMB_WEST_STAND = (48, 141)
ROOM04_BOMB_WEST_APPROACH = (48, 189)  # south-band around the plus / 0x68
ROOM30 = 0x30
ROOM40 = 0x40
ROOM30_ROM_NORTH = 1  # wall
ROOM30_ROM_SOUTH = 5  # key
ROOM30_ROM_WEST = 1  # wall
ROOM30_ROM_EAST = 4  # bomb
ROOM30_ROM_SECRET = 5  # block_stairs
ROOM40_ROM_NORTH = 5  # key (0x30 south pair)
ROOM40_ROM_SOUTH = 5  # key
ROOM40_ROM_WEST = 1  # wall
ROOM40_ROM_EAST = 1  # wall
ROOM40_ROM_SECRET = 7  # foes_item
ROOM20 = 0x20
ROOM20_ROM_NORTH = 4  # bomb
ROOM20_ROM_SOUTH = 1  # wall
ROOM20_ROM_WEST = 1  # wall
ROOM20_ROM_EAST = 1  # wall
ROOM2F = 0x2F
ROOM2F_ROM_NORTH = 2  # false
ROOM2F_ROM_SOUTH = 3  # false2
ROOM2F_ROM_WEST = 1  # wall
ROOM2F_ROM_EAST = 7  # shutter
ROOM31 = 0x31
ROOM31_ROM_NORTH = 0  # open
ROOM31_ROM_SOUTH = 7  # shutter
ROOM31_ROM_WEST = 4  # bomb (0x30 east pair)
ROOM31_ROM_EAST = 1  # wall
ROOM31_ROM_SECRET = 0  # none
ROOM41 = 0x41  # south of 0x31; north open. Loader stages 0x51, never 0x31.
ROOM41_ROM_NORTH = 0  # open (pairs 0x31 south shutter)
ROOM41_ROM_SOUTH = 7  # shutter
ROOM41_ROM_WEST = 1  # wall
ROOM41_ROM_EAST = 1  # wall
ROOM41_ROM_SECRET = 0  # none
ROOM51 = 0x51  # south of 0x41; north open. 0x41 loader stages here.
ROOM21 = 0x21
ROOM21_ROM_NORTH = 0  # open
ROOM21_ROM_SOUTH = 7  # shutter (0x31 north is open)
ROOM21_ROM_WEST = 1  # wall
ROOM21_ROM_EAST = 4  # bomb
ROOM21_ROM_SECRET = 0  # none
ROOM11 = 0x11  # north neighbor; south shutter. Loader stages 0x11, never 0x31.
# Plus layout (same D=0xA5 as 0x31): west-band around the plus, then south door.
ROOM21_SOUTH_Y = 189
ROOM21_WEST_X = 48
# Same west-bomb stand as 0x04. Measure live if 0x31 layout blocks it.
ROOM31_BOMB_WEST_STAND = (48, 141)
ROOM31_BOMB_WEST_APPROACH = (48, 189)
# Typical UW block-stairs reveal after kill-clear + push (disassembly).
ROOM30_STAIR_X = BLOCK_STAIRS_X
ROOM30_STAIR_Y = BLOCK_STAIRS_Y
ROOM30_PUSH_X = 0x60
ROOM30_PUSH_Y = 170
ROOM30_BLOCK_REST = (0x60, 0x90)
CELLAR_67 = 0x67
ROM_DOOR_NAMES: dict[int, str] = {
    0: "open",
    1: "wall",
    2: "false",
    3: "false2",
    4: "bomb",
    5: "key",
    6: "key2",
    7: "shutter",
}

BOMB_WALL_04_WEST = BombWall(
    room=ROOM04,
    stand=ROOM04_BOMB_WEST_STAND,
    face="LEFT",
    opens_to=ROOM03,
    opened_door_bit=DOOR_RIGHT,
    live=True,
    notes="LIVE: west bomb (48,141) LEFT opens 0x03 east hole (rr-sz8.3).",
)

BOMB_WALL_31_WEST = BombWall(
    room=ROOM31,
    stand=ROOM31_BOMB_WEST_STAND,
    face="LEFT",
    opens_to=ROOM30,
    opened_door_bit=DOOR_RIGHT,
    live=True,
    notes="LIVE: west bomb (48,141) LEFT opens 0x30 east hole (rr-sz8.3).",
)


def stair_loader_for(room: int) -> StairLoader:
    """Neighbor-scroll seed that lands Link in a walkable doorway."""
    value = int(room) & 0xFF
    if value in _PREFERRED_LOADERS:
        from_room, direction, link_x, link_y = _PREFERRED_LOADERS[value]
        return StairLoader(
            room=value,
            from_room=from_room,
            direction=direction,
            link_x=link_x,
            link_y=link_y,
            label=f"0x{from_room:02x}_hold_{direction.lower()}",
        )
    row = (value >> 4) & 0x0F
    if row < 7:
        from_room = value + 0x10
        return StairLoader(
            room=value,
            from_room=from_room,
            direction="UP",
            link_x=0x78,
            link_y=0x58,
            label=f"south_0x{from_room:02x}_hold_up",
        )
    from_room = value - 0x10
    return StairLoader(
        room=value,
        from_room=from_room,
        direction="DOWN",
        link_x=0x78,
        link_y=0xDD,
        label=f"north_0x{from_room:02x}_hold_down",
    )


def in_stair_source(snap: ZeldaSnapshot, room: int) -> bool:
    return (
        snap.mode == PLAY_MODE
        and snap.level == LEVEL9
        and snap.screen == int(room)
    )


def stair_room_objects(snap: ZeldaSnapshot) -> tuple[ZeldaObject, ...]:
    return tuple(
        obj
        for obj in snap.objects
        if obj.slot in _PLAY_SLOTS and (obj.type_id or obj.hp)
    )


def stair_object_summary(snap: ZeldaSnapshot) -> list[dict[str, int | str]]:
    return [
        {
            "slot": obj.slot,
            "type_id": obj.type_id,
            "type_name": object_name(obj.type_id),
            "hp": obj.hp,
            "x": obj.x,
            "y": obj.y,
            "state": obj.state,
        }
        for obj in stair_room_objects(snap)
    ]


def live_combat_objects(snap: ZeldaSnapshot) -> tuple[ZeldaObject, ...]:
    """Enemies worth chasing; skip bubbles / traps / residual types >= 0x40."""
    return tuple(
        obj
        for obj in stair_room_objects(snap)
        if 0x01 <= obj.type_id < 0x40 and obj.type_id not in (0x37, 0x3F)
    )


def on_stair_tile(snap: ZeldaSnapshot) -> bool:
    tile = int(snap.colliding_tile)
    return STAIR_TILE_LO <= tile <= STAIR_TILE_HI


def on_warp_tile(snap: ZeldaSnapshot) -> bool:
    tile = int(snap.colliding_tile)
    return on_stair_tile(snap) or tile == BLACK_MOUTH_TILE


def in_patra_cellar(snap: ZeldaSnapshot) -> bool:
    """True when the engine has entered cellar RoomId 0x77."""
    return (
        snap.level == LEVEL9
        and snap.screen == PATRA_STAIR_SOURCE
        and stair_transition_modes(snap.mode)
    )


def in_underworld_passage(snap: ZeldaSnapshot) -> bool:
    return snap.mode in (CELLAR_MODE, STAIRS_EXIT_MODE, ITEM_CELLAR_MODE)


def stair_transition_modes(mode: int) -> bool:
    return int(mode) in (
        CELLAR_MODE,
        STAIRS_EXIT_MODE,
        ITEM_CELLAR_MODE,
        STAIRS_ENTER_MODE,
    )


def landed_final_patra(snap: ZeldaSnapshot) -> bool:
    """True when the game stair loader has settled live Patra in 0x52."""
    return (
        final_patra_live(snap)
        and len(patra_eyes(snap)) == PATRA_EYE_COUNT
        and not bool(snap.cur_opened_doors & 0x08)
        and not bool(snap.open_doorway_mask & 0x08)
    )


def dest_report(snap: ZeldaSnapshot) -> dict[str, object]:
    return {
        "screen": int(snap.screen),
        "next_screen": int(snap.next_screen),
        "mode": int(snap.mode),
        "link": {"x": snap.link_x, "y": snap.link_y},
        "doors": door_bits(snap.cur_opened_doors),
        "mask": door_bits(snap.open_doorway_mask),
        "objects": stair_object_summary(snap),
        "final_patra_live": bool(final_patra_live(snap)),
        "patra_eyes": len(patra_eyes(snap)),
        "landed_final_patra": bool(landed_final_patra(snap)),
        "room_item_id": int(snap.room_item_id),
        "colliding_tile": int(snap.colliding_tile),
        "room_all_dead": int(snap.room_all_dead),
        "room_obj_count": int(snap.room_obj_count),
    }


def walk_to_step(
    snap: ZeldaSnapshot,
    x: int,
    y: int,
    *,
    y_first: bool = True,
    tol: int | None = None,
) -> FrameAction:
    limit = ALIGN_TOL if tol is None else int(tol)
    dx = int(x) - int(snap.link_x)
    dy = int(y) - int(snap.link_y)
    if abs(dx) <= limit and abs(dy) <= limit:
        return FrameAction(nes_idle_action(), "walk_arrived")
    if y_first and abs(dy) > limit:
        return FrameAction(nes_action("DOWN" if dy > 0 else "UP"), "walk_align_y")
    if abs(dx) > limit:
        return FrameAction(nes_action("RIGHT" if dx > 0 else "LEFT"), "walk_align_x")
    return FrameAction(nes_action("DOWN" if dy > 0 else "UP"), "walk_align_y")


def pushable_block(snap: ZeldaSnapshot) -> ZeldaObject | None:
    """Room 0x68 block object, if present."""
    for obj in stair_room_objects(snap):
        if obj.type_id == PUSHABLE_BLOCK:
            return obj
    return None


def in_room_13(snap: ZeldaSnapshot) -> bool:
    return in_stair_source(snap, ROOM13)


def room13_is_clean_predecessor_of_03() -> bool:
    """ROM: 0x13 north is a wall; 0x03 south is a wall."""
    return False


def in_room_04(snap: ZeldaSnapshot) -> bool:
    return in_stair_source(snap, ROOM04)


def in_room_30(snap: ZeldaSnapshot) -> bool:
    return in_stair_source(snap, ROOM30)


def room30_rom_secret_is_block_stairs() -> bool:
    return ROOM30_ROM_SECRET == 5


def room30_loader_avoids_04() -> bool:
    """True when the 0x30 neighbor-scroll does not stage 0x04 doors."""
    return stair_loader_for(ROOM30).from_room != ROOM04


def in_room_40(snap: ZeldaSnapshot) -> bool:
    return in_stair_source(snap, ROOM40)


def room40_rom_north_is_key() -> bool:
    return ROOM40_ROM_NORTH == 5 and ROOM30_ROM_SOUTH == 5


def room40_is_rom_predecessor_of_30() -> bool:
    """ROM only: 0x40 north and 0x30 south are key. Live dest is separate."""
    return room40_rom_north_is_key()


def room40_loader_avoids_30() -> bool:
    """True when the 0x40 neighbor-scroll does not stage 0x30 doors."""
    return stair_loader_for(ROOM40).from_room != ROOM30


def in_room_31(snap: ZeldaSnapshot) -> bool:
    return in_stair_source(snap, ROOM31)


def room31_rom_west_is_bomb() -> bool:
    return ROOM31_ROM_WEST == 4 and ROOM30_ROM_EAST == 4


def room31_is_rom_predecessor_of_30() -> bool:
    """ROM only: 0x31 west and 0x30 east are bomb. Live dest is separate."""
    return room31_rom_west_is_bomb()


def room31_loader_avoids_30() -> bool:
    """True when the 0x31 neighbor-scroll does not stage 0x30 doors."""
    return stair_loader_for(ROOM31).from_room != ROOM30


def in_room_21(snap: ZeldaSnapshot) -> bool:
    return in_stair_source(snap, ROOM21)


def room21_rom_south_is_shutter() -> bool:
    return ROOM21_ROM_SOUTH == 7 and ROOM31_ROM_NORTH == 0


def room21_is_rom_predecessor_of_31() -> bool:
    """ROM only: 0x21 south shutter pairs 0x31 north open. Live dest separate."""
    return room21_rom_south_is_shutter()


def room21_loader_avoids_31() -> bool:
    """True when the 0x21 neighbor-scroll does not stage 0x31 doors."""
    return stair_loader_for(ROOM21).from_room != ROOM31


def room21_to_31_step(snap: ZeldaSnapshot) -> FrameAction:
    """West-band around the plus, then DOWN the 0x21 south shutter → 0x31.

    No door poke on 0x31. Shutter may need a kill-clear first; the step
    itself is a free south push once the door is walkable.
    """
    if snap.level != LEVEL9:
        return FrameAction(nes_idle_action(), "wait_level9")
    if snap.transitioning or snap.mode in (4, 6, 7):
        return FrameAction(nes_action("DOWN"), "room31_scroll")
    if snap.mode != PLAY_MODE:
        return FrameAction(nes_idle_action(), f"wait_mode_{snap.mode}")
    if snap.screen == ROOM31:
        return FrameAction(nes_idle_action(), "room31_arrived")
    if snap.screen != ROOM21:
        return FrameAction(
            nes_idle_action(),
            f"unexpected_room_0x{snap.screen:02x}",
        )
    x = int(snap.link_x)
    y = int(snap.link_y)
    if y < ROOM21_SOUTH_Y - 6:
        if abs(x - NORTH_DOOR_X) <= NORTH_DOOR_X_TOL:
            return FrameAction(nes_action("LEFT"), "room21_off_plus")
        if abs(x - ROOM21_WEST_X) > ALIGN_TOL:
            return walk_to_step(snap, ROOM21_WEST_X, y, y_first=False)
        return walk_to_step(snap, ROOM21_WEST_X, ROOM21_SOUTH_Y, y_first=True)
    if abs(x - NORTH_DOOR_X) > NORTH_DOOR_X_TOL:
        direction = "LEFT" if x > NORTH_DOOR_X else "RIGHT"
        return FrameAction(nes_action(direction), "room21_align_x")
    return FrameAction(nes_action("DOWN"), "room21_push_south")


def in_room_41(snap: ZeldaSnapshot) -> bool:
    return in_stair_source(snap, ROOM41)


def room41_rom_north_is_open() -> bool:
    return ROOM41_ROM_NORTH == 0 and ROOM31_ROM_SOUTH == 7


def room41_is_rom_predecessor_of_31() -> bool:
    """ROM only: 0x41 north open pairs 0x31 south shutter. Live dest separate."""
    return room41_rom_north_is_open()


def room41_loader_avoids_31() -> bool:
    """True when the 0x41 neighbor-scroll does not stage 0x31 doors."""
    return stair_loader_for(ROOM41).from_room != ROOM31


# Live 0x41: south alcove after 0x51 UP loader; north is ROM-open.
ROOM41_SOUTH_Y = 189


def room41_to_31_step(snap: ZeldaSnapshot) -> FrameAction:
    """Door-column UP through the 0x41 north open door → hypothesized 0x31.

    No door poke on 0x31.
    """
    if snap.level != LEVEL9:
        return FrameAction(nes_idle_action(), "wait_level9")
    if snap.transitioning or snap.mode in (4, 6, 7):
        return FrameAction(nes_action("UP"), "room31_scroll")
    if snap.mode != PLAY_MODE:
        return FrameAction(nes_idle_action(), f"wait_mode_{snap.mode}")
    if snap.screen == ROOM31:
        return FrameAction(nes_idle_action(), "room31_arrived")
    if snap.screen != ROOM41:
        return FrameAction(
            nes_idle_action(),
            f"unexpected_room_0x{snap.screen:02x}",
        )
    x = int(snap.link_x)
    y = int(snap.link_y)
    if abs(x - NORTH_DOOR_X) > NORTH_DOOR_X_TOL and y < ROOM41_SOUTH_Y - 6:
        return walk_to_step(snap, x, ROOM41_SOUTH_Y, y_first=True)
    if abs(x - NORTH_DOOR_X) > NORTH_DOOR_X_TOL:
        direction = "LEFT" if x > NORTH_DOOR_X else "RIGHT"
        return FrameAction(nes_action(direction), "room41_align_x")
    return FrameAction(nes_action("UP"), "room41_push_north")


# Live 0x40: south alcove (120,205) UP the door column reaches the key door.
# Off-column after combat: drop to y=189, then recenter x=120. Mode 4/6/7
# are scroll; keep holding UP (idle during mode 4 aborts the transition).
ROOM40_SOUTH_Y = 189


def room40_to_30_step(snap: ZeldaSnapshot) -> FrameAction:
    """Door-column UP through the 0x40 north key door → hypothesized 0x30.

    Magical Key is in the fixture loadout.  No door poke on 0x30.
    """
    if snap.level != LEVEL9:
        return FrameAction(nes_idle_action(), "wait_level9")
    if snap.transitioning or snap.mode in (4, 6, 7):
        return FrameAction(nes_action("UP"), "room30_scroll")
    if snap.mode != PLAY_MODE:
        return FrameAction(nes_idle_action(), f"wait_mode_{snap.mode}")
    if snap.screen == ROOM30:
        return FrameAction(nes_idle_action(), "room30_arrived")
    if snap.screen != ROOM40:
        return FrameAction(
            nes_idle_action(),
            f"unexpected_room_0x{snap.screen:02x}",
        )
    x = int(snap.link_x)
    y = int(snap.link_y)
    if abs(x - NORTH_DOOR_X) > NORTH_DOOR_X_TOL and y < ROOM40_SOUTH_Y - 6:
        return walk_to_step(snap, x, ROOM40_SOUTH_Y, y_first=True)
    if abs(x - NORTH_DOOR_X) > NORTH_DOOR_X_TOL:
        direction = "LEFT" if x > NORTH_DOOR_X else "RIGHT"
        return FrameAction(nes_action(direction), "room40_align_x")
    return FrameAction(nes_action("UP"), "room40_push_north")


def room30_block_secret_open(snap: ZeldaSnapshot) -> bool:
    """True after the 0x30 0x68 has relocated to the engine stairs stand."""
    block = pushable_block(snap)
    return block is not None and int(block.x) >= 0xC0 and int(block.y) <= 0x70


def in_cellar_67(snap: ZeldaSnapshot) -> bool:
    return (
        snap.level == LEVEL9
        and snap.screen == CELLAR_67
        and stair_transition_modes(snap.mode)
    )


def room30_stairs_step(snap: ZeldaSnapshot) -> FrameAction:
    """Controller-only 0x30: push west 0x68 UP, then exact stand (208, 96).

    CheckWarps misses ALIGN_TOL=3 (206,93 stays in play). tol=0 like 0x03.
    """
    if snap.level != LEVEL9:
        return FrameAction(nes_idle_action(), "wait_level9")
    if snap.mode == 17:
        return FrameAction(nes_idle_action(), "link_death")
    if stair_transition_modes(snap.mode) or snap.transitioning:
        return FrameAction(nes_action("UP"), "stairs_transition")
    if snap.mode != PLAY_MODE:
        return FrameAction(nes_idle_action(), f"wait_mode_{snap.mode}")
    if snap.screen != ROOM30:
        return FrameAction(nes_idle_action(), f"left_source_0x{snap.screen:02x}")
    if int(snap.link_x) == ROOM30_STAIR_X and int(snap.link_y) == ROOM30_STAIR_Y:
        return FrameAction(nes_idle_action(), "stand_on_30_stairs")
    if not room30_block_secret_open(snap):
        on_col = abs(int(snap.link_x) - ROOM30_PUSH_X) <= 2
        if not on_col and snap.link_y < 180:
            return walk_to_step(snap, int(snap.link_x), 189, y_first=True)
        if not on_col:
            return walk_to_step(snap, ROOM30_PUSH_X, 189, y_first=False)
        if int(snap.link_y) > 172:
            return walk_to_step(
                snap, ROOM30_PUSH_X, ROOM30_PUSH_Y, y_first=True, tol=0
            )
        if abs(int(snap.link_x) - ROOM30_PUSH_X) > 1:
            return walk_to_step(
                snap, ROOM30_PUSH_X, ROOM30_PUSH_Y, y_first=False, tol=0
            )
        return FrameAction(nes_action("UP"), "push_30_west_block_up")
    if int(snap.link_y) < 180 and int(snap.link_x) < 200:
        return walk_to_step(snap, int(snap.link_x), 189, y_first=True)
    if abs(int(snap.link_x) - ROOM30_STAIR_X) > 0:
        return walk_to_step(snap, ROOM30_STAIR_X, 189, y_first=False, tol=0)
    return walk_to_step(
        snap, ROOM30_STAIR_X, ROOM30_STAIR_Y, y_first=True, tol=0
    )


def room04_rom_west_is_bomb() -> bool:
    return ROOM04_ROM_WEST == 4 and ROOM03_ROM_EAST == 4


def room04_is_rom_predecessor_of_03() -> bool:
    """ROM only: 0x04 west and 0x03 east are bomb. Live dest is separate."""
    return room04_rom_west_is_bomb()


def rom_door_name(code: int) -> str:
    return ROM_DOOR_NAMES.get(int(code) & 7, f"code_{int(code) & 7}")


def room03_rom_neighbors() -> tuple[dict[str, object], ...]:
    """Cardinal ROM neighbors of play 0x03 (no north: row 0)."""
    return (
        {
            "room": ROOM03,
            "dir": "self",
            "n": ROOM03_ROM_NORTH,
            "s": ROOM03_ROM_SOUTH,
            "w": ROOM03_ROM_WEST,
            "e": ROOM03_ROM_EAST,
            "n_name": rom_door_name(ROOM03_ROM_NORTH),
            "s_name": rom_door_name(ROOM03_ROM_SOUTH),
            "w_name": rom_door_name(ROOM03_ROM_WEST),
            "e_name": rom_door_name(ROOM03_ROM_EAST),
            "secret": "none",
        },
        {
            "room": ROOM13,
            "dir": "south",
            "n": ROOM13_ROM_NORTH,
            "s": ROOM13_ROM_SOUTH,
            "w": ROOM13_ROM_WEST,
            "e": ROOM13_ROM_EAST,
            "n_name": rom_door_name(ROOM13_ROM_NORTH),
            "s_name": rom_door_name(ROOM13_ROM_SOUTH),
            "w_name": rom_door_name(ROOM13_ROM_WEST),
            "e_name": rom_door_name(ROOM13_ROM_EAST),
            "secret": "none",
            "note": "disproved: 0x13 north wall / 0x03 south wall",
        },
        {
            "room": ROOM02,
            "dir": "west",
            "n": ROOM02_ROM_NORTH,
            "s": ROOM02_ROM_SOUTH,
            "w": ROOM02_ROM_WEST,
            "e": ROOM02_ROM_EAST,
            "n_name": rom_door_name(ROOM02_ROM_NORTH),
            "s_name": rom_door_name(ROOM02_ROM_SOUTH),
            "w_name": rom_door_name(ROOM02_ROM_WEST),
            "e_name": rom_door_name(ROOM02_ROM_EAST),
            "secret": "none",
            "note": "0x02 east wall / 0x03 west wall",
        },
        {
            "room": ROOM04,
            "dir": "east",
            "n": ROOM04_ROM_NORTH,
            "s": ROOM04_ROM_SOUTH,
            "w": ROOM04_ROM_WEST,
            "e": ROOM04_ROM_EAST,
            "n_name": rom_door_name(ROOM04_ROM_NORTH),
            "s_name": rom_door_name(ROOM04_ROM_SOUTH),
            "w_name": rom_door_name(ROOM04_ROM_WEST),
            "e_name": rom_door_name(ROOM04_ROM_EAST),
            "secret": "none",
            "note": "candidate: 0x04 west bomb / 0x03 east bomb",
        },
    )


def room30_rom_neighbors() -> tuple[dict[str, object], ...]:
    """Cardinal ROM neighbors of play 0x30 plus cellar 0x67 (successor)."""
    return (
        {
            "room": ROOM30,
            "dir": "self",
            "n": ROOM30_ROM_NORTH,
            "s": ROOM30_ROM_SOUTH,
            "w": ROOM30_ROM_WEST,
            "e": ROOM30_ROM_EAST,
            "n_name": rom_door_name(ROOM30_ROM_NORTH),
            "s_name": rom_door_name(ROOM30_ROM_SOUTH),
            "w_name": rom_door_name(ROOM30_ROM_WEST),
            "e_name": rom_door_name(ROOM30_ROM_EAST),
            "secret": "block_stairs",
        },
        {
            "room": ROOM20,
            "dir": "north",
            "n": ROOM20_ROM_NORTH,
            "s": ROOM20_ROM_SOUTH,
            "w": ROOM20_ROM_WEST,
            "e": ROOM20_ROM_EAST,
            "n_name": rom_door_name(ROOM20_ROM_NORTH),
            "s_name": rom_door_name(ROOM20_ROM_SOUTH),
            "w_name": rom_door_name(ROOM20_ROM_WEST),
            "e_name": rom_door_name(ROOM20_ROM_EAST),
            "secret": "none",
            "note": "disproved ROM: 0x20 south wall / 0x30 north wall",
        },
        {
            "room": ROOM40,
            "dir": "south",
            "n": ROOM40_ROM_NORTH,
            "s": ROOM40_ROM_SOUTH,
            "w": ROOM40_ROM_WEST,
            "e": ROOM40_ROM_EAST,
            "n_name": rom_door_name(ROOM40_ROM_NORTH),
            "s_name": rom_door_name(ROOM40_ROM_SOUTH),
            "w_name": rom_door_name(ROOM40_ROM_WEST),
            "e_name": rom_door_name(ROOM40_ROM_EAST),
            "secret": "foes_item",
            "note": "candidate: 0x40 north key / 0x30 south key",
        },
        {
            "room": ROOM2F,
            "dir": "west",
            "n": ROOM2F_ROM_NORTH,
            "s": ROOM2F_ROM_SOUTH,
            "w": ROOM2F_ROM_WEST,
            "e": ROOM2F_ROM_EAST,
            "n_name": rom_door_name(ROOM2F_ROM_NORTH),
            "s_name": rom_door_name(ROOM2F_ROM_SOUTH),
            "w_name": rom_door_name(ROOM2F_ROM_WEST),
            "e_name": rom_door_name(ROOM2F_ROM_EAST),
            "secret": "none",
            "note": "0x2F east shutter / 0x30 west wall",
        },
        {
            "room": ROOM31,
            "dir": "east",
            "n": ROOM31_ROM_NORTH,
            "s": ROOM31_ROM_SOUTH,
            "w": ROOM31_ROM_WEST,
            "e": ROOM31_ROM_EAST,
            "n_name": rom_door_name(ROOM31_ROM_NORTH),
            "s_name": rom_door_name(ROOM31_ROM_SOUTH),
            "w_name": rom_door_name(ROOM31_ROM_WEST),
            "e_name": rom_door_name(ROOM31_ROM_EAST),
            "secret": "none",
            "note": "LIVE dest: 0x31 west bomb → 0x30 (rr-sz8.3)",
        },
        {
            "room": CELLAR_67,
            "dir": "cellar_successor",
            "secret": "none",
            "note": "cellar 0x67 left→0x30 right→0x04; successor path, not pred",
        },
    )


def make_room04_bomb_west_controller() -> BombWallController:
    """0x04 west bomb wall → hypothesized 0x03. No new phase machine."""
    return BombWallController(
        wall=BOMB_WALL_04_WEST,
        level=LEVEL9,
        clear_spec=None,
        face_frames=6,
        step_back=0,
        wait_blast=BOMB_N_WAIT_BLAST,
        require_bomb_consumed=False,
        wait_hold_face=True,
        max_frames=8000,
    )


def room04_bomb_west_approach_step(snap: ZeldaSnapshot) -> FrameAction:
    """South-band around the plus / west 0x68, then up the west corridor.

    Naive LEFT at y≈149 hits the west block at (96,144).
    """
    ax, ay = ROOM04_BOMB_WEST_APPROACH
    sx, sy = ROOM04_BOMB_WEST_STAND
    if abs(int(snap.link_x) - ax) > ALIGN_TOL or int(snap.link_y) < ay - 6:
        return walk_to_step(snap, ax, ay, y_first=True)
    return walk_to_step(snap, sx, sy, y_first=True)


def make_room31_bomb_west_controller() -> BombWallController:
    """0x31 west bomb wall → hypothesized 0x30. No new phase machine."""
    return BombWallController(
        wall=BOMB_WALL_31_WEST,
        level=LEVEL9,
        clear_spec=None,
        face_frames=6,
        step_back=0,
        wait_blast=BOMB_N_WAIT_BLAST,
        require_bomb_consumed=False,
        wait_hold_face=True,
        max_frames=8000,
        stand_timeout=4000,
    )


def room31_bomb_west_approach_step(snap: ZeldaSnapshot) -> FrameAction:
    """South-band then up the west corridor. Same stand as 0x04 bomb-west."""
    ax, ay = ROOM31_BOMB_WEST_APPROACH
    sx, sy = ROOM31_BOMB_WEST_STAND
    if abs(int(snap.link_x) - ax) > ALIGN_TOL or int(snap.link_y) < ay - 6:
        return walk_to_step(snap, ax, ay, y_first=True)
    return walk_to_step(snap, sx, sy, y_first=True)


def pause_select_next_b_item_script() -> tuple[FrameAction, ...]:
    """Open pause, move B-cursor one slot right (bombs → bow/arrows), close."""
    frames: list[FrameAction] = [FrameAction(nes_action("START"), "pause_open")]
    frames.extend(FrameAction(nes_idle_action(), "pause_settle") for _ in range(24))
    frames.append(FrameAction(nes_action("RIGHT"), "pause_next_item"))
    frames.extend(FrameAction(nes_idle_action(), "pause_cursor") for _ in range(8))
    frames.append(FrameAction(nes_action("START"), "pause_close"))
    frames.extend(FrameAction(nes_idle_action(), "pause_resume") for _ in range(24))
    return tuple(frames)


def room13_to_03_step(snap: ZeldaSnapshot) -> FrameAction:
    """One frame of a north push 0x13 → 0x03.

    Kept as the nav micro that would apply IF a north door existed.  ROM
    and live both say it does not; a stuck north-wall finish is a
    retarget, not a success.
    """
    if snap.level != LEVEL9:
        return FrameAction(nes_idle_action(), "wait_level9")
    if snap.transitioning:
        return FrameAction(nes_action("UP"), "room03_scroll")
    if snap.mode != PLAY_MODE:
        return FrameAction(nes_idle_action(), f"wait_mode_{snap.mode}")
    if snap.screen == ROOM03:
        return FrameAction(nes_idle_action(), "room03_arrived")
    if snap.screen != ROOM13:
        return FrameAction(
            nes_idle_action(),
            f"unexpected_room_0x{snap.screen:02x}",
        )
    if abs(int(snap.link_x) - NORTH_DOOR_X) > NORTH_DOOR_X_TOL:
        direction = "LEFT" if snap.link_x > NORTH_DOOR_X else "RIGHT"
        return FrameAction(nes_action(direction), "room13_align_x")
    return FrameAction(nes_action("UP"), "room13_push_north")


def room03_west_block_pushed(snap: ZeldaSnapshot) -> bool:
    """True after the 0x03 west block has slid UP one tile (y 0x90 → 0x80)."""
    block = pushable_block(snap)
    return block is not None and int(block.y) <= ROOM03_BLOCK_PUSHED_Y


def room03_stairs_step(snap: ZeldaSnapshot) -> FrameAction:
    """Controller-only 0x03: push west block UP, walk the slot, stand (128,141).

    CheckWarps fires on the exact stand; ALIGN_TOL=3 is too loose (139 misses).
    """
    if snap.level != LEVEL9:
        return FrameAction(nes_idle_action(), "wait_level9")
    if snap.mode == 17:
        return FrameAction(nes_idle_action(), "link_death")
    if stair_transition_modes(snap.mode) or snap.transitioning:
        return FrameAction(nes_action("UP"), "stairs_transition")
    if snap.mode != PLAY_MODE:
        return FrameAction(nes_idle_action(), f"wait_mode_{snap.mode}")
    if snap.screen != 0x03:
        return FrameAction(nes_idle_action(), f"left_source_0x{snap.screen:02x}")
    if (
        int(snap.link_x) == ROOM03_STAIR_X
        and int(snap.link_y) == ROOM03_STAIR_Y
    ):
        return FrameAction(nes_idle_action(), "stand_on_03_stairs")
    if not room03_west_block_pushed(snap):
        east_open = bool(snap.cur_opened_doors & 0x01)
        se_x = 176 if east_open else 208
        if snap.link_y < 180 and snap.link_x > 150:
            # Open east hole: x-first to the east corridor. Closed: SE corner.
            return walk_to_step(snap, se_x, 189, y_first=not east_open)
        if snap.link_y < 180 and abs(int(snap.link_x) - ROOM03_PUSH_X) > 2:
            if int(snap.link_x) >= 110:
                return walk_to_step(snap, se_x, 189, y_first=False)
            if int(snap.link_y) < 136:
                # North of the plus at x≈103: DOWN hits diamond. West wall first.
                return walk_to_step(snap, 32, 189, y_first=False)
            return walk_to_step(snap, int(snap.link_x), 189, y_first=True)
        if abs(int(snap.link_x) - ROOM03_PUSH_X) > 2:
            return walk_to_step(snap, ROOM03_PUSH_X, 189, y_first=False)
        if int(snap.link_y) > 172:
            return walk_to_step(
                snap, ROOM03_PUSH_X, ROOM03_PUSH_Y, y_first=True, tol=0
            )
        # North of the unpushed block: leave the column. Do not DOWN
        # through it (that slides the 0x68 south and kills the secret).
        if int(snap.link_y) < 144:
            return walk_to_step(snap, 32, int(snap.link_y), y_first=False)
        if abs(int(snap.link_x) - ROOM03_PUSH_X) > 1:
            return walk_to_step(
                snap, ROOM03_PUSH_X, ROOM03_PUSH_Y, y_first=False, tol=0
            )
        return FrameAction(nes_action("UP"), "push_03_west_block_up")
    if int(snap.link_x) < 110:
        if int(snap.link_y) > 136 or int(snap.link_y) < 130:
            return walk_to_step(snap, ROOM03_PUSH_X, ROOM03_SLOT_Y, y_first=True)
        return walk_to_step(
            snap, ROOM03_STAIR_X, ROOM03_STAIR_Y, y_first=False, tol=0
        )
    return walk_to_step(
        snap, ROOM03_STAIR_X, ROOM03_STAIR_Y, y_first=False, tol=0
    )


def chase_sword_step(
    snap: ZeldaSnapshot,
    cooldown: int,
    types: tuple[int, ...] | None = None,
) -> tuple[FrameAction, int]:
    enemies = live_combat_objects(snap)
    if types is not None:
        enemies = tuple(obj for obj in enemies if obj.type_id in types)
    if not enemies:
        return FrameAction(nes_idle_action(), "chase_clear"), max(0, cooldown - 1)
    if cooldown > 0:
        return FrameAction(nes_idle_action(), "chase_cooldown"), cooldown - 1
    target = min(
        enemies,
        key=lambda obj: abs(int(obj.x) - snap.link_x) + abs(int(obj.y) - snap.link_y),
    )
    dx = int(target.x) - int(snap.link_x)
    dy = int(target.y) - int(snap.link_y)
    # Overlap / Like-Like grab: sword hitbox is in front of Link, so walk
    # would LEFT-idle forever at dx=dy=0. Slash in place.
    if abs(dx) <= 8 and abs(dy) <= 8:
        return FrameAction(nes_action("UP", "A"), "chase_overlap_slash"), 10
    for direction in ("UP", "DOWN", "LEFT", "RIGHT"):
        if in_sword_hitbox(
            snap.link_x,
            snap.link_y,
            direction,
            target.x,
            target.y,
            reach=24,
            half_width=16,
        ):
            return FrameAction(nes_action(direction, "A"), "chase_slash"), 10
    if abs(dx) >= abs(dy):
        return FrameAction(nes_action("RIGHT" if dx > 0 else "LEFT"), "chase_x"), 0
    return FrameAction(nes_action("DOWN" if dy > 0 else "UP"), "chase_y"), 0




def room03_invuln_on_push_column(snap: ZeldaSnapshot) -> bool:
    """True when invuln 0x2B is on the west column in the push approach."""
    for obj in live_combat_objects(snap):
        if obj.type_id != 0x2B:
            continue
        if abs(int(obj.x) - ROOM03_PUSH_X) <= 16 and 150 <= int(obj.y) <= 189:
            return True
    return False


def room03_like_like_blocks_push(snap: ZeldaSnapshot) -> bool:
    """True when a Like-Like grabs Link or sits on the 0x03 west push column."""
    for obj in live_combat_objects(snap):
        if obj.type_id != 0x17:
            continue
        grabbed = (
            abs(int(obj.x) - int(snap.link_x)) <= 8
            and abs(int(obj.y) - int(snap.link_y)) <= 8
        )
        on_west = 64 <= int(obj.x) <= 120 and 128 <= int(obj.y) <= 189
        if grabbed or on_west:
            return True
    return False

def cellar_exit_step(snap: ZeldaSnapshot, *, side: str = "left") -> FrameAction:
    """Walk to a mode-9 cellar mouth and hold UP.

    Disassembly ``CheckSubroom``: Y < 0x40 and UP; X < 0x80 reads AttrsA,
    else AttrsB.
    """
    if snap.mode == 17:
        return FrameAction(nes_idle_action(), "link_death")
    if snap.transitioning or snap.mode == STAIRS_ENTER_MODE:
        return FrameAction(nes_action("UP"), "cellar_enter_scroll")
    if snap.mode == STAIRS_EXIT_MODE:
        return FrameAction(nes_action("UP"), "cellar_exit_scroll")
    if snap.mode == ITEM_CELLAR_MODE:
        return FrameAction(nes_action("DOWN"), "item_cellar_leave")
    if snap.mode != CELLAR_MODE and snap.mode != PLAY_MODE:
        return FrameAction(nes_idle_action(), f"cellar_wait_mode_{snap.mode}")
    # Live 0x77: stay on x=$30 while climbing. $50 is only the mouth-top stand
    # (switching at y=$70 walked off the ladder into brick at y≈93).
    if side == "left":
        if int(snap.link_y) < 0x40 and abs(int(snap.link_x) - CELLAR_LEFT_X) <= 8:
            target_x = CELLAR_LEFT_X
        else:
            target_x = 0x30
    else:
        target_x = 0xC0 if int(snap.link_y) > 0x70 else CELLAR_RIGHT_X
    # Opposite-mouth spawn sits in a stairwell; LEFT/RIGHT there hits brick.
    # Drop to the pit, cross, then climb the requested mouth.
    wrong_side = (
        (side == "left" and int(snap.link_x) >= CELLAR_SPLIT_X)
        or (side == "right" and int(snap.link_x) < CELLAR_SPLIT_X)
    )
    if wrong_side and int(snap.link_y) < CELLAR_CORRIDOR_Y:
        return FrameAction(nes_action("DOWN"), f"cellar_to_corridor_{side}")
    if abs(int(snap.link_x) - target_x) > ALIGN_TOL:
        direction = "RIGHT" if snap.link_x < target_x else "LEFT"
        return FrameAction(nes_action(direction), f"cellar_align_x_{side}")
    if int(snap.link_y) > CELLAR_EXIT_Y:
        return FrameAction(nes_action("UP"), f"cellar_walk_up_{side}")
    return FrameAction(nes_action("UP"), f"cellar_push_up_{side}")


def take_stairs_step(
    snap: ZeldaSnapshot,
    *,
    source: int,
    target: tuple[int, int] | None,
    push: bool = False,
) -> FrameAction:
    """One frame toward a stair stand, optional left-block push, or tile stand."""
    if snap.level != LEVEL9:
        return FrameAction(nes_idle_action(), "wait_level9")
    if snap.mode == 17:
        return FrameAction(nes_idle_action(), "link_death")
    if stair_transition_modes(snap.mode) or snap.transitioning:
        return FrameAction(nes_action("UP"), "stairs_transition")
    if snap.mode != PLAY_MODE:
        return FrameAction(nes_idle_action(), f"wait_mode_{snap.mode}")
    if snap.screen != int(source):
        return FrameAction(nes_idle_action(), f"left_source_0x{snap.screen:02x}")
    if int(source) == 0x03:
        return room03_stairs_step(snap)
    if int(source) == ROOM30:
        return room30_stairs_step(snap)
    if on_warp_tile(snap):
        return FrameAction(nes_idle_action(), "on_stair_tile")
    if target is None:
        return FrameAction(nes_idle_action(), "no_stair_target")
    frame = walk_to_step(snap, target[0], target[1], y_first=True)
    if frame.reason == "walk_arrived" and push:
        return FrameAction(nes_action("LEFT"), "push_block_left")
    if frame.reason == "walk_arrived":
        return FrameAction(nes_action("UP"), "stand_on_stairs")
    return frame


__all__ = [
    "ALIGN_TOL",
    "BLACK_MOUTH_TILE",
    "BLOCK_PUSH_STANDS",
    "BLOCK_STAIRS_X",
    "BLOCK_STAIRS_Y",
    "CELLAR_CORRIDOR_Y",
    "CELLAR_EXIT_Y",
    "CELLAR_LEFT_X",
    "CELLAR_MODE",
    "CELLAR_RIGHT_X",
    "CELLAR_SPLIT_X",
    "PUSHABLE_BLOCK",
    "ROOM03",
    "ROOM03_BLOCK_PUSHED_Y",
    "ROOM03_ROM_EAST",
    "ROOM03_ROM_NORTH",
    "ROOM03_ROM_SOUTH",
    "ROOM03_ROM_WEST",
    "ROOM03_PUSH_X",
    "ROOM03_PUSH_Y",
    "ROOM03_STAIR_X",
    "ROOM03_STAIR_Y",
    "ROOM04",
    "ROOM04_BOMB_WEST_APPROACH",
    "ROOM04_BOMB_WEST_STAND",
    "ROOM04_ROM_EAST",
    "ROOM04_ROM_NORTH",
    "ROOM04_ROM_SOUTH",
    "ROOM04_ROM_WEST",
    "ROOM30",
    "ROOM30_ROM_EAST",
    "ROOM30_ROM_NORTH",
    "ROOM30_ROM_SECRET",
    "ROOM30_ROM_SOUTH",
    "ROOM30_ROM_WEST",
    "ROOM30_STAIR_X",
    "ROOM30_STAIR_Y",
    "ROOM30_PUSH_X",
    "ROOM30_PUSH_Y",
    "ROOM30_BLOCK_REST",
    "CELLAR_67",
    "ROOM40",
    "ROOM40_ROM_NORTH",
    "ROOM40_ROM_SOUTH",
    "ROOM40_ROM_WEST",
    "ROOM40_ROM_EAST",
    "ROOM40_ROM_SECRET",
    "ROOM20",
    "ROOM20_ROM_NORTH",
    "ROOM20_ROM_SOUTH",
    "ROOM20_ROM_WEST",
    "ROOM20_ROM_EAST",
    "ROOM2F",
    "ROOM2F_ROM_NORTH",
    "ROOM2F_ROM_SOUTH",
    "ROOM2F_ROM_WEST",
    "ROOM2F_ROM_EAST",
    "ROOM31",
    "ROOM31_ROM_NORTH",
    "ROOM31_ROM_SOUTH",
    "ROOM31_ROM_WEST",
    "ROOM31_ROM_EAST",
    "ROOM41",
    "ROOM41_ROM_NORTH",
    "ROOM41_ROM_SOUTH",
    "ROOM41_ROM_WEST",
    "ROOM41_ROM_EAST",
    "ROOM41_ROM_SECRET",
    "ROOM41_SOUTH_Y",
    "ROOM51",
    "ROOM21",
    "ROOM21_ROM_NORTH",
    "ROOM21_ROM_SOUTH",
    "ROOM21_ROM_WEST",
    "ROOM21_ROM_EAST",
    "ROOM21_ROM_SECRET",
    "ROOM21_SOUTH_Y",
    "ROOM21_WEST_X",
    "ROOM11",
    "ROOM02",
    "ROOM02_ROM_EAST",
    "ROOM02_ROM_NORTH",
    "ROOM02_ROM_SOUTH",
    "ROOM02_ROM_WEST",
    "ROM_DOOR_NAMES",
    "BOMB_WALL_04_WEST",
    "ROOM13",
    "ROOM13_ROM_EAST",
    "ROOM13_ROM_NORTH",
    "ROOM13_ROM_SOUTH",
    "ROOM13_ROM_WEST",
    "ITEM_CELLAR_MODE",
    "PLAY_STAIR_CANDIDATES",
    "LEVEL9_CELLAR_DEST_LEFT",
    "LEVEL9_CELLAR_DEST_RIGHT",
    "LEVEL9_CELLAR_ROOMS",
    "LEVEL9_STAIR_LIST_DC",
    "LEVEL9_STAIR_LIST_INES",
    "LEVEL9_STAIR_PAIRS",
    "LEVEL9_STAIR_SOURCES",
    "PATRA_STAIR_SOURCE",
    "cellar_dest_for",
    "cellar_for_play_room",
    "cellar_mouth_xy",
    "in_patra_cellar",
    "in_room_04",
    "in_room_13",
    "in_room_30",
    "in_room_40",
    "in_room_31",
    "in_room_21",
    "in_room_41",
    "room21_is_rom_predecessor_of_31",
    "room21_loader_avoids_31",
    "room21_rom_south_is_shutter",
    "room21_to_31_step",
    "room41_is_rom_predecessor_of_31",
    "room41_loader_avoids_31",
    "room41_rom_north_is_open",
    "room41_to_31_step",
    "in_cellar_67",
    "room30_block_secret_open",
    "room30_loader_avoids_04",
    "room30_rom_neighbors",
    "room30_rom_secret_is_block_stairs",
    "room30_stairs_step",
    "room40_is_rom_predecessor_of_30",
    "room40_loader_avoids_30",
    "room40_rom_north_is_key",
    "room40_to_30_step",
    "ROOM40_SOUTH_Y",
    "is_patra_cellar_source",
    "make_room04_bomb_west_controller",
    "make_room31_bomb_west_controller",
    "on_warp_tile",
    "pause_select_next_b_item_script",
    "rom_door_name",
    "room03_invuln_on_push_column",
    "room03_like_like_blocks_push",
    "room03_rom_neighbors",
    "room04_bomb_west_approach_step",
    "room04_is_rom_predecessor_of_03",
    "room04_rom_west_is_bomb",
    "room31_bomb_west_approach_step",
    "room31_is_rom_predecessor_of_30",
    "room31_loader_avoids_30",
    "room31_rom_west_is_bomb",
    "room13_is_clean_predecessor_of_03",
    "room13_to_03_step",
    "play_rooms_entering_cellar",
    "STAIRS_ENTER_MODE",
    "STAIRS_EXIT_MODE",
    "STAIR_STANDS",
    "STAIR_TILE_HI",
    "STAIR_TILE_LO",
    "StairLoader",
    "cellar_exit_step",
    "chase_sword_step",
    "dest_report",
    "in_stair_source",
    "in_underworld_passage",
    "landed_final_patra",
    "live_combat_objects",
    "on_stair_tile",
    "paired_stair_dest",
    "pushable_block",
    "room03_stairs_step",
    "room03_west_block_pushed",
    "stair_loader_for",
    "stair_object_summary",
    "stair_room_objects",
    "stair_transition_modes",
    "take_stairs_step",
    "walk_to_step",
]
