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
from zelda_i.dungeon.bomb_wall import BOMB_N_WAIT_BLAST, BombWallController
from zelda_i.combat import in_sword_hitbox
from zelda_i.dungeon.ids import object_name
from zelda_i.level2.puzzles import BombWall, DOOR_RIGHT
from zelda_i.level9.ganon import LEVEL9
from zelda_i.level9.patra import PATRA_EYE_COUNT, final_patra_live, patra_eyes
from zelda_i.level9.path import NORTH_DOOR_X, NORTH_DOOR_X_TOL
from zelda_i.level9.room62 import door_bits
from zelda_i.ram import PLAY_MODE, ZeldaObject, ZeldaSnapshot

LEVEL9_STAIR_LIST_INES = 0x19C20
LEVEL9_STAIR_LIST_DC = 0x19C10
LEVEL9_STAIR_PAIRS: tuple[tuple[int, int], ...] = (
    (0x60, 0x70),
    (0x72, 0x75),
    (0x67, 0x77),
    (0x00, 0x4F),
)
LEVEL9_CELLAR_ROOMS: tuple[int, ...] = (0x60, 0x70, 0x72, 0x75, 0x67, 0x77)
LEVEL9_CELLAR_DEST_LEFT: dict[int, int] = {
    0x60: 0x14, 0x70: 0x63, 0x72: 0x71, 0x75: 0x20, 0x67: 0x30, 0x77: 0x52,
}
LEVEL9_CELLAR_DEST_RIGHT: dict[int, int] = {
    0x60: 0x55, 0x70: 0x05, 0x72: 0x74, 0x75: 0x61, 0x67: 0x04, 0x77: 0x03,
}
PATRA_STAIR_SOURCE = 0x77
BLOCK_STAIRS_X = 0xD0
BLOCK_STAIRS_Y = 0x60
STAIR_TILE_LO = 0x70
STAIR_TILE_HI = 0x73
BLACK_MOUTH_TILE = 0x24
PLAY_STAIR_CANDIDATES: tuple[int, ...] = (0x03, 0x04, 0x13, 0x07, 0x30, 0x52, 0x76, 0x67)
CELLAR_MODE = 9
STAIRS_EXIT_MODE = 10
ITEM_CELLAR_MODE = 11
STAIRS_ENTER_MODE = 16
CELLAR_EXIT_Y = 0x3D
CELLAR_LEFT_X = 0x50
CELLAR_RIGHT_X = 0xB0
CELLAR_SPLIT_X = 0x80
CELLAR_CORRIDOR_Y = 0xB4
PUSHABLE_BLOCK = 0x68
ROOM03_STAIR_X = 0x80
ROOM03_STAIR_Y = 0x8D
ROOM03_PUSH_X = 0x60
ROOM03_PUSH_Y = 170
ROOM03_SLOT_Y = 133
ROOM03_BLOCK_REST_Y = 0x90
ROOM03_BLOCK_PUSHED_Y = 0x80
STAIR_STANDS: tuple[tuple[int, int], ...] = (
    (0x78, 0x90), (0x80, 0x90), (0x70, 0x90), (0x78, 0xA0), (0x78, 0x80),
    (0x88, 0x90), (0x68, 0x90), (BLOCK_STAIRS_X, BLOCK_STAIRS_Y), (0x60, 0x90),
    (0x90, 0x90), (0x78, 0x8D), (0x50, 0x80), (0xA0, 0x80), (0x40, 0xA0),
    (0xB0, 0xA0), (0x78, 0x70), (0x78, 0xB0),
)
BLOCK_PUSH_STANDS: tuple[tuple[int, int], ...] = (
    (0x88, 0x90), (0x80, 0x90), (0x78, 0x90), (0x90, 0x90),
    (0x70, 0x90), (0x88, 0x80), (0x88, 0xA0),
)
_PLAY_SLOTS = range(1, 13)
ALIGN_TOL = 3
SOUTH_DOOR_Y = 189
ROOM21_WEST_X = 48
BOMB_WEST_STAND = (48, 141)
BOMB_WEST_APPROACH = (48, 189)

# Enter through an open or false-wall door so Link is not trapped in a
# key-door alcove against a center block column (live 0x60 south mouth).
_PREFERRED_LOADERS: dict[int, tuple[int, str, int, int]] = {
    0x60: (0x50, "DOWN", 0x78, 0xDD),
    0x70: (0x60, "DOWN", 0x78, 0xDD),
    0x72: (0x62, "DOWN", 0x78, 0xDD),
    0x75: (0x65, "DOWN", 0x78, 0xDD),
    0x67: (0x77, "UP", 0x78, 0x58),
    0x77: (0x67, "DOWN", 0x78, 0xDD),
    0x00: (0x10, "UP", 0x78, 0x58),
    0x4F: (0x5F, "UP", 0x78, 0x58),
    0x03: (0x13, "UP", 0x78, 0x58),
    0x04: (0x14, "UP", 0x78, 0x58),
    0x30: (0x40, "UP", 0x78, 0x58),
    0x40: (0x50, "UP", 0x78, 0x58),
    0x20: (0x10, "UP", 0x78, 0x58),
    0x31: (0x41, "UP", 0x78, 0x58),
    0x21: (0x11, "DOWN", 0x78, 0xDD),
    0x41: (0x51, "UP", 0x78, 0x58),
    0x51: (0x61, "UP", 0x78, 0x58),
    0x13: (0x23, "UP", 0x78, 0x58),
    0x07: (0x17, "UP", 0x78, 0x58),
    0x52: (0x62, "UP", 0x78, 0x58),
    0x76: (0x66, "DOWN", 0x78, 0xDD),
}

# L7-9 door codes: 0 open, 1 wall, 2/3 false, 4 bomb, 5/6 key, 7 shutter.
# Values are (north, south, west, east, secret).
ROM_DOORS: dict[int, tuple[int, int, int, int, int]] = {
    0x13: (1, 5, 5, 1, 0),
    0x03: (1, 1, 1, 4, 0),
    0x04: (1, 1, 4, 1, 0),
    0x02: (1, 0, 5, 1, 0),
    0x30: (1, 5, 1, 4, 5),
    0x40: (5, 5, 1, 1, 7),
    0x20: (4, 1, 1, 1, 0),
    0x2F: (2, 3, 1, 7, 0),
    0x31: (0, 7, 4, 1, 0),
    0x41: (0, 7, 1, 1, 0),
    0x51: (0, 0, 7, 1, 1),
    0x61: (0, 0, 1, 0, 0),
    0x50: (5, 1, 1, 7, 0),
    0x21: (0, 7, 1, 4, 0),
}
ROM_DOOR_NAMES: dict[int, str] = {
    0: "open", 1: "wall", 2: "false", 3: "false2",
    4: "bomb", 5: "key", 6: "key2", 7: "shutter",
}
ROM_SECRET_NAMES: dict[int, str] = {
    0: "none", 1: "all_dead", 5: "block_stairs", 7: "foes_item",
}
_OPP = {"n": "s", "s": "n", "w": "e", "e": "w"}
_NSWE = "nswe"

ROOM13, ROOM03, ROOM04, ROOM02 = 0x13, 0x03, 0x04, 0x02
ROOM30, ROOM40, ROOM20, ROOM2F = 0x30, 0x40, 0x20, 0x2F
ROOM31, ROOM41, ROOM51, ROOM61, ROOM50 = 0x31, 0x41, 0x51, 0x61, 0x50
ROOM21, ROOM11 = 0x21, 0x11
CELLAR_67 = 0x67
ROOM30_STAIR_X, ROOM30_STAIR_Y = BLOCK_STAIRS_X, BLOCK_STAIRS_Y
ROOM30_PUSH_X, ROOM30_PUSH_Y = 0x60, 170
ROOM30_BLOCK_REST = (0x60, 0x90)
ROOM04_BOMB_WEST_STAND = BOMB_WEST_STAND
ROOM04_BOMB_WEST_APPROACH = BOMB_WEST_APPROACH
ROOM31_BOMB_WEST_STAND = BOMB_WEST_STAND
ROOM31_BOMB_WEST_APPROACH = BOMB_WEST_APPROACH
ROOM21_SOUTH_Y = SOUTH_DOOR_Y
ROOM41_SOUTH_Y = SOUTH_DOOR_Y
ROOM40_SOUTH_Y = SOUTH_DOOR_Y


def _bind_rom_aliases() -> None:
    g = globals()
    for room, (north, south, west, east, secret) in ROM_DOORS.items():
        prefix = f"ROOM{room:02X}_ROM"
        g[f"{prefix}_NORTH"] = north
        g[f"{prefix}_SOUTH"] = south
        g[f"{prefix}_WEST"] = west
        g[f"{prefix}_EAST"] = east
        g[f"{prefix}_SECRET"] = secret


_bind_rom_aliases()

BOMB_WALL_04_WEST = BombWall(
    room=ROOM04,
    stand=BOMB_WEST_STAND,
    face="LEFT",
    opens_to=ROOM03,
    opened_door_bit=DOOR_RIGHT,
    live=True,
    notes="LIVE: west bomb (48,141) LEFT opens 0x03 east hole (rr-sz8.3).",
)
BOMB_WALL_31_WEST = BombWall(
    room=ROOM31,
    stand=BOMB_WEST_STAND,
    face="LEFT",
    opens_to=ROOM30,
    opened_door_bit=DOOR_RIGHT,
    live=True,
    notes="LIVE: west bomb (48,141) LEFT opens 0x30 east hole (rr-sz8.3).",
)


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
    value = int(room) & 0xFF
    for left, right in LEVEL9_STAIR_PAIRS:
        if value == left:
            return right
        if value == right:
            return left
    return None


def cellar_dest_for(room: int, *, side: str = "left") -> int | None:
    table = LEVEL9_CELLAR_DEST_LEFT if side == "left" else LEVEL9_CELLAR_DEST_RIGHT
    return table.get(int(room) & 0xFF)


def is_patra_cellar_source(room: int) -> bool:
    return (int(room) & 0xFF) == PATRA_STAIR_SOURCE


def play_rooms_entering_cellar(cellar: int) -> tuple[tuple[int, str], ...]:
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
    value = int(play) & 0xFF
    for cellar in LEVEL9_CELLAR_ROOMS:
        if LEVEL9_CELLAR_DEST_LEFT.get(cellar) == value:
            return (cellar, "left")
        if LEVEL9_CELLAR_DEST_RIGHT.get(cellar) == value:
            return (cellar, "right")
    return None


def cellar_mouth_xy(*, side: str = "left") -> tuple[int, int]:
    x = CELLAR_LEFT_X if side == "left" else CELLAR_RIGHT_X
    return (x, CELLAR_EXIT_Y)


def stair_loader_for(room: int) -> StairLoader:
    value = int(room) & 0xFF
    if value in _PREFERRED_LOADERS:
        from_room, direction, link_x, link_y = _PREFERRED_LOADERS[value]
        return StairLoader(
            room=value, from_room=from_room, direction=direction,
            link_x=link_x, link_y=link_y,
            label=f"0x{from_room:02x}_hold_{direction.lower()}",
        )
    row = (value >> 4) & 0x0F
    if row < 7:
        from_room = value + 0x10
        return StairLoader(
            room=value, from_room=from_room, direction="UP",
            link_x=0x78, link_y=0x58, label=f"south_0x{from_room:02x}_hold_up",
        )
    from_room = value - 0x10
    return StairLoader(
        room=value, from_room=from_room, direction="DOWN",
        link_x=0x78, link_y=0xDD, label=f"north_0x{from_room:02x}_hold_down",
    )


def in_stair_source(snap: ZeldaSnapshot, room: int) -> bool:
    return snap.mode == PLAY_MODE and snap.level == LEVEL9 and snap.screen == int(room)


def rom_door_name(code: int) -> str:
    return ROM_DOOR_NAMES.get(int(code) & 7, f"code_{int(code) & 7}")


def rom_side(room: int, side: str) -> int:
    return ROM_DOORS[int(room)][_NSWE.index(side[0])]


def rom_secret(room: int) -> int:
    return ROM_DOORS[int(room)][4]


def rom_pair(a: int, b: int, side: str) -> tuple[int, int]:
    """Door codes on ``a.side`` and the facing side of ``b``."""
    return rom_side(a, side), rom_side(b, _OPP[side[0]])


def loader_avoids(room: int, other: int) -> bool:
    return stair_loader_for(int(room)).from_room != int(other)


def rom_door_row(
    room: int, *, direction: str = "self", note: str | None = None
) -> dict[str, object]:
    north, south, west, east, secret = ROM_DOORS[int(room)]
    row: dict[str, object] = {
        "room": int(room), "dir": direction,
        "n": north, "s": south, "w": west, "e": east,
        "n_name": rom_door_name(north), "s_name": rom_door_name(south),
        "w_name": rom_door_name(west), "e_name": rom_door_name(east),
        "secret": ROM_SECRET_NAMES.get(secret, "none"),
    }
    if note:
        row["note"] = note
    return row


def rom_doors_report(*rooms: int) -> dict[str, dict[str, object]]:
    out: dict[str, dict[str, object]] = {}
    for room in rooms:
        north, south, west, east, secret = ROM_DOORS[int(room)]
        out[f"0x{int(room):02X}"] = {
            "north": north, "south": south, "west": west, "east": east,
            "north_name": rom_door_name(north), "south_name": rom_door_name(south),
            "west_name": rom_door_name(west), "east_name": rom_door_name(east),
            "secret": secret, "secret_name": ROM_SECRET_NAMES.get(secret, "none"),
        }
    return out


def room03_rom_neighbors() -> tuple[dict[str, object], ...]:
    return (
        rom_door_row(ROOM03),
        rom_door_row(ROOM13, direction="south", note="disproved: 0x13 north wall / 0x03 south wall"),
        rom_door_row(ROOM02, direction="west", note="0x02 east wall / 0x03 west wall"),
        rom_door_row(ROOM04, direction="east", note="candidate: 0x04 west bomb / 0x03 east bomb"),
    )


def room30_rom_neighbors() -> tuple[dict[str, object], ...]:
    return (
        rom_door_row(ROOM30),
        rom_door_row(ROOM20, direction="north", note="disproved ROM: 0x20 south wall / 0x30 north wall"),
        rom_door_row(ROOM40, direction="south", note="candidate: 0x40 north key / 0x30 south key"),
        rom_door_row(ROOM2F, direction="west", note="0x2F east shutter / 0x30 west wall"),
        rom_door_row(ROOM31, direction="east", note="LIVE dest: 0x31 west bomb → 0x30 (rr-sz8.3)"),
        {
            "room": CELLAR_67, "dir": "cellar_successor", "secret": "none",
            "note": "cellar 0x67 left→0x30 right→0x04; successor path, not pred",
        },
    )


def stair_room_objects(snap: ZeldaSnapshot) -> tuple[ZeldaObject, ...]:
    return tuple(obj for obj in snap.objects if obj.slot in _PLAY_SLOTS and (obj.type_id or obj.hp))


def stair_object_summary(snap: ZeldaSnapshot) -> list[dict[str, int | str]]:
    return [
        {
            "slot": obj.slot, "type_id": obj.type_id, "type_name": object_name(obj.type_id),
            "hp": obj.hp, "x": obj.x, "y": obj.y, "state": obj.state,
        }
        for obj in stair_room_objects(snap)
    ]


def live_combat_objects(snap: ZeldaSnapshot) -> tuple[ZeldaObject, ...]:
    return tuple(
        obj for obj in stair_room_objects(snap)
        if 0x01 <= obj.type_id < 0x40 and obj.type_id not in (0x37, 0x3F)
    )


def on_stair_tile(snap: ZeldaSnapshot) -> bool:
    return STAIR_TILE_LO <= int(snap.colliding_tile) <= STAIR_TILE_HI


def on_warp_tile(snap: ZeldaSnapshot) -> bool:
    return on_stair_tile(snap) or int(snap.colliding_tile) == BLACK_MOUTH_TILE


def stair_transition_modes(mode: int) -> bool:
    return int(mode) in (CELLAR_MODE, STAIRS_EXIT_MODE, ITEM_CELLAR_MODE, STAIRS_ENTER_MODE)


def in_cellar(snap: ZeldaSnapshot, room: int) -> bool:
    return snap.level == LEVEL9 and snap.screen == int(room) and stair_transition_modes(snap.mode)


def in_patra_cellar(snap: ZeldaSnapshot) -> bool:
    return in_cellar(snap, PATRA_STAIR_SOURCE)


def in_cellar_67(snap: ZeldaSnapshot) -> bool:
    return in_cellar(snap, CELLAR_67)


def in_underworld_passage(snap: ZeldaSnapshot) -> bool:
    return snap.mode in (CELLAR_MODE, STAIRS_EXIT_MODE, ITEM_CELLAR_MODE)


def landed_final_patra(snap: ZeldaSnapshot) -> bool:
    return (
        final_patra_live(snap)
        and len(patra_eyes(snap)) == PATRA_EYE_COUNT
        and not bool(snap.cur_opened_doors & 0x08)
        and not bool(snap.open_doorway_mask & 0x08)
    )


def dest_report(snap: ZeldaSnapshot) -> dict[str, object]:
    return {
        "screen": int(snap.screen), "next_screen": int(snap.next_screen),
        "mode": int(snap.mode), "link": {"x": snap.link_x, "y": snap.link_y},
        "doors": door_bits(snap.cur_opened_doors), "mask": door_bits(snap.open_doorway_mask),
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
    snap: ZeldaSnapshot, x: int, y: int, *, y_first: bool = True, tol: int | None = None,
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
    for obj in stair_room_objects(snap):
        if obj.type_id == PUSHABLE_BLOCK:
            return obj
    return None


def door_column_step(
    snap: ZeldaSnapshot,
    *,
    source: int,
    dest: int,
    direction: str,
    align_x: int = NORTH_DOOR_X,
    south_y: int | None = SOUTH_DOOR_Y,
    west_band: int | None = None,
    align_tol: int = NORTH_DOOR_X_TOL,
) -> FrameAction:
    """One frame through a cardinal door column (north key/open or south shutter)."""
    if snap.level != LEVEL9:
        return FrameAction(nes_idle_action(), "wait_level9")
    if snap.transitioning or snap.mode in (4, 6, 7):
        return FrameAction(nes_action(direction), f"room{dest:02x}_scroll")
    if snap.mode != PLAY_MODE:
        return FrameAction(nes_idle_action(), f"wait_mode_{snap.mode}")
    if snap.screen == dest:
        return FrameAction(nes_idle_action(), f"room{dest:02x}_arrived")
    if snap.screen != source:
        return FrameAction(nes_idle_action(), f"unexpected_room_0x{snap.screen:02x}")
    x, y = int(snap.link_x), int(snap.link_y)
    if west_band is not None and south_y is not None and y < south_y - 6:
        if abs(x - align_x) <= align_tol:
            return FrameAction(nes_action("LEFT"), f"room{source:02x}_off_plus")
        if abs(x - west_band) > ALIGN_TOL:
            return walk_to_step(snap, west_band, y, y_first=False)
        return walk_to_step(snap, west_band, south_y, y_first=True)
    if south_y is not None and west_band is None and abs(x - align_x) > align_tol and y < south_y - 6:
        return walk_to_step(snap, x, south_y, y_first=True)
    if abs(x - align_x) > align_tol:
        return FrameAction(
            nes_action("LEFT" if x > align_x else "RIGHT"), f"room{source:02x}_align_x"
        )
    return FrameAction(nes_action(direction), f"room{source:02x}_push_{direction.lower()}")


def make_bomb_west_controller(
    wall: BombWall, *, stand_timeout: int | None = None,
) -> BombWallController:
    return BombWallController(
        wall=wall,
        level=LEVEL9,
        clear_spec=None,
        face_frames=6,
        step_back=0,
        wait_blast=BOMB_N_WAIT_BLAST,
        require_bomb_consumed=False,
        wait_hold_face=True,
        max_frames=8000,
        stand_timeout=2500 if stand_timeout is None else stand_timeout,
    )


def bomb_west_approach_step(
    snap: ZeldaSnapshot,
    *,
    approach: tuple[int, int] = BOMB_WEST_APPROACH,
    stand: tuple[int, int] = BOMB_WEST_STAND,
) -> FrameAction:
    ax, ay = approach
    sx, sy = stand
    if abs(int(snap.link_x) - ax) > ALIGN_TOL or int(snap.link_y) < ay - 6:
        return walk_to_step(snap, ax, ay, y_first=True)
    return walk_to_step(snap, sx, sy, y_first=True)


def pause_select_next_b_item_script() -> tuple[FrameAction, ...]:
    frames: list[FrameAction] = [FrameAction(nes_action("START"), "pause_open")]
    frames.extend(FrameAction(nes_idle_action(), "pause_settle") for _ in range(24))
    frames.append(FrameAction(nes_action("RIGHT"), "pause_next_item"))
    frames.extend(FrameAction(nes_idle_action(), "pause_cursor") for _ in range(8))
    frames.append(FrameAction(nes_action("START"), "pause_close"))
    frames.extend(FrameAction(nes_idle_action(), "pause_resume") for _ in range(24))
    return tuple(frames)


def room30_block_secret_open(snap: ZeldaSnapshot) -> bool:
    block = pushable_block(snap)
    return block is not None and int(block.x) >= 0xC0 and int(block.y) <= 0x70


def room30_stairs_step(snap: ZeldaSnapshot) -> FrameAction:
    """Push west 0x68 UP, then exact stand (208, 96). CheckWarps misses ALIGN_TOL=3."""
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
            return walk_to_step(snap, ROOM30_PUSH_X, ROOM30_PUSH_Y, y_first=True, tol=0)
        if abs(int(snap.link_x) - ROOM30_PUSH_X) > 1:
            return walk_to_step(snap, ROOM30_PUSH_X, ROOM30_PUSH_Y, y_first=False, tol=0)
        return FrameAction(nes_action("UP"), "push_30_west_block_up")
    if int(snap.link_y) < 180 and int(snap.link_x) < 200:
        return walk_to_step(snap, int(snap.link_x), 189, y_first=True)
    if abs(int(snap.link_x) - ROOM30_STAIR_X) > 0:
        return walk_to_step(snap, ROOM30_STAIR_X, 189, y_first=False, tol=0)
    return walk_to_step(snap, ROOM30_STAIR_X, ROOM30_STAIR_Y, y_first=True, tol=0)


def room03_west_block_pushed(snap: ZeldaSnapshot) -> bool:
    block = pushable_block(snap)
    return block is not None and int(block.y) <= ROOM03_BLOCK_PUSHED_Y


def room03_stairs_step(snap: ZeldaSnapshot) -> FrameAction:
    """Push west block UP, walk the slot, stand (128,141). CheckWarps is exact."""
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
    if int(snap.link_x) == ROOM03_STAIR_X and int(snap.link_y) == ROOM03_STAIR_Y:
        return FrameAction(nes_idle_action(), "stand_on_03_stairs")
    if not room03_west_block_pushed(snap):
        east_open = bool(snap.cur_opened_doors & 0x01)
        se_x = 176 if east_open else 208
        if snap.link_y < 180 and snap.link_x > 150:
            return walk_to_step(snap, se_x, 189, y_first=not east_open)
        if snap.link_y < 180 and abs(int(snap.link_x) - ROOM03_PUSH_X) > 2:
            if int(snap.link_x) >= 110:
                return walk_to_step(snap, se_x, 189, y_first=False)
            if int(snap.link_y) < 136:
                return walk_to_step(snap, 32, 189, y_first=False)
            return walk_to_step(snap, int(snap.link_x), 189, y_first=True)
        if abs(int(snap.link_x) - ROOM03_PUSH_X) > 2:
            return walk_to_step(snap, ROOM03_PUSH_X, 189, y_first=False)
        if int(snap.link_y) > 172:
            return walk_to_step(snap, ROOM03_PUSH_X, ROOM03_PUSH_Y, y_first=True, tol=0)
        if int(snap.link_y) < 144:
            return walk_to_step(snap, 32, int(snap.link_y), y_first=False)
        if abs(int(snap.link_x) - ROOM03_PUSH_X) > 1:
            return walk_to_step(snap, ROOM03_PUSH_X, ROOM03_PUSH_Y, y_first=False, tol=0)
        return FrameAction(nes_action("UP"), "push_03_west_block_up")
    if int(snap.link_x) < 110:
        if int(snap.link_y) > 136 or int(snap.link_y) < 130:
            return walk_to_step(snap, ROOM03_PUSH_X, ROOM03_SLOT_Y, y_first=True)
        return walk_to_step(snap, ROOM03_STAIR_X, ROOM03_STAIR_Y, y_first=False, tol=0)
    return walk_to_step(snap, ROOM03_STAIR_X, ROOM03_STAIR_Y, y_first=False, tol=0)


def chase_sword_step(
    snap: ZeldaSnapshot, cooldown: int, types: tuple[int, ...] | None = None,
) -> tuple[FrameAction, int]:
    enemies = live_combat_objects(snap)
    if types is not None:
        enemies = tuple(obj for obj in enemies if obj.type_id in types)
    if not enemies:
        return FrameAction(nes_idle_action(), "chase_clear"), max(0, cooldown - 1)
    if cooldown > 0:
        return FrameAction(nes_idle_action(), "chase_cooldown"), cooldown - 1
    target = min(enemies, key=lambda obj: abs(int(obj.x) - snap.link_x) + abs(int(obj.y) - snap.link_y))
    dx = int(target.x) - int(snap.link_x)
    dy = int(target.y) - int(snap.link_y)
    if abs(dx) <= 8 and abs(dy) <= 8:
        return FrameAction(nes_action("UP", "A"), "chase_overlap_slash"), 10
    for direction in ("UP", "DOWN", "LEFT", "RIGHT"):
        if in_sword_hitbox(snap.link_x, snap.link_y, direction, target.x, target.y, reach=24, half_width=16):
            return FrameAction(nes_action(direction, "A"), "chase_slash"), 10
    if abs(dx) >= abs(dy):
        return FrameAction(nes_action("RIGHT" if dx > 0 else "LEFT"), "chase_x"), 0
    return FrameAction(nes_action("DOWN" if dy > 0 else "UP"), "chase_y"), 0


def room03_invuln_on_push_column(snap: ZeldaSnapshot) -> bool:
    for obj in live_combat_objects(snap):
        if obj.type_id != 0x2B:
            continue
        if abs(int(obj.x) - ROOM03_PUSH_X) <= 16 and 150 <= int(obj.y) <= 189:
            return True
    return False


def room03_like_like_blocks_push(snap: ZeldaSnapshot) -> bool:
    for obj in live_combat_objects(snap):
        if obj.type_id != 0x17:
            continue
        grabbed = abs(int(obj.x) - int(snap.link_x)) <= 8 and abs(int(obj.y) - int(snap.link_y)) <= 8
        on_west = 64 <= int(obj.x) <= 120 and 128 <= int(obj.y) <= 189
        if grabbed or on_west:
            return True
    return False


def cellar_exit_step(snap: ZeldaSnapshot, *, side: str = "left") -> FrameAction:
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
    if side == "left":
        target_x = CELLAR_LEFT_X if int(snap.link_y) < 0x40 and abs(int(snap.link_x) - CELLAR_LEFT_X) <= 8 else 0x30
    else:
        target_x = 0xC0 if int(snap.link_y) > 0x70 else CELLAR_RIGHT_X
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
    snap: ZeldaSnapshot, *, source: int, target: tuple[int, int] | None, push: bool = False,
) -> FrameAction:
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
