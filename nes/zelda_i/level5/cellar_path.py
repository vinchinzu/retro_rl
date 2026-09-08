"""Level 5 whistle-return cellar / east hops.

0x06 block stairs → cellar 0x07 → 0x64, then east 0x05 / 0x64 → 0x65.

Room specs and stop predicates remain in ``level5.dungeon``.
"""

from __future__ import annotations

from retro_harness.nes import nes_action, nes_idle_action

from zelda_i.level5.dungeon import (
    LEVEL_5,
    ROOM_L5_BLUE_64,
    ROOM_L5_CELLAR_07,
    ROOM_L5_PASSAGE_06,
    ROOM_L5_WEST_65,
)
from zelda_i.level5.path import _step, walk_axis
from zelda_i.level5.whistle_path import (
    CELLAR_MODES,
    L5_CELLAR_FLOOR_Y,
    L5_CELLAR_LEFT_X,
    L5_CELLAR_RIGHT_X,
    ROOM_06_BLOCK_X,
    _in_cellar,
)
from zelda_i.ram import PLAY_MODE


def _raw_axis(env, assist, total: list[int], axis: str, target: int, max_f: int = 700) -> bool:
    """Hold one axis toward target. walk_axis bails on cellar mode 9 (stall 40)."""
    return walk_axis(
        env,
        assist,
        total,
        axis,
        target,
        max_f=max_f,
        stall_limit=180,
        done=lambda snap: snap.mode == PLAY_MODE and snap.screen == ROOM_L5_BLUE_64,
    )


def cellar_to_64(env, assist, total: list[int]) -> dict:
    """From L5 cellar 0x07 (0x06-stairs spawn ~128,141), left mouth → 0x64.

    walk_axis bails on cellar mode 9. Proven live: raw-step to x=192 without
    climbing, drop to pit y=189, floor-cross x=48, climb y=61, UP.
    y=165/x=192/y=61/UP is the 0x06 mouth (do not use it from this spawn).
    """
    from zelda_i.dungeon.ops import idle, push_dir
    from zelda_i.ram import read_snapshot as _rs

    log = []
    snap = _rs(env.get_ram())
    start = {"xy": [snap.link_x, snap.link_y], "room": snap.screen, "mode": snap.mode}

    def rec(tag):
        s = _rs(env.get_ram())
        log.append({"tag": tag, "xy": [s.link_x, s.link_y], "mode": s.mode, "room": s.screen, "tile": int(s.colliding_tile)})
        return s

    def left_ok(s) -> bool:
        return s.mode == PLAY_MODE and s.screen == ROOM_L5_BLUE_64

    # Reach the right drop without climbing the 0x06 mouth (y<120).
    if snap.link_x >= 80 and snap.link_y < 170:
        for _ in range(480):
            s = _rs(env.get_ram())
            if left_ok(s):
                break
            if abs(s.link_x - L5_CELLAR_RIGHT_X) <= 2 and s.link_y >= 180:
                break
            if s.link_y < 120:
                _step(env, assist, total, nes_action("DOWN"))
            elif abs(s.link_x - L5_CELLAR_RIGHT_X) > 2:
                _step(env, assist, total, nes_action("RIGHT" if s.link_x < L5_CELLAR_RIGHT_X else "LEFT"))
            else:
                _step(env, assist, total, nes_action("DOWN"))
        rec("right_col")
    # Pit floor first. y=165 on the right ladder cannot cross left.
    _raw_axis(env, assist, total, "y", L5_CELLAR_FLOOR_Y, max_f=500)
    rec("pit")
    _raw_axis(env, assist, total, "x", L5_CELLAR_LEFT_X, max_f=700)
    rec("left_col")
    _raw_axis(env, assist, total, "y", 61, max_f=500)
    rec("left_climb")
    push_dir(env, assist, total, "UP", frames=260)
    idle(env, assist, total, 12)
    for _ in range(280):
        snap = _rs(env.get_ram())
        if left_ok(snap):
            break
        if snap.mode == PLAY_MODE and snap.screen not in (ROOM_L5_CELLAR_07,):
            break
        if abs(snap.link_x - L5_CELLAR_LEFT_X) > 4:
            _step(env, assist, total, nes_action("LEFT" if snap.link_x > L5_CELLAR_LEFT_X else "RIGHT"))
        else:
            _step(env, assist, total, nes_action("UP"))
    idle(env, assist, total, 16)
    rec("after_up")
    snap = _rs(env.get_ram())
    return {
        "path": "raw_x192_pit189_x48_y61_up",
        "start": start,
        "log": log,
        "dest": snap.screen,
        "mode": snap.mode,
        "xy": [snap.link_x, snap.link_y],
        "success": snap.level == LEVEL_5 and snap.screen == ROOM_L5_BLUE_64 and snap.mode == PLAY_MODE,
    }


def walk_east_from_05(env, assist, total: list[int]) -> dict:
    """Cleared 0x05 east (already unlocked) → 0x06."""
    from zelda_i.dungeon.ops import goto, idle, push_dir
    from zelda_i.ram import read_snapshot as _rs

    snap = _rs(env.get_ram())
    keys0 = int(snap.keys)
    walk_axis(env, assist, total, "y", 141, max_f=400)
    walk_axis(env, assist, total, "x", 224, max_f=500)
    goto(env, assist, total, 224, 141, tol=3, max_f=300)
    push_dir(env, assist, total, "RIGHT", frames=240)
    idle(env, assist, total, 16)
    snap = _rs(env.get_ram())
    if snap.screen != ROOM_L5_PASSAGE_06:
        for _ in range(240):
            snap = _rs(env.get_ram())
            if snap.mode == PLAY_MODE and snap.screen == ROOM_L5_PASSAGE_06:
                break
            _step(env, assist, total, nes_idle_action())
        idle(env, assist, total, 12)
    snap = _rs(env.get_ram())
    return {
        "path": "east_from_05",
        "keys_in": keys0,
        "keys_out": int(snap.keys),
        "dest": snap.screen,
        "xy": [snap.link_x, snap.link_y],
        "mode": snap.mode,
        "success": snap.level == LEVEL_5 and snap.screen == ROOM_L5_PASSAGE_06 and snap.mode == PLAY_MODE,
    }


def take_block_stairs_06(env, assist, total: list[int]) -> dict:
    """0x06: north-around diamond, push 0x68 UP, idle on (96,133) → cellar 0x07.

    Do not walk south (key door 0x16). Do not hunt diamond-center tiles —
    those 0x70–0x73 stands do not warp. Live: block (96,144)→(96,128),
    stairs stand (96,133) (outbound cellar spawn).
    """
    from zelda_i.dungeon.ops import idle, push_dir
    from zelda_i.ram import ADDR_WHISTLE, read_snapshot as _rs, read_u8

    snap = _rs(env.get_ram())
    start = {
        "xy": [snap.link_x, snap.link_y],
        "mode": snap.mode,
        "room": snap.screen,
        "keys": int(snap.keys),
        "whistle": int(read_u8(env.get_ram(), ADDR_WHISTLE)),
    }
    log = [dict(start, tag="start")]

    def done(s) -> bool:
        if s.mode == PLAY_MODE and s.screen == 0x16:
            return False
        if _in_cellar(s):
            return True
        return s.level == LEVEL_5 and s.screen == ROOM_L5_CELLAR_07

    def rec(tag: str) -> None:
        s = _rs(env.get_ram())
        blocks = [
            {"x": o.x, "y": o.y}
            for o in s.objects
            if 1 <= o.slot <= 12 and o.type_id == 0x68
        ]
        log.append(
            {
                "tag": tag,
                "xy": [s.link_x, s.link_y],
                "mode": s.mode,
                "room": s.screen,
                "tile": int(s.colliding_tile),
                "blocks": blocks,
                "keys": int(s.keys),
            }
        )

    if snap.link_x < 48:
        walk_axis(env, assist, total, "x", 48, max_f=240)
    walk_axis(env, assist, total, "y", 93, max_f=400)
    rec("north_wall")
    if not done(_rs(env.get_ram())):
        walk_axis(env, assist, total, "x", 64, max_f=300)
        rec("nw")
    if not done(_rs(env.get_ram())):
        walk_axis(env, assist, total, "y", 160, max_f=400)
        rec("left_south")
        walk_axis(env, assist, total, "x", ROOM_06_BLOCK_X, max_f=300)
        rec("under_block")
        push_dir(env, assist, total, "UP", frames=140)
        idle(env, assist, total, 8)
        rec("pushed")

    # Live warp: after block is north at (96,128), walk RIGHT from (112,141)
    # onto (128,141). Standing on (96,133) does not warp.
    if not done(_rs(env.get_ram())) and _rs(env.get_ram()).screen == ROOM_L5_PASSAGE_06:
        walk_axis(env, assist, total, "y", 141, max_f=240)
        walk_axis(env, assist, total, "x", 112, max_f=240)
        rec("pre_stair_112_141")
        walk_axis(env, assist, total, "x", 128, max_f=240)
        rec("walk_128_141")
        for direction in ("RIGHT", "UP", "DOWN"):
            if done(_rs(env.get_ram())):
                break
            for _ in range(24):
                snap = _rs(env.get_ram())
                if done(snap) or snap.screen != ROOM_L5_PASSAGE_06:
                    break
                if snap.link_y > 196:
                    _step(env, assist, total, nes_action("UP"))
                    continue
                _step(env, assist, total, nes_action(direction))
            rec(f"nudge_{direction}")

    idle(env, assist, total, 12)
    snap = _rs(env.get_ram())
    cellar = _in_cellar(snap)
    ok = (
        (cellar or (snap.level == LEVEL_5 and snap.screen == ROOM_L5_CELLAR_07))
        and snap.screen != 0x16
    )
    return {
        "path": "push68_right_128_141",
        "start": start,
        "log": log,
        "dest": snap.screen,
        "mode": snap.mode,
        "xy": [snap.link_x, snap.link_y],
        "tile": int(snap.colliding_tile),
        "cellar": cellar,
        "whistle": int(read_u8(env.get_ram(), ADDR_WHISTLE)),
        "keys": int(snap.keys),
        "success": ok,
        "south_key_drop": snap.screen == 0x16,
    }


def walk_east_from_64(env, assist, total: list[int]) -> dict:
    """0x64 east bomb-hole → 0x65. North-around the diamond; never (120,141).

    Spawn after cellar is ~(96,157) on the stairs column. Cardinal east/south
    into the diamond drops back into 0x07. Leave west to x=64, north wall
    y=93, east to x=208, then door y=141 RIGHT. South-band fallback if needed.
    """
    from zelda_i.dungeon.ops import idle, push_dir
    from zelda_i.ram import read_snapshot as _rs

    snap = _rs(env.get_ram())
    start = {"xy": [snap.link_x, snap.link_y], "room": snap.screen, "mode": snap.mode}

    def in_cellar(s) -> bool:
        return s.mode in CELLAR_MODES or (s.level == LEVEL_5 and s.screen == ROOM_L5_CELLAR_07)

    def go(steps) -> bool:
        for axis, tgt in steps:
            s = _rs(env.get_ram())
            if in_cellar(s):
                return False
            if s.mode == PLAY_MODE and s.screen == ROOM_L5_WEST_65:
                return True
            walk_axis(env, assist, total, axis, tgt, max_f=450)
            s = _rs(env.get_ram())
            if in_cellar(s):
                return False
            if s.mode == PLAY_MODE and s.screen == ROOM_L5_WEST_65:
                return True
        return not in_cellar(_rs(env.get_ram()))

    # North-around first (same leave used to take 0x06/0x64 stairs).
    ok = go((("x", 64), ("y", 93), ("x", 208), ("y", 141), ("x", 224)))
    via = "north93_east_bombhole"
    if not ok or in_cellar(_rs(env.get_ram())):
        # Still on 0x64: south-band fallback, stay off x=120.
        if _rs(env.get_ram()).screen == ROOM_L5_BLUE_64 and not in_cellar(_rs(env.get_ram())):
            ok = go((("x", 64), ("y", 189), ("x", 208), ("y", 141), ("x", 224)))
            via = "south189_east_bombhole"

    if _rs(env.get_ram()).screen == ROOM_L5_BLUE_64 and not in_cellar(_rs(env.get_ram())):
        push_dir(env, assist, total, "RIGHT", frames=240)
        idle(env, assist, total, 16)
        for _ in range(240):
            snap = _rs(env.get_ram())
            if snap.mode == PLAY_MODE and snap.screen == ROOM_L5_WEST_65:
                break
            if in_cellar(snap):
                break
            if snap.mode in (6, 7, 4, 16):
                _step(env, assist, total, nes_action("RIGHT"))
            else:
                _step(env, assist, total, nes_idle_action())
        idle(env, assist, total, 12)

    snap = _rs(env.get_ram())
    return {
        "path": via,
        "start": start,
        "dest": snap.screen,
        "mode": snap.mode,
        "xy": [snap.link_x, snap.link_y],
        "success": snap.level == LEVEL_5 and snap.screen == ROOM_L5_WEST_65 and snap.mode == PLAY_MODE,
    }
