"""Level 5 Digdogger / Triforce north hops.

0x57 east Zols → 0x47 north Gibdos. Do not clear Zols (statue 0x5f seals y≈125).

Room specs and stop predicates remain in ``level5_dungeon``.
Import from ``zelda_i.level5.path`` (public facade).
"""

from __future__ import annotations

from retro_harness.nes import nes_action, nes_idle_action

from zelda_i.level5.dungeon import LEVEL_5
from zelda_i.level5.path import NORTH_DOOR_X, _step, walk_axis
from zelda_i.ram import PLAY_MODE

ROOM_L5_EAST_ZOLS = 0x57
ROOM_L5_NORTH_GIBDOS = 0x47
# 0x57 foes_item statue after a Zol clear. Blocks the north channel at y≈125.
STATUE_5F_TYPE = 0x5F
STATUE_5F_X = 128
STATUE_5F_Y = 128
DIAMOND_NORTH_Y = 109
NORTH_DOOR_Y = 93


def walk_north_from_57(env, assist, total: list[int]) -> dict:
    """0x57 east Zols → 0x47 north Gibdos. ROM N=open.

    Do **not** clear the Zols first. Secret ``foes_item`` drops statue ``0x5f``
    at (128, 128) and Link cannot get north of y≈125 after that.

    Try in this order (L5 Forward walked the north door):
    1. key-north — x=120, y=93, push UP (key spends if the bit is locked)
    2. push the center 0x5f
    3. diamond north-around — y=109 band, then door column
    """
    from zelda_i.dungeon.ops import idle, push_dir
    from zelda_i.ram import read_snapshot as _rs

    snap = _rs(env.get_ram())
    keys0 = int(snap.keys)
    log = [{"step": "start", "xy": [snap.link_x, snap.link_y], "doors": int(snap.cur_opened_doors), "keys": keys0}]

    def at_47(s) -> bool:
        return s.level == LEVEL_5 and s.screen == ROOM_L5_NORTH_GIBDOS and s.mode == PLAY_MODE

    def push_up(frames: int = 280) -> None:
        push_dir(env, assist, total, "UP", frames=frames)
        idle(env, assist, total, 16)
        for _ in range(240):
            s = _rs(env.get_ram())
            if at_47(s) or s.screen != ROOM_L5_EAST_ZOLS:
                break
            if s.mode in (6, 7, 4, 16):
                _step(env, assist, total, nes_action("UP"))
            else:
                _step(env, assist, total, nes_idle_action())
        idle(env, assist, total, 8)

    # 1. key-north
    walk_axis(env, assist, total, "y", 141, max_f=400)
    walk_axis(env, assist, total, "x", NORTH_DOOR_X, max_f=500)
    walk_axis(env, assist, total, "y", NORTH_DOOR_Y, max_f=400)
    snap = _rs(env.get_ram())
    log.append({"step": "key_north_stand", "xy": [snap.link_x, snap.link_y], "doors": int(snap.cur_opened_doors)})
    push_up(280)
    snap = _rs(env.get_ram())
    log.append({"step": "key_north", "xy": [snap.link_x, snap.link_y], "room": snap.screen, "keys": int(snap.keys)})
    if at_47(snap):
        return {
            "path": "key_north",
            "log": log,
            "keys_in": keys0,
            "keys_out": int(snap.keys),
            "key_spent": int(snap.keys) < keys0,
            "dest": snap.screen,
            "xy": [snap.link_x, snap.link_y],
            "mode": snap.mode,
            "success": True,
        }

    # 2. push center 0x5f
    statue = None
    for o in snap.objects:
        if 1 <= o.slot <= 12 and o.type_id == STATUE_5F_TYPE:
            statue = o
            break
    if statue is not None or snap.link_y > 120:
        walk_axis(env, assist, total, "y", STATUE_5F_Y + 16, max_f=300)
        walk_axis(env, assist, total, "x", STATUE_5F_X, max_f=300)
        walk_axis(env, assist, total, "y", STATUE_5F_Y, max_f=200)
        push_dir(env, assist, total, "UP", frames=80)
        idle(env, assist, total, 8)
        snap = _rs(env.get_ram())
        log.append({"step": "push_5f", "xy": [snap.link_x, snap.link_y], "room": snap.screen})
        walk_axis(env, assist, total, "x", NORTH_DOOR_X, max_f=300)
        walk_axis(env, assist, total, "y", NORTH_DOOR_Y, max_f=300)
        push_up(240)
        snap = _rs(env.get_ram())
        log.append({"step": "after_push_5f", "xy": [snap.link_x, snap.link_y], "room": snap.screen})
        if at_47(snap):
            return {
                "path": "push_5f",
                "log": log,
                "keys_in": keys0,
                "keys_out": int(snap.keys),
                "key_spent": int(snap.keys) < keys0,
                "dest": snap.screen,
                "xy": [snap.link_x, snap.link_y],
                "mode": snap.mode,
                "success": True,
            }

    # 3. diamond north-around
    for name, steps in (
        ("west109", (("y", DIAMOND_NORTH_Y), ("x", 96), ("x", NORTH_DOOR_X), ("y", NORTH_DOOR_Y))),
        ("east109", (("y", DIAMOND_NORTH_Y), ("x", 160), ("x", NORTH_DOOR_X), ("y", NORTH_DOOR_Y))),
    ):
        if at_47(_rs(env.get_ram())):
            break
        if _rs(env.get_ram()).screen != ROOM_L5_EAST_ZOLS:
            break
        for axis, tgt in steps:
            walk_axis(env, assist, total, axis, tgt, max_f=400)
            snap = _rs(env.get_ram())
            log.append({"step": f"{name}:{axis}:{tgt}", "xy": [snap.link_x, snap.link_y], "room": snap.screen})
            if at_47(snap):
                break
        push_up(280)
        snap = _rs(env.get_ram())
        log.append({"step": name, "xy": [snap.link_x, snap.link_y], "room": snap.screen})
        if at_47(snap):
            return {
                "path": f"diamond_{name}",
                "log": log,
                "keys_in": keys0,
                "keys_out": int(snap.keys),
                "key_spent": int(snap.keys) < keys0,
                "dest": snap.screen,
                "xy": [snap.link_x, snap.link_y],
                "mode": snap.mode,
                "success": True,
            }

    snap = _rs(env.get_ram())
    return {
        "path": "north_blocked",
        "log": log,
        "keys_in": keys0,
        "keys_out": int(snap.keys),
        "key_spent": int(snap.keys) < keys0,
        "dest": snap.screen,
        "xy": [snap.link_x, snap.link_y],
        "mode": snap.mode,
        "success": at_47(snap),
    }
