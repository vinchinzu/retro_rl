"""Locked L5 floor: Level5Whistle65 -> 0x24 Digdogger -> TF bit 0x10.

Exact floor (do not deviate):
  0x65 east bomb -> 0x66 -> UP 0x56 -> 0x57 -> 0x47 -> 0x37 -> 0x27
  -> 0x26 -> 0x25 -> Digdogger 0x24. Whistle-shrink, then TF.

Already in 0x65. Skip 0x65 north. 0x64 east hop only if somehow on 0x64.
0x66 UP to 0x56 (not make_west65_controller / EastKey).
One env, assisted, stop_record. No Level5Complete / STATUS without TF 0x10.
"""
from __future__ import annotations

import importlib.util
from dataclasses import replace
from pathlib import Path

from retro_harness.env import make_env, reset_obs
from retro_harness.nes import nes_action, nes_idle_action
from retro_harness.segment_runner import configure_headless, save_rgb_png, write_json_report
from zelda_i.assist import UnlimitedHealthAssist
from zelda_i.dungeon import DungeonPhase, GenericDungeonRoomController
from zelda_i.dungeon_ids import object_name
from zelda_i.dungeon_ops import exit_door, idle, push_dir
from zelda_i.level5_dungeon import (
    LEVEL_5,
    ROOM_25_SPEC,
    ROOM_26_SPEC,
    ROOM_27_SPEC,
    ROOM_66_SPEC,
    Level5PolsVoiceController,
    level5_in_room_24,
    level5_in_room_25,
    level5_in_room_26,
    level5_room_25_cleared,
    level5_room_26_cleared,
    level5_room_27_cleared,
    level5_room_66_cleared,
)
from zelda_i.level5_path import (
    bomb_east_from_65,
    select_b_item_menu,
    walk_axis,
    walk_east_from_64,
    walk_west_from_25,
    walk_west_from_26,
    walk_west_from_27,
)
from zelda_i.paths import GAME, GAME_DIR, RECORDINGS_DIR
from zelda_i.ram import ADDR_TRIFORCE, ADDR_WHISTLE, PLAY_MODE, read_snapshot, read_u8

HERE = Path(__file__).resolve()
_spec = importlib.util.spec_from_file_location("l4stitch", HERE.parent / "_stitch_l4_to_24_door.py")
l4 = importlib.util.module_from_spec(_spec)
assert _spec.loader is not None
_spec.loader.exec_module(l4)

_wspec = importlib.util.spec_from_file_location("l5whistle", HERE.parent / "_probe_l5_whistle_path.py")
w = importlib.util.module_from_spec(_wspec)
assert _wspec.loader is not None
_wspec.loader.exec_module(w)

TAG = "l5_w65_locked_tf"
TF_BIT = 0x10
DIGDOGGER = 0x38
ZOL_TYPES = l4.ZOL_TYPES
STITCH_MAP = {
    0x05: "six-Darknut",
    0x14: "L5 triforce",
    0x24: "Digdogger",
    0x25: "west Pols Voice",
    0x26: "west Gibdos",
    0x27: "mixed Pols/Gibdo/Keese",
    0x37: "Darknuts + compass",
    0x47: "north Gibdos",
    0x55: "west Zols",
    0x56: "north Dodongos",
    0x57: "east Zols",
    0x64: "Blue Darknut stairs",
    0x65: "west Gibdo pocket",
    0x66: "3x Gibdo first key",
    0x76: "L5 entrance",
}


def room_name(screen):
    return STITCH_MAP.get(int(screen), f"room 0x{int(screen):02x}")


def pin(env):
    s = read_snapshot(env.get_ram())
    tf = int(read_u8(env.get_ram(), ADDR_TRIFORCE))
    return {
        "mode": s.mode,
        "level": s.level,
        "screen": s.screen,
        "screen_hex": f"0x{s.screen:02x}",
        "name": room_name(s.screen) if s.level == 5 else None,
        "x": s.link_x,
        "y": s.link_y,
        "keys": int(s.keys),
        "bombs": int(s.bombs),
        "doors": int(s.cur_opened_doors),
        "mask": int(s.open_doorway_mask),
        "all_dead": int(s.room_all_dead),
        "whistle": int(read_u8(env.get_ram(), ADDR_WHISTLE)),
        "triforce": tf,
        "tf_hex": hex(tf),
        "tf_l5_bit": bool(tf & TF_BIT),
    }


def live_objects(env):
    s = read_snapshot(env.get_ram())
    out = []
    for o in s.objects:
        if 1 <= o.slot <= 12 and o.type_id not in (0, 0xFF):
            out.append(
                {
                    "slot": o.slot,
                    "type": o.type_id,
                    "type_hex": f"0x{o.type_id:02x}",
                    "name": object_name(o.type_id),
                    "hp": o.hp,
                    "x": o.x,
                    "y": o.y,
                }
            )
    return out


def live_of(env, types, *, hp_required=True):
    out = []
    for o in read_snapshot(env.get_ram()).objects:
        if 1 <= o.slot <= 12 and o.type_id in types:
            if (not hp_required) or o.hp > 0:
                out.append(o)
    return out


def hop_ok(env, dest):
    s = read_snapshot(env.get_ram())
    return s.level == LEVEL_5 and s.screen == dest


DOOR_PATHS = {
    "RIGHT": (
        (("y", 141), ("x", 208), ("y", 141), ("x", 224)),
        (("y", 189), ("x", 208), ("y", 141), ("x", 224)),
        (("y", 93), ("x", 208), ("y", 141), ("x", 224)),
        (("y", 109), ("x", 208), ("y", 141), ("x", 224)),
        (("x", 208), ("y", 141), ("x", 224)),
    ),
    "LEFT": (
        (("y", 189), ("x", 32), ("y", 141)),
        (("y", 141), ("x", 32)),
        (("y", 93), ("x", 32), ("y", 141)),
    ),
    "UP": (
        (("x", 80), ("y", 93), ("x", 120)),
        (("x", 160), ("y", 93), ("x", 120)),
        (("y", 173), ("x", 120), ("y", 93)),
        (("y", 93), ("x", 120)),
        (("x", 120), ("y", 93)),
    ),
    "DOWN": (
        (("x", 120), ("y", 205)),
        (("y", 189), ("x", 120), ("y", 205)),
    ),
}


def take_door(env, assist, n, direction, expect):
    """Multi-approach door walk. 0x56 east uses the live recon path first."""
    room0 = read_snapshot(env.get_ram()).screen
    tried = []
    for steps in DOOR_PATHS[direction]:
        if read_snapshot(env.get_ram()).screen != room0:
            break
        for axis, tgt in steps:
            walk_axis(env, assist, n, axis, tgt, max_f=400)
        push_dir(env, assist, n, direction, frames=280)
        idle(env, assist, n, 16)
        wait_play(env, assist, n, expect, max_f=200)
        snap = read_snapshot(env.get_ram())
        rec = {"steps": list(steps), "dest": f"0x{snap.screen:02x}", "xy": [snap.link_x, snap.link_y], "mode": snap.mode}
        tried.append(rec)
        print("DOOR", direction, rec, flush=True)
        if snap.screen == expect and snap.mode == PLAY_MODE:
            return {"ok": True, "dest": snap.screen, "tried": tried}
        if snap.screen != room0 and snap.mode == PLAY_MODE:
            return {"ok": snap.screen == expect, "dest": snap.screen, "tried": tried}
    # last-chance exit_door
    hop = exit_door(env, assist, n, direction)
    wait_play(env, assist, n, expect, max_f=200)
    snap = read_snapshot(env.get_ram())
    return {"ok": snap.screen == expect, "dest": snap.screen, "tried": tried, "exit_door": hop}


def wait_play(env, assist, n, room, max_f=360):
    saw = False
    for _ in range(max_f):
        s = read_snapshot(env.get_ram())
        if s.level == LEVEL_5 and s.screen == room:
            saw = True
            if s.mode == PLAY_MODE and not s.transitioning:
                idle(env, assist, n, 16)
                return True
        env.step(nes_idle_action())
        n[0] += 1
        assist.apply_env(env, frame=n[0])
    s = read_snapshot(env.get_ram())
    return saw and s.level == LEVEL_5 and s.screen == room


def fight_spec(env, spec, assist, n, controller=None):
    ctl = controller or GenericDungeonRoomController(spec)
    for _ in range(spec.max_frames):
        assist.apply_env(env, frame=n[0])
        action = ctl.step(read_snapshot(env.get_ram()))
        env.step(action.action)
        n[0] += 1
        if ctl.success or ctl.phase is DungeonPhase.FAILED:
            break
    return ctl


def maybe_clear(env, spec, assist, n, types, *, controller=None, already=None, type_only=()):
    """Fight only if live enemies remain. Already-cleared rooms walk through."""
    if already is not None and already(env.get_ram()):
        return {"skipped": True, "reason": "predicate_clear", "ok": True}
    live = []
    for o in read_snapshot(env.get_ram()).objects:
        if 1 <= o.slot <= 12 and o.type_id in types:
            if o.type_id in type_only or o.hp > 0:
                live.append(o)
    if not live:
        return {"skipped": True, "reason": "no_live", "ok": True, "all_dead": int(read_snapshot(env.get_ram()).room_all_dead)}
    ctl = fight_spec(env, spec, assist, n, controller=controller)
    live2 = []
    for o in read_snapshot(env.get_ram()).objects:
        if 1 <= o.slot <= 12 and o.type_id in types:
            if o.type_id in type_only or o.hp > 0:
                live2.append(o)
    ok = (already(env.get_ram()) if already is not None else (ctl.success or not live2))
    return {
        "skipped": False,
        "ok": bool(ok),
        "ctl": ctl.report(),
        "live_after": len(live2),
    }


def walk_north_from_66(env, assist, n):
    """0x66 west mouth -> wait shutter -> south band y=173 -> x=120 -> UP 0x56.

    No 0x65-north fallback. Locked route is 0x66 UP 0x56 only.
    """
    idle(env, assist, n, 80)
    s = read_snapshot(env.get_ram())
    doors0 = {"doors": int(s.cur_opened_doors), "mask": int(s.open_doorway_mask), "all_dead": int(s.room_all_dead)}
    print("66_doors_after_idle", doors0, pin(env), flush=True)
    walk_axis(env, assist, n, "y", 173, max_f=400)
    walk_axis(env, assist, n, "x", 120, max_f=500)
    idle(env, assist, n, 40)
    walk_axis(env, assist, n, "y", 93, max_f=400)
    walk_axis(env, assist, n, "x", 120, max_f=200)
    push_dir(env, assist, n, "UP", frames=280)
    idle(env, assist, n, 16)
    wait_play(env, assist, n, 0x56)
    if hop_ok(env, 0x56):
        s = read_snapshot(env.get_ram())
        return {"path": "idle_south173_x120_up", "dest": s.screen, "xy": [s.link_x, s.link_y], "doors0": doors0, "success": True}
    # one more push after recenter
    walk_axis(env, assist, n, "x", 120, max_f=200)
    walk_axis(env, assist, n, "y", 93, max_f=200)
    push_dir(env, assist, n, "UP", frames=240)
    idle(env, assist, n, 16)
    wait_play(env, assist, n, 0x56)
    s = read_snapshot(env.get_ram())
    return {
        "path": "idle_south173_x120_up",
        "dest": s.screen,
        "xy": [s.link_x, s.link_y],
        "doors0": doors0,
        "success": hop_ok(env, 0x56),
    }


def _step(env, assist, n, action):
    env.step(action)
    n[0] += 1
    assist.apply_env(env, frame=n[0])


def fight_digdogger_and_tf(env, assist, n):
    """Hold-B recorder from midroom. Shrink 0x38 -> 0x18, sword, north TF.

    Live: tap-B from the east door does not shrink. Hold B 12f, freeze ~240f.
    South-east (192,189) and center (120,141) both work on Level5Whistle24.
    """
    walk_axis(env, assist, n, "y", 141, max_f=300)
    walk_axis(env, assist, n, "x", 120, max_f=400)
    idle(env, assist, n, 8)
    menu = select_b_item_menu(env, assist, n, 5)
    print("MENU", menu, flush=True)
    shrunk = False
    after_b = None
    for attempt in range(6):
        if attempt == 3:
            walk_axis(env, assist, n, "y", 189, max_f=300)
            walk_axis(env, assist, n, "x", 192, max_f=300)
            idle(env, assist, n, 8)
        for _ in range(12):
            _step(env, assist, n, nes_action("B"))
        for i in range(22):
            idle(env, assist, n, 12)
            s = read_snapshot(env.get_ram())
            live = [
                o
                for o in s.objects
                if 1 <= o.slot <= 12 and o.type_id not in (0, 0xFF, 0x55, 0x4E, 0x40, 0x68) and o.hp > 0
            ]
            types = [(hex(o.type_id), o.hp) for o in live]
            print("SONG", attempt, i, types, flush=True)
            if any(o.type_id == 0x18 for o in live) or (live and all(o.type_id != DIGDOGGER for o in live)):
                shrunk = True
                after_b = live_objects(env)
                break
        if shrunk:
            break
    if after_b is None:
        after_b = live_objects(env)
    print("SHRUNK", shrunk, after_b, flush=True)
    fight = None
    small = live_of(env, (0x18,))
    if small:
        fight = w.fight_type(env, assist, n, 0x24, 0x18, expected=len(small))
        idle(env, assist, n, 16)
        print("BOSS18", {k: fight[k] for k in fight if k != "controller"}, flush=True)
    leftovers = [
        o
        for o in read_snapshot(env.get_ram()).objects
        if 1 <= o.slot <= 12 and o.type_id not in (0, 0xFF, 0x55, 0x4E, 0x40, 0x1A) and o.hp > 0
    ]
    extra = None
    if leftovers:
        extra = w.fight_type(env, assist, n, 0x24, leftovers[0].type_id, expected=len(leftovers))
        idle(env, assist, n, 12)
        print("EXTRA", {k: extra[k] for k in extra if k != "controller"}, flush=True)
    for tx, ty in ((120, 141), (144, 141), (96, 141), (160, 141), (80, 141), (120, 125), (120, 157)):
        walk_axis(env, assist, n, "y", ty, max_f=200)
        walk_axis(env, assist, n, "x", tx, max_f=200)
        idle(env, assist, n, 8)
    walk_axis(env, assist, n, "y", 141, max_f=300)
    walk_axis(env, assist, n, "x", 120, max_f=300)
    walk_axis(env, assist, n, "y", 93, max_f=400)
    push_dir(env, assist, n, "UP", frames=240)
    idle(env, assist, n, 20)
    wait_play(env, assist, n, 0x14, max_f=280)
    tf0 = int(read_u8(env.get_ram(), ADDR_TRIFORCE))
    snap = read_snapshot(env.get_ram())
    tf_walk = None
    if snap.screen == 0x14 or snap.room_item_id == 0x1B:
        tf_walk = w.hunt_item(env, assist, n, ADDR_TRIFORCE)
        idle(env, assist, n, 16)
    tf1 = int(read_u8(env.get_ram(), ADDR_TRIFORCE))
    if not (tf1 & TF_BIT) and hop_ok(env, 0x14):
        for tx, ty in ((120, 141), (120, 125), (120, 157), (96, 141), (144, 141), (120, 109)):
            walk_axis(env, assist, n, "y", ty, max_f=200)
            walk_axis(env, assist, n, "x", tx, max_f=200)
            idle(env, assist, n, 12)
            tf1 = int(read_u8(env.get_ram(), ADDR_TRIFORCE))
            if tf1 & TF_BIT:
                break
    idle(env, assist, n, 20)
    snap = read_snapshot(env.get_ram())
    dead = not any(o.type_id == DIGDOGGER and o.hp > 0 for o in snap.objects if 1 <= o.slot <= 12)
    return {
        "menu": menu,
        "after_whistle_objs": after_b,
        "fight": None if fight is None else {k: fight[k] for k in fight if k != "controller"},
        "extra": None if extra is None else {k: extra[k] for k in extra if k != "controller"},
        "tf_walk": None if tf_walk is None else {k: v for k, v in tf_walk.items() if k != "hits"},
        "tf_in": tf0,
        "tf_out": tf1,
        "tf_l5": bool(tf1 & TF_BIT),
        "digdogger_dead": dead,
        "room": snap.screen,
        "xy": [snap.link_x, snap.link_y],
    }


def from_56_to_24(env, assist, n, seams):
    hop57 = take_door(env, assist, n, "RIGHT", 0x57)
    wait_play(env, assist, n, 0x57)
    if not hop_ok(env, 0x57):
        seams.append({"name": "0x56 east to 0x57 east Zols", "ok": False, "hop": hop57, **pin(env)})
        return "fail_hop_56_to_57"
    # ROM 0x57 N=open W=open. Live: y=93 x=120 UP reaches 0x47 with Zols still up.
    # Do not block the locked floor on a Zol timeout.
    live57 = live_of(env, ZOL_TYPES)
    c57 = {"skipped": True, "reason": "north_open_no_clear_required", "live": len(live57), "ok": True}
    if live57:
        print("57_live_skip_clear", len(live57), pin(env), flush=True)
    seams.append({"name": "0x57 east Zols", "ok": hop_ok(env, 0x57), "clear": c57, **pin(env)})
    print("57", True, c57, pin(env), flush=True)
    hop47 = take_door(env, assist, n, "UP", 0x47)
    wait_play(env, assist, n, 0x47)
    if not hop_ok(env, 0x47):
        seams.append({"name": "0x57 north to 0x47 north Gibdos", "ok": False, "hop": hop47, **pin(env)})
        return "fail_hop_57_to_47"
    # ROM 0x47 N/S/W=open. Walk north; do not block on Gibdo clear.
    seams.append({"name": "0x47 north Gibdos", "ok": hop_ok(env, 0x47), "clear": {"skipped": True, "reason": "north_open"}, **pin(env)})
    print("47", hop_ok(env, 0x47), pin(env), flush=True)
    hop37 = l4.walk_north_from_47(env, assist, n)
    wait_play(env, assist, n, 0x37)
    if not hop_ok(env, 0x37):
        hop37 = take_door(env, assist, n, "UP", 0x37)
        wait_play(env, assist, n, 0x37)
    if not hop_ok(env, 0x37):
        seams.append({"name": "0x47 north to 0x37 Darknuts + compass", "ok": False, "hop": hop37, **pin(env)})
        return "fail_hop_47_to_37"
    # ROM 0x37 N/S=open.
    seams.append({"name": "0x37 Darknuts + compass", "ok": True, "clear": {"skipped": True, "reason": "north_open"}, **pin(env)})
    print("37", True, pin(env), flush=True)
    hop27 = take_door(env, assist, n, "UP", 0x27)
    wait_play(env, assist, n, 0x27)
    if not hop_ok(env, 0x27):
        seams.append({"name": "0x37 north to 0x27 mixed Pols/Gibdo/Keese", "ok": False, "hop": hop27, **pin(env)})
        return "fail_hop_37_to_27"
    # ROM 0x27 W=key. walk_west_from_27 spends a key. No clear required.
    seams.append({"name": "0x27 mixed Pols/Gibdo/Keese", "ok": hop_ok(env, 0x27), "clear": {"skipped": True, "reason": "west_key_open_after_spend"}, **pin(env)})
    print("27", hop_ok(env, 0x27), pin(env), flush=True)
    hop26 = walk_west_from_27(env, assist, n)
    wait_play(env, assist, n, 0x26)
    if not (level5_in_room_26(env.get_ram()) or hop_ok(env, 0x26)):
        seams.append({"name": "0x27 west key to 0x26 west Gibdos", "ok": False, "hop": hop26, **pin(env)})
        return "fail_hop_27_to_26"
    # ROM 0x26 W=open.
    seams.append({"name": "0x26 west Gibdos", "ok": True, "clear": {"skipped": True, "reason": "west_open"}, **pin(env)})
    print("26", True, pin(env), flush=True)
    hop25 = walk_west_from_26(env, assist, n)
    wait_play(env, assist, n, 0x25)
    if not (level5_in_room_25(env.get_ram()) or hop_ok(env, 0x25)):
        seams.append({"name": "0x26 west to 0x25 west Pols Voice", "ok": False, "hop": hop25, **pin(env)})
        return "fail_hop_26_to_25"
    # ROM 0x25 W=key. walk_west_from_25 spends a key. No clear required.
    seams.append({"name": "0x25 west Pols Voice", "ok": hop_ok(env, 0x25) or level5_in_room_25(env.get_ram()), "clear": {"skipped": True, "reason": "west_key"}, **pin(env)})
    print("25", True, pin(env), flush=True)
    hop24 = walk_west_from_25(env, assist, n)
    wait_play(env, assist, n, 0x24)
    door24 = level5_in_room_24(env.get_ram()) or hop_ok(env, 0x24)
    seams.append({"name": "0x24 Digdogger", "ok": door24, "hop": hop24, "objs": live_objects(env), **pin(env)})
    print("24", door24, pin(env), live_objects(env), flush=True)
    if not door24:
        return "fail_hop_25_to_24"
    return None


def main() -> int:
    configure_headless()
    out = RECORDINGS_DIR / "stitches"
    movie = out / "bk2_w65_locked_tf"
    movie.mkdir(parents=True, exist_ok=True)
    assist = UnlimitedHealthAssist(enabled=True)
    n = [0]
    seams = []
    blocker = None
    boss = None
    start = None
    final = None
    env = make_env(GAME, "Level5Whistle65", GAME_DIR, render_mode="rgb_array", record=str(movie))
    try:
        reset_obs(env)
        env.step(nes_idle_action())
        n[0] += 1
        assist.apply_env(env, frame=0)
        idle(env, assist, n, 12)
        start = pin(env)
        print("START", start, "objs", live_objects(env), flush=True)
        seams.append({"name": "0x65 west Gibdo pocket", "ok": start["screen"] == 0x65 and start["whistle"] == 1, **start})

        if start["whistle"] != 1:
            blocker = "start_whistle_not_1"
        elif start["screen"] == 0x64:
            hop65 = walk_east_from_64(env, assist, n)
            wait_play(env, assist, n, 0x65)
            ok65 = hop65.get("success") or hop_ok(env, 0x65)
            seams.append({"name": "0x64 Blue Darknut stairs east (fallback)", "ok": ok65, "hop": hop65, **pin(env)})
            print("64EAST_FALLBACK", ok65, pin(env), flush=True)
            if not ok65:
                blocker = "fail_hop_64_to_65"
        elif start["screen"] != 0x65:
            blocker = f"start_not_0x65_got_0x{start['screen']:02x}"

        if blocker is None:
            hop66 = bomb_east_from_65(env, assist, n)
            wait_play(env, assist, n, 0x66)
            ok66 = hop66.get("success") or hop_ok(env, 0x66)
            seams.append({"name": "0x66 3x Gibdo first key", "ok": ok66, "hop": hop66, **pin(env)})
            print("66", ok66, hop66, pin(env), live_objects(env), flush=True)
            if not ok66:
                blocker = "fail_hop_65_bomb_east_to_66"
            else:
                c66 = maybe_clear(env, ROOM_66_SPEC, assist, n, ROOM_66_SPEC.enemy_types, already=level5_room_66_cleared)
                ok66c = hop_ok(env, 0x66) and (c66.get("ok") or level5_room_66_cleared(env.get_ram()) or not live_of(env, ROOM_66_SPEC.enemy_types))
                seams.append({"name": "0x66 3x Gibdo first key (clear)", "ok": ok66c, "clear": c66, **pin(env)})
                print("66_clear", ok66c, c66, pin(env), flush=True)
                hop56 = walk_north_from_66(env, assist, n)
                wait_play(env, assist, n, 0x56)
                ok56 = hop56.get("success") or hop_ok(env, 0x56)
                seams.append({"name": "0x56 north Dodongos", "ok": ok56, "hop": hop56, **pin(env)})
                print("56", ok56, hop56, pin(env), flush=True)
                if not ok56:
                    blocker = "fail_hop_66_up_to_56"
                else:
                    blocker = from_56_to_24(env, assist, n, seams)
                    if blocker is None:
                        boss = fight_digdogger_and_tf(env, assist, n)
                        print("BOSS", boss, flush=True)
                        seams.append(
                            {
                                "name": "0x14 L5 triforce" if hop_ok(env, 0x14) else "0x24 Digdogger",
                                "ok": bool(boss.get("tf_l5")),
                                "boss": {k: boss[k] for k in boss if k != "after_whistle_objs"},
                                **pin(env),
                            }
                        )
                        if not boss.get("tf_l5"):
                            blocker = "tf_bit_0x10_not_set"

        final = pin(env)
        shot = out / f"{TAG}_final.png"
        save_rgb_png(env.step(nes_idle_action())[0], shot)
        if tf_l5_now := bool(final and final.get("tf_l5_bit")):
            from retro_harness.env import state_path, write_state_bytes
            from zelda_i.dungeon_trace import write_state_provenance
            write_state_bytes(state_path(GAME_DIR, GAME, "Level5Triforce"), env.em.get_state())
            write_state_provenance(
                GAME_DIR / "custom_integrations" / GAME / "Level5Triforce.state",
                source_state_path=GAME_DIR / "custom_integrations" / GAME / "Level5Whistle65.state",
                request={
                    "segment": "Level5Triforce",
                    "predecessor_entry": True,
                    "start_state": "Level5Whistle65",
                    "via": "locked 65e-bomb 66-up 56-57-47-37-27-26-25-24 whistle-shrink TF 0x10",
                    "key_poke": False,
                    "door_poke": False,
                    "bomb_count_poke": False,
                    "selected_item_poke": False,
                },
                selected_trial={
                    "success": True,
                    "room": int(final["screen"]),
                    "triforce_0x0671": int(final["triforce"]),
                    "tf_l5_bit": True,
                    "whistle_0x065C": 1,
                },
                natural_entry=False,
            )
            print("PINNED Level5Triforce (TF bit 0x10 real)", flush=True)
        if hasattr(env, "stop_record"):
            env.stop_record()
    finally:
        try:
            if hasattr(env, "stop_record"):
                env.stop_record()
        except Exception:
            pass
        env.close()

    bk2s = sorted(movie.glob("*.bk2"), key=lambda p: p.stat().st_mtime)
    bk2 = str(bk2s[-1]) if bk2s else None
    tf = None if final is None else final.get("triforce")
    tf_l5 = bool(final and final.get("tf_l5_bit"))
    report = {
        "ok": tf_l5 and blocker is None,
        "segment": TAG,
        "continuous_emulator_session": True,
        "track": "assisted",
        "status_claim": False,
        "level5_complete_claim": False,
        "start_state": "Level5Whistle65",
        "end_claim": "l5_triforce_bit_0x10" if tf_l5 else None,
        "whistle_0x065C": None if final is None else final.get("whistle"),
        "triforce_0x0671": tf,
        "tf_l5_bit_0x10_real": tf_l5,
        "digdogger_dead": None if boss is None else boss.get("digdogger_dead"),
        "boss": None if boss is None else {k: boss[k] for k in boss if k != "after_whistle_objs"},
        "total_frames": n[0],
        "start": start,
        "final": final,
        "seams": seams,
        "room_sequence": [
            f"0x{s.get('screen'):02x} {s.get('name')}" for s in seams if s.get("screen") is not None
        ],
        "blocker": blocker,
        "bk2": bk2,
        "png": str(out / f"{TAG}_final.png"),
        "pokes": False,
        "path_note": (
            "Locked: Level5Whistle65 already 0x65 west Gibdo pocket. "
            "East bomb 0x66, UP 0x56, east 0x57, north 0x47, north 0x37, "
            "north 0x27, west 0x26, west 0x25, west 0x24. Skip 0x65 north. "
            "Did not claim Level5Complete unless TF bit 0x10 is real."
        ),
    }
    path = out / f"{TAG}.json"
    write_json_report(path, report)
    print(
        f"wrote {path} frames={n[0]} bk2={bk2} tf={tf} tf_l5={tf_l5} "
        f"blocker={blocker} whistle={report['whistle_0x065C']}",
        flush=True,
    )
    return 0 if report["ok"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
