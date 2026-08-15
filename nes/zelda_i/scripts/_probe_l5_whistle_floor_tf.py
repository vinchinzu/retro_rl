"""Level5Whistle: dump 0x04, exit cellar, walk live graph to 0x24, Digdogger, TF 0x10.

No door/key/item pokes. Survival OK. Not a Clean STATUS claim.
"""
from __future__ import annotations

import importlib.util
from pathlib import Path

from retro_harness.nes import nes_action, nes_idle_action
from retro_harness.segment_runner import configure_headless
from zelda_i.dungeon_ops import idle, push_dir
from zelda_i.level5_path import (
    cellar_other_mouth,
    exit_whistle_04,
    take_center_stairs_06,
    walk_east_from_05,
    walk_west_from_25,
    walk_west_from_26,
    walk_west_from_27,
)
from zelda_i.level9_stairs import (
    dest_report,
    on_stair_tile,
    walk_to_step,
)
from zelda_i.paths import RECORDINGS_DIR
from zelda_i.ram import (
    ADDR_TRIFORCE,
    ADDR_WHISTLE,
    PLAY_MODE,
    read_snapshot,
    read_u8,
)

HERE = Path(__file__).resolve()
_spec = importlib.util.spec_from_file_location("l5whistle", HERE.parent / "_probe_l5_whistle_path.py")
w = importlib.util.module_from_spec(_spec)
assert _spec.loader is not None
_spec.loader.exec_module(w)

TF_BIT = 0x10
CELLAR = (9, 10, 11, 16)

DOOR_PATHS = {
    "RIGHT": (
        (("y", 93), ("x", 208), ("y", 141), ("x", 224)),
        (("y", 141), ("x", 224)),
        (("y", 189), ("x", 224), ("y", 141)),
        (("x", 208), ("y", 141), ("x", 224)),
    ),
    "LEFT": (
        (("y", 93), ("x", 32), ("y", 141)),
        (("y", 141), ("x", 32)),
        (("y", 189), ("x", 32), ("y", 141)),
        (("x", 208), ("y", 189), ("x", 32), ("y", 141)),
        (("y", 117), ("x", 32), ("y", 141)),
    ),
    "UP": (
        (("x", 80), ("y", 93), ("x", 120)),
        (("x", 160), ("y", 93), ("x", 120)),
        (("y", 93), ("x", 120)),
        (("x", 120), ("y", 93)),
        (("y", 109), ("x", 120), ("y", 93)),
    ),
    "DOWN": (
        (("x", 120), ("y", 205)),
        (("y", 189), ("x", 120), ("y", 205)),
        (("x", 80), ("y", 205), ("x", 120)),
    ),
}


def raw_goto(env, assist, total, x, y, *, y_first=True, max_f=500, tol=2):
    last = None
    stall = 0
    trail = []
    for i in range(max_f):
        snap = read_snapshot(env.get_ram())
        if abs(snap.link_x - x) <= tol and abs(snap.link_y - y) <= tol:
            return True, trail
        frame = walk_to_step(snap, x, y, y_first=y_first, tol=tol)
        if frame.reason == "walk_arrived":
            return True, trail
        w.step(env, assist, total, frame.action)
        snap2 = read_snapshot(env.get_ram())
        pos = (snap2.link_x, snap2.link_y, snap2.mode, snap2.screen)
        if i % 20 == 0 or pos[:2] != (snap.link_x, snap.link_y):
            trail.append(
                {
                    "i": i,
                    "xy": [snap2.link_x, snap2.link_y],
                    "mode": snap2.mode,
                    "room": f"0x{snap2.screen:02x}",
                    "tile": int(snap2.colliding_tile),
                    "stair": bool(on_stair_tile(snap2)),
                    "reason": frame.reason,
                }
            )
        if pos == last:
            stall += 1
            if stall >= 40:
                return False, trail
        else:
            stall = 0
        last = pos
    return False, trail


def hold(env, assist, total, d, n):
    for _ in range(n):
        w.step(env, assist, total, nes_action(d))


def wait_play(env, assist, total, room=None, max_f=280):
    for _ in range(max_f):
        snap = read_snapshot(env.get_ram())
        room_ok = room is None or snap.screen == room
        if snap.mode == PLAY_MODE and room_ok and not snap.transitioning:
            idle(env, assist, total, 10)
            return True
        w.step(env, assist, total, nes_idle_action())
    return False


def take_door(env, assist, total, direction, expect=None):
    room0 = read_snapshot(env.get_ram()).screen
    tried = []
    for steps in DOOR_PATHS[direction]:
        snap = read_snapshot(env.get_ram())
        if snap.screen != room0:
            break
        for axis, tgt in steps:
            w.walk_axis(env, assist, total, axis, tgt, max_f=360)
        push_dir(env, assist, total, direction, frames=220)
        idle(env, assist, total, 12)
        wait_play(env, assist, total, max_f=200)
        snap = read_snapshot(env.get_ram())
        rec = {
            "steps": list(steps),
            "dest": f"0x{snap.screen:02x}",
            "mode": snap.mode,
            "xy": [snap.link_x, snap.link_y],
        }
        tried.append(rec)
        print("DOOR", direction, rec, flush=True)
        if snap.screen != room0 and snap.mode == PLAY_MODE:
            if expect is None or snap.screen == expect:
                return {"ok": True, "dest": snap.screen, "tried": tried}
            return {"ok": False, "dest": snap.screen, "tried": tried, "wrong": True}
    snap = read_snapshot(env.get_ram())
    return {"ok": snap.screen != room0, "dest": snap.screen, "tried": tried}


def dump_start(env, assist, total):
    snap = read_snapshot(env.get_ram())
    live = w.dump_live(snap, env.get_ram())
    tiles = []
    stands = (
        (136, 141),
        (160, 141),
        (176, 141),
        (120, 141),
        (112, 141),
        (176, 165),
        (176, 189),
        (120, 189),
        (48, 189),
        (48, 141),
        (48, 65),
        (80, 141),
        (192, 141),
        (208, 96),
        (120, 125),
    )
    # Do not walk the exit yet — only nearby alcove stands so we do not drop.
    for tx, ty in ((136, 141), (160, 141), (176, 141), (120, 141), (112, 141)):
        ok, trail = raw_goto(env, assist, total, tx, ty, y_first=False, max_f=240)
        snap = read_snapshot(env.get_ram())
        rec = {
            "stand": [tx, ty],
            "ok": ok,
            "xy": [snap.link_x, snap.link_y],
            "tile": int(snap.colliding_tile),
            "stair": bool(on_stair_tile(snap)),
            "mode": snap.mode,
            "room": f"0x{snap.screen:02x}",
        }
        tiles.append(rec)
        print("TILE", rec, flush=True)
        if snap.mode == PLAY_MODE and snap.screen != 0x04:
            break
    stair_hits = [t for t in tiles if t["stair"] or 0x70 <= t["tile"] <= 0x73]
    return {
        "live": live,
        "dest": dest_report(read_snapshot(env.get_ram())),
        "rom": w.rom_room(0x04),
        "tiles": tiles,
        "stair_tiles_0x70_0x73": stair_hits,
        "cellar_mode": bool(read_snapshot(env.get_ram()).mode in CELLAR),
        "whistle_0x065C": int(read_u8(env.get_ram(), ADDR_WHISTLE)),
    }


def try_06_south(env, assist, total):
    """Shorter: 0x06 key-south → 0x16 if the live/ROM door accepts a key."""
    keys0 = int(read_snapshot(env.get_ram()).keys)
    hop = take_door(env, assist, total, "DOWN", expect=0x16)
    hop["keys_in"] = keys0
    hop["keys_out"] = int(read_snapshot(env.get_ram()).keys)
    hop["key_spent"] = hop["keys_out"] < keys0
    return hop


def walk_16_toward_24(env, assist, total, hops):
    """From 0x16 try south 0x26 or west 0x15 then south 0x25."""
    south = take_door(env, assist, total, "DOWN")
    hops.append({"hop": "0x16_south", "dest": f"0x{south['dest']:02x}", "ok": south["ok"]})
    print("16SOUTH", hops[-1], flush=True)
    snap = read_snapshot(env.get_ram())
    if snap.screen == 0x26:
        west = walk_west_from_26(env, assist, total)
        hops.append({"hop": "0x26_west", "dest": f"0x{west.get('dest'):02x}", "ok": west.get("success")})
        print("26WEST", hops[-1], flush=True)
        if west.get("success"):
            return True
    if snap.screen == 0x16:
        west = take_door(env, assist, total, "LEFT")
        hops.append({"hop": "0x16_west", "dest": f"0x{west['dest']:02x}", "ok": west["ok"]})
        print("16WEST", hops[-1], flush=True)
        snap = read_snapshot(env.get_ram())
        if snap.screen == 0x15:
            south2 = take_door(env, assist, total, "DOWN")
            hops.append({"hop": "0x15_south", "dest": f"0x{south2['dest']:02x}", "ok": south2["ok"]})
            print("15SOUTH", hops[-1], flush=True)
            if read_snapshot(env.get_ram()).screen == 0x25:
                return True
    return read_snapshot(env.get_ram()).screen in (0x25, 0x24, 0x26)


def long_path_to_24(env, assist, total, hops):
    """0x06 stairs → 0x07 → 0x64 E → 0x65 U → 0x55 R → … → 0x25."""
    stairs = take_center_stairs_06(env, assist, total)
    hops.append(
        {
            "hop": "0x06_stairs",
            "dest": f"0x{stairs.get('dest'):02x}",
            "mode": stairs.get("mode"),
            "ok": stairs.get("success"),
        }
    )
    print("STAIRS06", hops[-1], stairs.get("xy"), flush=True)
    snap = read_snapshot(env.get_ram())
    if not (snap.mode in CELLAR or snap.screen == 0x07):
        return False

    cellar = cellar_other_mouth(env, assist, total)
    hops.append(
        {
            "hop": "0x07_other_mouth",
            "dest": f"0x{cellar.get('dest'):02x}",
            "ok": cellar.get("dest") == 0x64,
            "side": cellar.get("chose_side"),
        }
    )
    print("CELLAR07", hops[-1], cellar.get("xy"), flush=True)
    if read_snapshot(env.get_ram()).screen != 0x64:
        # If we landed 0x06, try left mouth explicitly.
        raw_goto(env, assist, total, 48, 189, y_first=True, max_f=500)
        push_dir(env, assist, total, "UP", frames=260)
        idle(env, assist, total, 16)
        wait_play(env, assist, total, max_f=240)
        hops.append({"hop": "0x07_left_retry", "dest": f"0x{read_snapshot(env.get_ram()).screen:02x}"})
        print("CELLAR07B", hops[-1], flush=True)
    if read_snapshot(env.get_ram()).screen != 0x64:
        return False

    chain = (
        (0x64, "RIGHT", 0x65),
        (0x65, "UP", 0x55),
        (0x55, "RIGHT", 0x56),
        (0x56, "RIGHT", 0x57),
        (0x57, "UP", 0x47),
        (0x47, "UP", 0x37),
        (0x37, "UP", 0x27),
    )
    for src, d, dst in chain:
        snap = read_snapshot(env.get_ram())
        if snap.screen != src:
            hops.append({"hop": f"0x{src:02x}_{d}", "ok": False, "at": f"0x{snap.screen:02x}"})
            print("OFFPATH", hops[-1], flush=True)
            return False
        hop = take_door(env, assist, total, d, expect=dst)
        hops.append({"hop": f"0x{src:02x}_{d}", "dest": f"0x{hop['dest']:02x}", "ok": hop["ok"] and hop["dest"] == dst})
        print("HOP", hops[-1], flush=True)
        if not hops[-1]["ok"]:
            return False
    west27 = walk_west_from_27(env, assist, total)
    hops.append({"hop": "0x27_west", "dest": f"0x{west27.get('dest'):02x}", "ok": west27.get("success")})
    print("27WEST", hops[-1], flush=True)
    if not west27.get("success"):
        return False
    west26 = walk_west_from_26(env, assist, total)
    hops.append({"hop": "0x26_west", "dest": f"0x{west26.get('dest'):02x}", "ok": west26.get("success")})
    print("26WEST", hops[-1], flush=True)
    return bool(west26.get("success"))


def digdogger_tf(env, assist, total):
    snap = read_snapshot(env.get_ram())
    if snap.screen != 0x24:
        west = walk_west_from_25(env, assist, total)
        print("25WEST", west.get("success"), "dest", f"0x{west.get('dest'):02x}", flush=True)
        if not west.get("success"):
            return {"ok": False, "reason": "west_25_missed_24", "west": {k: west[k] for k in west if k != "log"}}
    at24 = w.dump_live(read_snapshot(env.get_ram()), env.get_ram())
    w.shot(env, assist, total, "l5_24_enter")
    menu = w.select_whistle_menu(env, assist, total)
    for _ in range(5):
        w.step(env, assist, total, nes_action("B"))
        idle(env, assist, total, 50)
    idle(env, assist, total, 80)
    after_b = w.dump_live(read_snapshot(env.get_ram()), env.get_ram())
    print(
        "WHISTLE_B",
        menu,
        "objs",
        [(o["type_hex"], o["type_name"], o["hp"]) for o in after_b.get("objects") or []],
        flush=True,
    )
    fight = None
    snap = read_snapshot(env.get_ram())
    bosses = w.live_boss(snap)
    if bosses:
        fight = w.fight_type(env, assist, total, 0x24, 0x38, expected=len(bosses))
        idle(env, assist, total, 16)
        print("BOSS", fight.get("ok"), "end", fight.get("end_n"), flush=True)
    snap = read_snapshot(env.get_ram())
    leftovers = [
        o
        for o in snap.objects
        if 1 <= o.slot <= 12 and o.type_id not in (0, 0xFF, 0x55, 0x4E, 0x40, 0x1A) and o.hp > 0
    ]
    extra = None
    if leftovers:
        extra = w.fight_type(env, assist, total, int(snap.screen), leftovers[0].type_id, expected=len(leftovers))
        idle(env, assist, total, 12)
        print("EXTRA", extra.get("ok"), extra.get("end_n"), flush=True)
    # Heart 0x1A stands
    for tx, ty in ((120, 141), (144, 141), (96, 141), (120, 125), (120, 157), (160, 141), (80, 141)):
        w.walk_axis(env, assist, total, "y", ty, max_f=200)
        w.walk_axis(env, assist, total, "x", tx, max_f=200)
        idle(env, assist, total, 8)
    after_heart = w.dump_live(read_snapshot(env.get_ram()), env.get_ram())
    w.shot(env, assist, total, "l5_24_after_heart")
    room0 = read_snapshot(env.get_ram()).screen
    north = take_door(env, assist, total, "UP", expect=0x14)
    print("NORTH24", north, flush=True)
    snap = read_snapshot(env.get_ram())
    tf0 = int(read_u8(env.get_ram(), ADDR_TRIFORCE))
    tf_walk = None
    if snap.screen == 0x14 or snap.room_item_id == 0x1B:
        tf_walk = w.hunt_item(env, assist, total, ADDR_TRIFORCE)
        idle(env, assist, total, 24)
    ram = env.get_ram()
    snap = read_snapshot(ram)
    tf1 = int(read_u8(ram, ADDR_TRIFORCE))
    final = w.dump_live(snap, ram)
    png = w.shot(env, assist, total, "l5_14_triforce")
    rec = {
        "ok": bool(tf1 & TF_BIT),
        "at24": at24,
        "menu": menu,
        "after_whistle": {
            "objs": [(o["type_hex"], o["type_name"], o["hp"]) for o in after_b.get("objects") or []]
        },
        "fight": None if fight is None else {k: fight[k] for k in fight if k != "controller"},
        "extra": None if extra is None else {k: extra[k] for k in extra if k != "controller"},
        "after_heart": after_heart,
        "north": north,
        "tf_walk": None if tf_walk is None else {k: v for k, v in tf_walk.items() if k != "hits"},
        "tf_in": tf0,
        "tf_out": tf1,
        "tf_l5_bit": bool(tf1 & TF_BIT),
        "room": f"0x{snap.screen:02x}",
        "whistle_0x065C": int(read_u8(ram, ADDR_WHISTLE)),
        "final": final,
        "screenshot": png,
        "pokes": False,
        "status_claim": None,
    }
    w.write_dump("l5_24_whistle_boss", rec)
    if rec["ok"]:
        w.save_ckpt(
            env,
            "Level5Triforce",
            "Level5Whistle",
            {
                "segment": "Level5Triforce",
                "via": "0x24 whistle Digdogger, heart, north 0x14 TF bit 0x10",
                "key_poke": False,
                "door_poke": False,
                "bomb_count_poke": False,
                "selected_item_poke": False,
            },
            {
                "success": True,
                "room": int(snap.screen),
                "triforce_0x0671": tf1,
                "tf_l5_bit": True,
                "whistle_0x065C": 1,
            },
        )
    return rec


def main() -> dict:
    configure_headless()
    RECORDINGS_DIR.mkdir(parents=True, exist_ok=True)
    env, assist, _ = w.open_env("Level5Whistle")
    total = [1]
    hops = []
    checkpoints = []
    try:
        idle(env, assist, total, 12)
        snap = read_snapshot(env.get_ram())
        whistle = int(read_u8(env.get_ram(), ADDR_WHISTLE))
        print(
            "START",
            f"0x{snap.screen:02x}",
            "mode",
            snap.mode,
            [snap.link_x, snap.link_y],
            "whistle",
            whistle,
            "tile",
            snap.colliding_tile,
            flush=True,
        )
        if snap.screen != 0x04 or whistle < 1:
            rec = {
                "ok": False,
                "reason": "Level5Whistle_not_0x04_or_whistle_lost",
                "room": f"0x{snap.screen:02x}",
                "whistle_0x065C": whistle,
                "pokes": False,
                "status_claim": None,
            }
            w.write_dump("l5_04_exit", rec)
            return rec

        start_dump = dump_start(env, assist, total)
        w.shot(env, assist, total, "l5_04_exit_start")

        # Reload so the tile walk does not consume the alcove drop.
        env.close()
        env, assist, _ = w.open_env("Level5Whistle")
        total = [1]
        idle(env, assist, total, 12)

        # Also try walk_to_step mouths if the canned exit fails.
        walk = exit_whistle_04(env, assist, total)
        hops.append({"hop": "exit_whistle_04", **{k: walk[k] for k in walk if k != "log"}})
        print("EXIT04", hops[-1], flush=True)
        snap = read_snapshot(env.get_ram())
        dest = dest_report(snap)
        floor = w.dump_live(snap, env.get_ram())
        png = w.shot(env, assist, total, "l5_04_exit")
        exit_body = {
            "ok": snap.mode == PLAY_MODE and snap.screen != 0x04,
            "start": start_dump,
            "exit": {k: walk[k] for k in walk if k != "log"},
            "floor": floor,
            "dest": dest,
            "whistle_0x065C": int(read_u8(env.get_ram(), ADDR_WHISTLE)),
            "screenshot": png,
            "pokes": False,
            "status_claim": None,
        }
        w.write_dump("l5_04_exit", exit_body)
        if not exit_body["ok"]:
            # Last try: both mouths via walk_to_step from a fresh load.
            env.close()
            env, assist, _ = w.open_env("Level5Whistle")
            total = [1]
            idle(env, assist, total, 200)
            for mouth, seq in (
                ("right_drop_left_up", ((176, 141, False), (176, 189, True), (48, 189, False))),
                ("right_drop_right_up", ((176, 141, False), (176, 189, True), (192, 189, False))),
            ):
                for x, y, yf in seq:
                    raw_goto(env, assist, total, x, y, y_first=yf, max_f=500)
                    s = read_snapshot(env.get_ram())
                    print("MOUTH", mouth, [x, y], [s.link_x, s.link_y], "tile", s.colliding_tile, flush=True)
                    if s.mode == PLAY_MODE and s.screen != 0x04:
                        break
                push_dir(env, assist, total, "UP", frames=280)
                idle(env, assist, total, 16)
                wait_play(env, assist, total, max_f=240)
                s = read_snapshot(env.get_ram())
                hops.append({"hop": f"mouth_{mouth}", "dest": f"0x{s.screen:02x}", "mode": s.mode, "xy": [s.link_x, s.link_y]})
                print("MOUTH_END", hops[-1], flush=True)
                if s.mode == PLAY_MODE and s.screen != 0x04:
                    dest = dest_report(s)
                    floor = w.dump_live(s, env.get_ram())
                    png = w.shot(env, assist, total, "l5_04_exit")
                    exit_body.update({"ok": True, "floor": floor, "dest": dest, "screenshot": png, "via": mouth})
                    w.write_dump("l5_04_exit", exit_body)
                    break
            snap = read_snapshot(env.get_ram())
            if not (snap.mode == PLAY_MODE and snap.screen != 0x04):
                rec = {
                    "ok": False,
                    "failed_room": "0x04",
                    "reason": "basement_exit_failed",
                    "pose": w.dump_live(snap, env.get_ram()),
                    "dest": dest_report(snap),
                    "hops": hops,
                    "pokes": False,
                    "status_claim": None,
                }
                w.write_dump("l5_04_exit", rec)
                print("STOP still in cellar", rec["pose"].get("xy"), "mode", snap.mode, "tile", snap.colliding_tile, flush=True)
                return rec

        floor_room = read_snapshot(env.get_ram()).screen
        w.dump_and_save_room(
            env,
            assist,
            total,
            f"l5_{floor_room:02x}_floor",
            "Level5WhistleFloor",
            "Level5Whistle",
            f"0x04 cellar exit dest 0x{floor_room:02x}",
        )
        checkpoints.append("Level5WhistleFloor")

        # 0x05 EAST → 0x06
        if read_snapshot(env.get_ram()).screen == 0x05:
            east = walk_east_from_05(env, assist, total)
            hops.append({"hop": "0x05_east", "dest": f"0x{east.get('dest'):02x}", "ok": east.get("success")})
            print("EAST05", hops[-1], flush=True)
            if not east.get("success"):
                retry = take_door(env, assist, total, "RIGHT", expect=0x06)
                hops.append({"hop": "0x05_east_retry", "dest": f"0x{retry['dest']:02x}", "ok": retry["ok"]})
                print("EAST05B", hops[-1], flush=True)
            if read_snapshot(env.get_ram()).screen != 0x06:
                rec = {
                    "ok": False,
                    "failed_room": "0x05",
                    "reason": "east_not_0x06",
                    "hops": hops,
                    "now": w.dump_live(read_snapshot(env.get_ram()), env.get_ram()),
                    "pokes": False,
                    "status_claim": None,
                    "checkpoints": checkpoints,
                }
                w.write_dump("l5_whistle_to_24", rec)
                return rec

        # Prefer shorter 0x06 south if it is a live/key door.
        reached_25 = False
        if read_snapshot(env.get_ram()).screen == 0x06:
            south = try_06_south(env, assist, total)
            hops.append(
                {
                    "hop": "0x06_south",
                    "dest": f"0x{south['dest']:02x}",
                    "ok": south["ok"],
                    "key_spent": south.get("key_spent"),
                }
            )
            print("SOUTH06", hops[-1], flush=True)
            if read_snapshot(env.get_ram()).screen == 0x16:
                reached_25 = walk_16_toward_24(env, assist, total, hops)
            if not reached_25 and read_snapshot(env.get_ram()).screen == 0x06:
                reached_25 = long_path_to_24(env, assist, total, hops)
            elif not reached_25 and read_snapshot(env.get_ram()).screen not in (0x25, 0x24, 0x26):
                # Came back or elsewhere; try long path only from 0x06.
                if read_snapshot(env.get_ram()).screen != 0x06:
                    rec = {
                        "ok": False,
                        "failed_room": f"0x{read_snapshot(env.get_ram()).screen:02x}",
                        "reason": "short_path_dead_end",
                        "hops": hops,
                        "now": w.dump_live(read_snapshot(env.get_ram()), env.get_ram()),
                        "pokes": False,
                        "status_claim": None,
                        "checkpoints": checkpoints,
                    }
                    w.write_dump("l5_whistle_to_24", rec)
                    return rec
                reached_25 = long_path_to_24(env, assist, total, hops)

        snap = read_snapshot(env.get_ram())
        print("PRE24", f"0x{snap.screen:02x}", [snap.link_x, snap.link_y], "whistle", int(read_u8(env.get_ram(), ADDR_WHISTLE)), flush=True)
        if snap.screen not in (0x24, 0x25):
            rec = {
                "ok": False,
                "failed_room": f"0x{snap.screen:02x}",
                "reason": "did_not_reach_0x25_or_0x24",
                "hops": hops,
                "now": w.dump_live(snap, env.get_ram()),
                "pokes": False,
                "status_claim": None,
                "checkpoints": checkpoints,
                "whistle_0x065C": int(read_u8(env.get_ram(), ADDR_WHISTLE)),
            }
            w.write_dump("l5_whistle_to_24", rec)
            w.shot(env, assist, total, "l5_whistle_to_24")
            return rec

        boss = digdogger_tf(env, assist, total)
        rec = {
            "ok": bool(boss.get("ok")),
            "exit_dest": f"0x{floor_room:02x}",
            "whistle_0x065C": int(read_u8(env.get_ram(), ADDR_WHISTLE)),
            "tf_l5_bit": boss.get("tf_l5_bit"),
            "digdogger": {k: boss[k] for k in boss if k not in ("final", "at24", "after_heart")},
            "hops": hops,
            "checkpoints": checkpoints + (["Level5Triforce"] if boss.get("ok") else []),
            "pokes": False,
            "status_claim": None,
        }
        w.write_dump("l5_whistle_floor_tf", rec)
        print(
            "DONE",
            "exit",
            rec["exit_dest"],
            "whistle",
            rec["whistle_0x065C"],
            "tf",
            rec["tf_l5_bit"],
            "ok",
            rec["ok"],
            flush=True,
        )
        return rec
    finally:
        env.close()


if __name__ == "__main__":
    r = main()
    print(
        "OK",
        r.get("ok"),
        "EXIT",
        r.get("exit_dest"),
        "WHISTLE",
        r.get("whistle_0x065C"),
        "TF",
        r.get("tf_l5_bit"),
        "FAILED",
        r.get("failed_room"),
        "CKPT",
        r.get("checkpoints"),
    )
