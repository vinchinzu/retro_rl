"""From Level5WhistleFloor (0x05, whistle=1): skip 0x16 dead-end, stairs 0x06→0x07→0x64, live graph to 0x24, Digdogger, TF 0x10."""
from __future__ import annotations

import importlib.util
from pathlib import Path

from retro_harness.nes import nes_action, nes_idle_action
from retro_harness.segment_runner import configure_headless
from zelda_i.dungeon_ops import idle, push_dir
from zelda_i.level5_path import (
    cellar_other_mouth,
    take_center_stairs_06,
    walk_east_from_05,
    walk_west_from_25,
    walk_west_from_26,
    walk_west_from_27,
)
from zelda_i.level9_stairs import dest_report, on_stair_tile, walk_to_step
from zelda_i.paths import RECORDINGS_DIR
from zelda_i.ram import ADDR_TRIFORCE, ADDR_WHISTLE, PLAY_MODE, read_snapshot, read_u8

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
    ),
}


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
        rec = {"steps": list(steps), "dest": f"0x{snap.screen:02x}", "xy": [snap.link_x, snap.link_y], "mode": snap.mode}
        tried.append(rec)
        print("DOOR", direction, rec, flush=True)
        if snap.screen != room0 and snap.mode == PLAY_MODE:
            ok = expect is None or snap.screen == expect
            return {"ok": ok, "dest": snap.screen, "tried": tried}
    snap = read_snapshot(env.get_ram())
    return {"ok": snap.screen != room0, "dest": snap.screen, "tried": tried}


def take_06_stairs_hard(env, assist, total):
    """South-gap canned path, then walk-through the 0x73 tile, then (208,96)."""
    canned = take_center_stairs_06(env, assist, total)
    snap = read_snapshot(env.get_ram())
    print("STAIRS06_CANNED", canned.get("success"), "dest", f"0x{canned.get('dest'):02x}", "mode", canned.get("mode"), canned.get("xy"), "tile", snap.colliding_tile, "stair", on_stair_tile(snap), flush=True)
    if canned.get("success") or snap.mode in CELLAR or snap.screen == 0x07:
        return {"ok": True, "via": "canned", "dest": snap.screen, "mode": snap.mode, "canned": {k: canned[k] for k in canned if k != "log"}}

    approaches = (
        ((80, 189), (80, 141), (120, 141)),
        ((96, 189), (96, 149), (120, 141)),
        ((64, 189), (64, 141), (120, 141)),
        ((120, 189), (120, 141)),
        ((120, 173), (120, 141)),
        ((80, 93), (80, 141), (120, 141)),
        ((160, 189), (160, 141), (120, 141)),
        ((208, 141), (208, 96)),
        ((192, 96), (208, 96)),
        ((120, 125), (120, 141)),
        ((104, 141), (120, 141)),
        ((136, 141), (120, 141)),
    )
    log = []
    for stands in approaches:
        if read_snapshot(env.get_ram()).screen != 0x06:
            break
        for tx, ty in stands:
            for _ in range(320):
                snap = read_snapshot(env.get_ram())
                if snap.mode in CELLAR or snap.screen == 0x07:
                    return {"ok": True, "via": stands, "dest": snap.screen, "mode": snap.mode, "xy": [snap.link_x, snap.link_y]}
                frame = walk_to_step(snap, tx, ty, y_first=True, tol=2)
                if frame.reason == "walk_arrived":
                    idle(env, assist, total, 20)
                    break
                w.step(env, assist, total, frame.action)
            snap = read_snapshot(env.get_ram())
            rec = {"stand": [tx, ty], "xy": [snap.link_x, snap.link_y], "tile": int(snap.colliding_tile), "stair": bool(on_stair_tile(snap)), "mode": snap.mode}
            log.append(rec)
            print("ST06", rec, flush=True)
            if snap.mode in CELLAR or snap.screen == 0x07:
                return {"ok": True, "via": stands, "dest": snap.screen, "mode": snap.mode, "log": log[-8:]}
            if on_stair_tile(snap) and snap.colliding_tile != 0x24:
                idle(env, assist, total, 24)
                for d in ("UP", "DOWN", "LEFT", "RIGHT"):
                    for _ in range(12):
                        snap = read_snapshot(env.get_ram())
                        if snap.mode in CELLAR or snap.screen == 0x07:
                            return {"ok": True, "via": f"nudge_{d}", "dest": snap.screen, "mode": snap.mode}
                        w.step(env, assist, total, nes_action(d))
                    idle(env, assist, total, 8)
                    if read_snapshot(env.get_ram()).mode in CELLAR:
                        return {"ok": True, "via": f"nudge_{d}", "dest": read_snapshot(env.get_ram()).screen, "mode": 9}
    snap = read_snapshot(env.get_ram())
    return {"ok": False, "dest": snap.screen, "mode": snap.mode, "xy": [snap.link_x, snap.link_y], "tile": int(snap.colliding_tile), "stair": bool(on_stair_tile(snap)), "log": log[-12:]}


def long_path(env, assist, total, hops):
    cellar = cellar_other_mouth(env, assist, total)
    hops.append({"hop": "0x07_other", "dest": f"0x{cellar.get('dest'):02x}", "ok": cellar.get("dest") == 0x64, "side": cellar.get("chose_side")})
    print("CELLAR07", hops[-1], cellar.get("xy"), flush=True)
    snap = read_snapshot(env.get_ram())
    if snap.screen != 0x64:
        # explicit left mouth
        w.walk_axis(env, assist, total, "y", 189, max_f=400)
        w.walk_axis(env, assist, total, "x", 48, max_f=500)
        push_dir(env, assist, total, "UP", frames=280)
        idle(env, assist, total, 16)
        wait_play(env, assist, total, max_f=240)
        hops.append({"hop": "0x07_left", "dest": f"0x{read_snapshot(env.get_ram()).screen:02x}"})
        print("CELLAR07B", hops[-1], flush=True)
    if read_snapshot(env.get_ram()).screen != 0x64:
        return False

    w.dump_and_save_room(env, assist, total, "l5_64_whistle_floor", "Level5Whistle64", "Level5WhistleFloor", "0x07 left mouth")
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
    print("AT24 objs", [(o["type_hex"], o["type_name"], o["hp"]) for o in at24.get("objects") or []], flush=True)
    menu = w.select_whistle_menu(env, assist, total)
    for _ in range(5):
        w.step(env, assist, total, nes_action("B"))
        idle(env, assist, total, 50)
    idle(env, assist, total, 80)
    after_b = w.dump_live(read_snapshot(env.get_ram()), env.get_ram())
    print("WHISTLE_B", menu, "objs", [(o["type_hex"], o["type_name"], o["hp"]) for o in after_b.get("objects") or []], flush=True)
    fight = None
    bosses = w.live_boss(read_snapshot(env.get_ram()))
    if bosses:
        fight = w.fight_type(env, assist, total, 0x24, 0x38, expected=len(bosses))
        idle(env, assist, total, 16)
        print("BOSS", fight.get("ok"), "end", fight.get("end_n"), flush=True)
    snap = read_snapshot(env.get_ram())
    leftovers = [o for o in snap.objects if 1 <= o.slot <= 12 and o.type_id not in (0, 0xFF, 0x55, 0x4E, 0x40, 0x1A) and o.hp > 0]
    extra = None
    if leftovers:
        extra = w.fight_type(env, assist, total, int(snap.screen), leftovers[0].type_id, expected=len(leftovers))
        idle(env, assist, total, 12)
        print("EXTRA", extra.get("ok"), extra.get("end_n"), flush=True)
    for tx, ty in ((120, 141), (144, 141), (96, 141), (120, 125), (120, 157), (160, 141), (80, 141)):
        w.walk_axis(env, assist, total, "y", ty, max_f=200)
        w.walk_axis(env, assist, total, "x", tx, max_f=200)
        idle(env, assist, total, 8)
    after_heart = w.dump_live(read_snapshot(env.get_ram()), env.get_ram())
    w.shot(env, assist, total, "l5_24_after_heart")
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
        "after_whistle_objs": [(o["type_hex"], o["type_name"], o["hp"]) for o in after_b.get("objects") or []],
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
            "Level5WhistleFloor",
            {
                "segment": "Level5Triforce",
                "via": "0x24 whistle Digdogger, heart, north 0x14 TF bit 0x10",
                "key_poke": False,
                "door_poke": False,
                "bomb_count_poke": False,
                "selected_item_poke": False,
            },
            {"success": True, "room": int(snap.screen), "triforce_0x0671": tf1, "tf_l5_bit": True, "whistle_0x065C": 1},
        )
    return rec


def main() -> dict:
    configure_headless()
    RECORDINGS_DIR.mkdir(parents=True, exist_ok=True)
    env, assist, _ = w.open_env("Level5WhistleFloor")
    total = [1]
    hops = []
    checkpoints = ["Level5WhistleFloor"]
    try:
        idle(env, assist, total, 12)
        snap = read_snapshot(env.get_ram())
        whistle = int(read_u8(env.get_ram(), ADDR_WHISTLE))
        start = w.dump_live(snap, env.get_ram())
        print("START", f"0x{snap.screen:02x}", "mode", snap.mode, [snap.link_x, snap.link_y], "whistle", whistle, "doors", start.get("doors"), flush=True)
        if snap.screen != 0x05 or whistle < 1 or snap.mode != PLAY_MODE:
            rec = {"ok": False, "reason": "floor_not_0x05_play", "now": start, "pokes": False, "status_claim": None}
            w.write_dump("l5_whistle_to_24", rec)
            return rec

        east = walk_east_from_05(env, assist, total)
        hops.append({"hop": "0x05_east", "dest": f"0x{east.get('dest'):02x}", "ok": east.get("success")})
        print("EAST05", hops[-1], flush=True)
        if not east.get("success"):
            retry = take_door(env, assist, total, "RIGHT", expect=0x06)
            hops.append({"hop": "0x05_east_retry", "dest": f"0x{retry['dest']:02x}", "ok": retry["ok"]})
        if read_snapshot(env.get_ram()).screen != 0x06:
            rec = {"ok": False, "failed_room": "0x05", "hops": hops, "now": w.dump_live(read_snapshot(env.get_ram()), env.get_ram()), "pokes": False, "status_claim": None}
            w.write_dump("l5_whistle_to_24", rec)
            return rec

        stairs = take_06_stairs_hard(env, assist, total)
        hops.append({"hop": "0x06_stairs", "dest": f"0x{stairs.get('dest'):02x}", "mode": stairs.get("mode"), "ok": stairs.get("ok"), "via": stairs.get("via")})
        print("STAIRS06", hops[-1], flush=True)
        snap = read_snapshot(env.get_ram())
        if not (snap.mode in CELLAR or snap.screen == 0x07):
            rec = {
                "ok": False,
                "failed_room": "0x06",
                "reason": "stairs_not_cellar",
                "stairs": stairs,
                "now": w.dump_live(snap, env.get_ram()),
                "dest": dest_report(snap),
                "hops": hops,
                "pokes": False,
                "status_claim": None,
            }
            w.write_dump("l5_whistle_to_24", rec)
            w.shot(env, assist, total, "l5_06_stairs_fail")
            return rec

        if not long_path(env, assist, total, hops):
            snap = read_snapshot(env.get_ram())
            rec = {
                "ok": False,
                "failed_room": f"0x{snap.screen:02x}",
                "reason": "long_path_broke",
                "hops": hops,
                "now": w.dump_live(snap, env.get_ram()),
                "pokes": False,
                "status_claim": None,
                "whistle_0x065C": int(read_u8(env.get_ram(), ADDR_WHISTLE)),
            }
            w.write_dump("l5_whistle_to_24", rec)
            w.shot(env, assist, total, "l5_whistle_to_24")
            return rec

        boss = digdogger_tf(env, assist, total)
        rec = {
            "ok": bool(boss.get("ok")),
            "exit_dest": "0x05",
            "whistle_0x065C": int(read_u8(env.get_ram(), ADDR_WHISTLE)),
            "tf_l5_bit": boss.get("tf_l5_bit"),
            "digdogger": {k: boss[k] for k in boss if k not in ("final", "at24", "after_heart")},
            "hops": hops,
            "checkpoints": checkpoints + (["Level5Triforce"] if boss.get("ok") else []),
            "pokes": False,
            "status_claim": None,
        }
        w.write_dump("l5_whistle_floor_tf", rec)
        print("DONE", rec["ok"], "tf", rec["tf_l5_bit"], "whistle", rec["whistle_0x065C"], flush=True)
        return rec
    finally:
        env.close()


if __name__ == "__main__":
    r = main()
    print("OK", r.get("ok"), "TF", r.get("tf_l5_bit"), "FAILED", r.get("failed_room"), "HOPS", r.get("hops"), "CKPT", r.get("checkpoints"))
