"""Level5Whistle64 → 0x65, bomb EAST 0x66 (live/ROM bomb), north 0x56, graph to 0x24, Digdogger, TF."""
from __future__ import annotations

import importlib.util
from pathlib import Path

from retro_harness.nes import nes_action, nes_idle_action
from retro_harness.segment_runner import configure_headless
from zelda_i.dungeon_ops import idle, push_dir
from zelda_i.level5_path import select_b_item_menu, walk_west_from_25, walk_west_from_26, walk_west_from_27
from zelda_i.paths import RECORDINGS_DIR
from zelda_i.ram import ADDR_TRIFORCE, ADDR_WHISTLE, PLAY_MODE, read_snapshot, read_u8

HERE = Path(__file__).resolve()
_spec = importlib.util.spec_from_file_location("l5whistle", HERE.parent / "_probe_l5_whistle_path.py")
w = importlib.util.module_from_spec(_spec)
assert _spec.loader is not None
_spec.loader.exec_module(w)

TF_BIT = 0x10

DOOR_PATHS = {
    "RIGHT": (
        (("y", 189), ("x", 208), ("y", 141), ("x", 224)),
        (("x", 80), ("y", 93), ("x", 208), ("y", 141), ("x", 224)),
        (("y", 93), ("x", 208), ("y", 141), ("x", 224)),
        (("y", 109), ("x", 208), ("y", 141), ("x", 224)),
        (("y", 141), ("x", 224)),
    ),
    "LEFT": (
        (("y", 189), ("x", 32), ("y", 141)),
        (("x", 208), ("y", 189), ("x", 32), ("y", 141)),
        (("y", 93), ("x", 32), ("y", 141)),
        (("y", 141), ("x", 32)),
    ),
    "UP": (
        (("x", 80), ("y", 93), ("x", 120)),
        (("x", 160), ("y", 93), ("x", 120)),
        (("y", 93), ("x", 120)),
        (("x", 120), ("y", 93)),
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
        if read_snapshot(env.get_ram()).screen != room0:
            break
        for axis, tgt in steps:
            w.walk_axis(env, assist, total, axis, tgt, max_f=400)
        push_dir(env, assist, total, direction, frames=240)
        idle(env, assist, total, 12)
        wait_play(env, assist, total, max_f=200)
        snap = read_snapshot(env.get_ram())
        rec = {"steps": list(steps), "dest": f"0x{snap.screen:02x}", "xy": [snap.link_x, snap.link_y]}
        tried.append(rec)
        print("DOOR", direction, rec, flush=True)
        if snap.screen != room0 and snap.mode == PLAY_MODE:
            return {"ok": expect is None or snap.screen == expect, "dest": snap.screen, "tried": tried}
    snap = read_snapshot(env.get_ram())
    return {"ok": snap.screen != room0, "dest": snap.screen, "tried": tried}


def bomb_east_65(env, assist, total):
    """One bomb at 0x65 east wall. Dest must become 0x66. No poke."""
    approaches = (
        (("y", 189), ("x", 208), ("y", 141), ("x", 224)),
        (("y", 93), ("x", 208), ("y", 141), ("x", 224)),
        (("y", 109), ("x", 208), ("y", 141), ("x", 224)),
    )
    room0 = 0x65
    for steps in approaches:
        if read_snapshot(env.get_ram()).screen != room0:
            break
        for axis, tgt in steps:
            w.walk_axis(env, assist, total, axis, tgt, max_f=400)
        snap = read_snapshot(env.get_ram())
        print("BOMB_STAND", [snap.link_x, snap.link_y], flush=True)
        if abs(snap.link_x - 224) > 16 or abs(snap.link_y - 141) > 16:
            continue
        for _ in range(8):
            w.step(env, assist, total, nes_action("RIGHT"))
        idle(env, assist, total, 6)
        menu = select_b_item_menu(env, assist, total, 1)
        bombs0 = int(read_snapshot(env.get_ram()).bombs)
        w.step(env, assist, total, nes_action("RIGHT", "B"))
        for _ in range(16):
            w.step(env, assist, total, nes_action("LEFT"))
        idle(env, assist, total, 100)
        for _ in range(360):
            snap = read_snapshot(env.get_ram())
            if snap.mode == PLAY_MODE and snap.screen == 0x66:
                break
            w.step(env, assist, total, nes_action("RIGHT"))
        idle(env, assist, total, 20)
        wait_play(env, assist, total, max_f=200)
        snap = read_snapshot(env.get_ram())
        rec = {
            "steps": list(steps),
            "menu": menu,
            "bombs_in": bombs0,
            "bombs_out": int(snap.bombs),
            "dest": f"0x{snap.screen:02x}",
            "xy": [snap.link_x, snap.link_y],
            "ok": snap.screen == 0x66,
        }
        print("BOMBEAST", rec, flush=True)
        if rec["ok"]:
            return rec
    snap = read_snapshot(env.get_ram())
    return {"ok": snap.screen == 0x66, "dest": f"0x{snap.screen:02x}", "xy": [snap.link_x, snap.link_y]}


def digdogger_tf(env, assist, total):
    snap = read_snapshot(env.get_ram())
    if snap.screen != 0x24:
        west = walk_west_from_25(env, assist, total)
        print("25WEST", west.get("success"), "dest", f"0x{west.get('dest'):02x}", flush=True)
        if not west.get("success"):
            return {"ok": False, "reason": "west_25_missed_24"}
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
        "menu": menu,
        "after_whistle_objs": [(o["type_hex"], o["type_name"], o["hp"]) for o in after_b.get("objects") or []],
        "fight": None if fight is None else {k: fight[k] for k in fight if k != "controller"},
        "extra": None if extra is None else {k: extra[k] for k in extra if k != "controller"},
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
            "Level5Whistle64",
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
    env, assist, _ = w.open_env("Level5Whistle64")
    total = [1]
    hops = []
    checkpoints = ["Level5WhistleFloor", "Level5Whistle64"]
    try:
        idle(env, assist, total, 10)
        print("START", f"0x{read_snapshot(env.get_ram()).screen:02x}", [read_snapshot(env.get_ram()).link_x, read_snapshot(env.get_ram()).link_y], flush=True)

        hop = take_door(env, assist, total, "RIGHT", expect=0x65)
        hops.append({"hop": "0x64_east", "dest": f"0x{hop['dest']:02x}", "ok": hop["ok"]})
        print("EAST64", hops[-1], flush=True)
        if not hop["ok"]:
            return {"ok": False, "failed_room": "0x64", "hops": hops, "pokes": False, "status_claim": None}

        w.dump_and_save_room(env, assist, total, "l5_65_whistle", "Level5Whistle65", "Level5Whistle64", "0x64 south-gap east")
        checkpoints.append("Level5Whistle65")

        bomb = bomb_east_65(env, assist, total)
        hops.append({"hop": "0x65_bomb_east", **{k: bomb[k] for k in bomb if k != "menu"}})
        print("BOMB65E", hops[-1], flush=True)
        if not bomb.get("ok"):
            rec = {"ok": False, "failed_room": "0x65", "reason": "bomb_east_not_0x66", "hops": hops, "now": w.dump_live(read_snapshot(env.get_ram()), env.get_ram()), "pokes": False, "status_claim": None}
            w.write_dump("l5_whistle_to_24", rec)
            w.shot(env, assist, total, "l5_65_bomb_east_fail")
            return rec

        w.dump_and_save_room(env, assist, total, "l5_66_whistle", "Level5Whistle66", "Level5Whistle64", "0x65 bomb east")
        checkpoints.append("Level5Whistle66")

        # 0x66 north shutter should be open from earlier clear → 0x56
        north = take_door(env, assist, total, "UP", expect=0x56)
        hops.append({"hop": "0x66_north", "dest": f"0x{north['dest']:02x}", "ok": north["ok"]})
        print("NORTH66", hops[-1], flush=True)
        if not north["ok"]:
            rec = {"ok": False, "failed_room": "0x66", "reason": "north_not_0x56", "hops": hops, "now": w.dump_live(read_snapshot(env.get_ram()), env.get_ram()), "pokes": False, "status_claim": None}
            w.write_dump("l5_whistle_to_24", rec)
            w.shot(env, assist, total, "l5_66_north_fail")
            return rec

        chain = (
            (0x56, "RIGHT", 0x57),
            (0x57, "UP", 0x47),
            (0x47, "UP", 0x37),
            (0x37, "UP", 0x27),
        )
        for src, d, dst in chain:
            snap = read_snapshot(env.get_ram())
            if snap.screen != src:
                hops.append({"hop": f"0x{src:02x}_{d}", "ok": False, "at": f"0x{snap.screen:02x}"})
                rec = {"ok": False, "failed_room": f"0x{snap.screen:02x}", "hops": hops, "now": w.dump_live(snap, env.get_ram()), "pokes": False, "status_claim": None}
                w.write_dump("l5_whistle_to_24", rec)
                return rec
            hop = take_door(env, assist, total, d, expect=dst)
            hops.append({"hop": f"0x{src:02x}_{d}", "dest": f"0x{hop['dest']:02x}", "ok": hop["ok"] and hop["dest"] == dst})
            print("HOP", hops[-1], flush=True)
            if not hops[-1]["ok"]:
                rec = {"ok": False, "failed_room": f"0x{src:02x}", "hops": hops, "now": w.dump_live(read_snapshot(env.get_ram()), env.get_ram()), "pokes": False, "status_claim": None}
                w.write_dump("l5_whistle_to_24", rec)
                w.shot(env, assist, total, "l5_whistle_to_24")
                return rec

        west27 = walk_west_from_27(env, assist, total)
        hops.append({"hop": "0x27_west", "dest": f"0x{west27.get('dest'):02x}", "ok": west27.get("success")})
        print("27WEST", hops[-1], flush=True)
        if not west27.get("success"):
            rec = {"ok": False, "failed_room": "0x27", "hops": hops, "pokes": False, "status_claim": None}
            w.write_dump("l5_whistle_to_24", rec)
            return rec
        west26 = walk_west_from_26(env, assist, total)
        hops.append({"hop": "0x26_west", "dest": f"0x{west26.get('dest'):02x}", "ok": west26.get("success")})
        print("26WEST", hops[-1], flush=True)
        if not west26.get("success"):
            rec = {"ok": False, "failed_room": "0x26", "hops": hops, "pokes": False, "status_claim": None}
            w.write_dump("l5_whistle_to_24", rec)
            return rec

        boss = digdogger_tf(env, assist, total)
        rec = {
            "ok": bool(boss.get("ok")),
            "exit_dest": "0x05",
            "whistle_0x065C": int(read_u8(env.get_ram(), ADDR_WHISTLE)),
            "tf_l5_bit": boss.get("tf_l5_bit"),
            "digdogger": {k: boss[k] for k in boss if k not in ("final",)},
            "hops": hops,
            "checkpoints": checkpoints + (["Level5Triforce"] if boss.get("ok") else []),
            "pokes": False,
            "status_claim": None,
        }
        w.write_dump("l5_whistle_floor_tf", rec)
        print("DONE", rec["ok"], "tf", rec["tf_l5_bit"], flush=True)
        return rec
    finally:
        env.close()


if __name__ == "__main__":
    r = main()
    print("OK", r.get("ok"), "TF", r.get("tf_l5_bit"), "FAILED", r.get("failed_room"), "HOPS", r.get("hops"), "CKPT", r.get("checkpoints"))
