"""Level5Whistle27 → 0x26 → 0x25 → 0x24. Whistle-shrink Digdogger, sword, heart, north TF 0x10."""
from __future__ import annotations

import importlib.util
from pathlib import Path

from retro_harness.nes import nes_action, nes_idle_action
from retro_harness.segment_runner import configure_headless
from zelda_i.dungeon_ops import idle, push_dir
from zelda_i.level5_path import walk_west_from_25, walk_west_from_26, walk_west_from_27
from zelda_i.paths import RECORDINGS_DIR
from zelda_i.ram import ADDR_TRIFORCE, ADDR_WHISTLE, PLAY_MODE, read_snapshot, read_u8

HERE = Path(__file__).resolve()
_spec = importlib.util.spec_from_file_location("l5whistle", HERE.parent / "_probe_l5_whistle_path.py")
w = importlib.util.module_from_spec(_spec)
assert _spec.loader is not None
_spec.loader.exec_module(w)

TF_BIT = 0x10


def wait_play(env, assist, total, room=None, max_f=280):
    for _ in range(max_f):
        snap = read_snapshot(env.get_ram())
        room_ok = room is None or snap.screen == room
        if snap.mode == PLAY_MODE and room_ok and not snap.transitioning:
            idle(env, assist, total, 12)
            return True
        w.step(env, assist, total, nes_idle_action())
    return False


def go_west(env, assist, total, fn, expect):
    rec = fn(env, assist, total)
    wait_play(env, assist, total, expect, max_f=280)
    snap = read_snapshot(env.get_ram())
    if snap.screen != expect:
        w.walk_axis(env, assist, total, "y", 141, max_f=300)
        w.walk_axis(env, assist, total, "x", 32, max_f=400)
        push_dir(env, assist, total, "LEFT", frames=240)
        idle(env, assist, total, 16)
        wait_play(env, assist, total, expect, max_f=240)
        snap = read_snapshot(env.get_ram())
    return {
        "ok": snap.screen == expect and snap.mode == PLAY_MODE,
        "dest": snap.screen,
        "xy": [snap.link_x, snap.link_y],
        "mode": snap.mode,
        "raw_success": rec.get("success"),
    }


def main() -> dict:
    configure_headless()
    RECORDINGS_DIR.mkdir(parents=True, exist_ok=True)
    env, assist, _ = w.open_env("Level5Whistle27")
    total = [1]
    hops = []
    try:
        idle(env, assist, total, 12)
        snap = read_snapshot(env.get_ram())
        whistle = int(read_u8(env.get_ram(), ADDR_WHISTLE))
        print("START", f"0x{snap.screen:02x}", [snap.link_x, snap.link_y], "whistle", whistle, "keys", snap.keys, flush=True)
        if whistle < 1:
            return {"ok": False, "reason": "whistle_lost", "pokes": False, "status_claim": None}

        if snap.screen == 0x27:
            h = go_west(env, assist, total, walk_west_from_27, 0x26)
            hops.append({"hop": "0x27_west", **h, "dest": f"0x{h['dest']:02x}"})
            print("27WEST", hops[-1], flush=True)
        if read_snapshot(env.get_ram()).screen == 0x26:
            h = go_west(env, assist, total, walk_west_from_26, 0x25)
            hops.append({"hop": "0x26_west", **h, "dest": f"0x{h['dest']:02x}"})
            print("26WEST", hops[-1], flush=True)
        if read_snapshot(env.get_ram()).screen == 0x25:
            h = go_west(env, assist, total, walk_west_from_25, 0x24)
            hops.append({"hop": "0x25_west", **h, "dest": f"0x{h['dest']:02x}"})
            print("25WEST", hops[-1], flush=True)

        snap = read_snapshot(env.get_ram())
        wait_play(env, assist, total, 0x24, max_f=200)
        snap = read_snapshot(env.get_ram())
        print("AT24?", f"0x{snap.screen:02x}", "mode", snap.mode, [snap.link_x, snap.link_y], flush=True)
        if snap.screen != 0x24 or snap.mode != PLAY_MODE:
            rec = {"ok": False, "failed_room": f"0x{snap.screen:02x}", "reason": "not_in_0x24", "hops": hops, "now": w.dump_live(snap, env.get_ram()), "pokes": False, "status_claim": None}
            w.write_dump("l5_24_whistle_boss", rec)
            return rec

        at24 = w.dump_live(snap, env.get_ram())
        w.shot(env, assist, total, "l5_24_enter")
        print("AT24 objs", [(o["type_hex"], o["type_name"], o["hp"]) for o in at24.get("objects") or []], flush=True)

        menu = w.select_whistle_menu(env, assist, total)
        print("MENU", menu, flush=True)
        for _ in range(6):
            w.step(env, assist, total, nes_action("B"))
            idle(env, assist, total, 60)
        idle(env, assist, total, 90)
        after_b = w.dump_live(read_snapshot(env.get_ram()), env.get_ram())
        print("WHISTLE_B objs", [(o["type_hex"], o["type_name"], o["hp"]) for o in after_b.get("objects") or []], flush=True)

        fight = None
        bosses = w.live_boss(read_snapshot(env.get_ram()))
        if bosses:
            fight = w.fight_type(env, assist, total, 0x24, 0x38, expected=len(bosses))
            idle(env, assist, total, 20)
            print("BOSS", fight.get("ok"), "end", fight.get("end_n"), "frames", fight.get("frames"), flush=True)
        snap = read_snapshot(env.get_ram())
        leftovers = [o for o in snap.objects if 1 <= o.slot <= 12 and o.type_id not in (0, 0xFF, 0x55, 0x4E, 0x40, 0x1A) and o.hp > 0]
        extra = None
        if leftovers:
            extra = w.fight_type(env, assist, total, int(snap.screen), leftovers[0].type_id, expected=len(leftovers))
            idle(env, assist, total, 16)
            print("EXTRA", extra.get("ok"), extra.get("end_n"), flush=True)

        for tx, ty in ((120, 141), (144, 141), (96, 141), (120, 125), (120, 157), (160, 141), (80, 141), (176, 141), (64, 141)):
            w.walk_axis(env, assist, total, "y", ty, max_f=200)
            w.walk_axis(env, assist, total, "x", tx, max_f=200)
            idle(env, assist, total, 8)
        after_heart = w.dump_live(read_snapshot(env.get_ram()), env.get_ram())
        w.shot(env, assist, total, "l5_24_after_heart")

        room0 = read_snapshot(env.get_ram()).screen
        w.walk_axis(env, assist, total, "x", 120, max_f=300)
        w.walk_axis(env, assist, total, "y", 93, max_f=400)
        push_dir(env, assist, total, "UP", frames=260)
        idle(env, assist, total, 16)
        wait_play(env, assist, total, max_f=240)
        snap = read_snapshot(env.get_ram())
        north = {"dest": f"0x{snap.screen:02x}", "ok": snap.screen == 0x14, "xy": [snap.link_x, snap.link_y]}
        print("NORTH24", north, flush=True)

        tf0 = int(read_u8(env.get_ram(), ADDR_TRIFORCE))
        tf_walk = None
        if snap.screen == 0x14 or snap.room_item_id == 0x1B:
            tf_walk = w.hunt_item(env, assist, total, ADDR_TRIFORCE)
            idle(env, assist, total, 30)
        ram = env.get_ram()
        snap = read_snapshot(ram)
        tf1 = int(read_u8(ram, ADDR_TRIFORCE))
        final = w.dump_live(snap, ram)
        png = w.shot(env, assist, total, "l5_14_triforce")
        rec = {
            "ok": bool(tf1 & TF_BIT),
            "hops": hops,
            "at24_objs": [(o["type_hex"], o["type_name"], o["hp"]) for o in at24.get("objects") or []],
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
                "Level5Whistle27",
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
        print("DONE", rec["ok"], "tf", rec["tf_l5_bit"], "room", rec["room"], "tf_hex", hex(tf1), flush=True)
        return rec
    finally:
        env.close()


if __name__ == "__main__":
    r = main()
    print("OK", r.get("ok"), "TF", r.get("tf_l5_bit"), "ROOM", r.get("room"), "FAILED", r.get("failed_room"), "HOPS", r.get("hops"))
