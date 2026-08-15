"""From Level5Whistle27 walk to 0x24, save enter, play recorder ONCE, sword small Digdoggers, TF."""
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
    fn(env, assist, total)
    wait_play(env, assist, total, expect, max_f=280)
    snap = read_snapshot(env.get_ram())
    if snap.screen != expect:
        w.walk_axis(env, assist, total, "y", 141, max_f=300)
        w.walk_axis(env, assist, total, "x", 32, max_f=400)
        push_dir(env, assist, total, "LEFT", frames=240)
        idle(env, assist, total, 16)
        wait_play(env, assist, total, expect, max_f=240)
    snap = read_snapshot(env.get_ram())
    return snap.screen == expect


def objs(snap):
    return [(f"0x{o.type_id:02x}", o.hp, o.x, o.y) for o in snap.objects if 1 <= o.slot <= 12 and o.type_id not in (0, 0xFF)]


def main() -> dict:
    configure_headless()
    RECORDINGS_DIR.mkdir(parents=True, exist_ok=True)
    env, assist, _ = w.open_env("Level5Whistle27")
    total = [1]
    try:
        idle(env, assist, total, 10)
        go_west(env, assist, total, walk_west_from_27, 0x26)
        go_west(env, assist, total, walk_west_from_26, 0x25)
        go_west(env, assist, total, walk_west_from_25, 0x24)
        wait_play(env, assist, total, 0x24, max_f=200)
        snap = read_snapshot(env.get_ram())
        print("AT24", [snap.link_x, snap.link_y], "objs", objs(snap), "whistle", int(read_u8(env.get_ram(), ADDR_WHISTLE)), flush=True)
        if snap.screen != 0x24:
            return {"ok": False, "reason": "not_24", "room": hex(snap.screen)}
        w.dump_and_save_room(env, assist, total, "l5_24_enter", "Level5Whistle24", "Level5Whistle27", "0x25 west door")

        # Center, select recorder, play ONCE, wait the song out.
        w.walk_axis(env, assist, total, "y", 141, max_f=300)
        w.walk_axis(env, assist, total, "x", 120, max_f=400)
        idle(env, assist, total, 8)
        menu = w.select_whistle_menu(env, assist, total)
        print("MENU", menu, flush=True)
        for _ in range(3):
            w.step(env, assist, total, nes_action("B"))
        log = []
        for i in range(360):
            idle(env, assist, total, 1)
            if i % 30 == 0:
                snap = read_snapshot(env.get_ram())
                rec = {"f": i, "mode": snap.mode, "objs": objs(snap), "xy": [snap.link_x, snap.link_y]}
                log.append(rec)
                print("SONG", rec, flush=True)
                types = {o.type_id for o in snap.objects if 1 <= o.slot <= 12 and o.type_id not in (0, 0xFF, 0x55)}
                if 0x38 not in types and types:
                    break
                if 0x38 in types:
                    big = [o for o in snap.objects if o.type_id == 0x38 and o.hp > 0]
                    if big and big[0].hp < 240:
                        break
        snap = read_snapshot(env.get_ram())
        after = {"objs": objs(snap), "mode": snap.mode, "xy": [snap.link_x, snap.link_y]}
        print("AFTER_SONG", after, flush=True)
        w.shot(env, assist, total, "l5_24_after_whistle")

        fight = None
        extra = None
        bosses = w.live_boss(snap)
        small = [o for o in snap.objects if 1 <= o.slot <= 12 and o.type_id in (0x18, 0x38) and o.hp > 0]
        if bosses and all(o.hp >= 200 for o in bosses):
            # still big — try one more single play
            for _ in range(3):
                w.step(env, assist, total, nes_action("B"))
            idle(env, assist, total, 240)
            snap = read_snapshot(env.get_ram())
            print("AFTER_SONG2", objs(snap), flush=True)
            bosses = w.live_boss(snap)
            small = [o for o in snap.objects if 1 <= o.slot <= 12 and o.type_id in (0x18, 0x38) and o.hp > 0]
        if small:
            tid = small[0].type_id
            fight = w.fight_type(env, assist, total, 0x24, tid, expected=len(small))
            idle(env, assist, total, 16)
            print("FIGHT", tid, fight.get("ok"), fight.get("end_n"), fight.get("frames"), flush=True)
        snap = read_snapshot(env.get_ram())
        leftovers = [o for o in snap.objects if 1 <= o.slot <= 12 and o.type_id not in (0, 0xFF, 0x55, 0x4E, 0x40, 0x1A) and o.hp > 0]
        if leftovers:
            extra = w.fight_type(env, assist, total, 0x24, leftovers[0].type_id, expected=len(leftovers))
            idle(env, assist, total, 12)
            print("EXTRA", extra.get("ok"), extra.get("end_n"), flush=True)

        for tx, ty in ((120, 141), (144, 141), (96, 141), (120, 125), (120, 157), (160, 141), (80, 141)):
            w.walk_axis(env, assist, total, "y", ty, max_f=200)
            w.walk_axis(env, assist, total, "x", tx, max_f=200)
            idle(env, assist, total, 8)
        w.shot(env, assist, total, "l5_24_after_heart")
        w.walk_axis(env, assist, total, "x", 120, max_f=300)
        w.walk_axis(env, assist, total, "y", 93, max_f=400)
        push_dir(env, assist, total, "UP", frames=260)
        idle(env, assist, total, 16)
        wait_play(env, assist, total, max_f=240)
        snap = read_snapshot(env.get_ram())
        print("NORTH", f"0x{snap.screen:02x}", [snap.link_x, snap.link_y], "item", snap.room_item_id, flush=True)
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
            "menu": menu,
            "song_log": log,
            "after_song": after,
            "fight": None if fight is None else {k: fight[k] for k in fight if k != "controller"},
            "extra": None if extra is None else {k: extra[k] for k in extra if k != "controller"},
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
                "Level5Whistle24",
                {
                    "segment": "Level5Triforce",
                    "via": "0x24 recorder once, sword small Digdogger, north 0x14 TF 0x10",
                    "key_poke": False,
                    "door_poke": False,
                    "bomb_count_poke": False,
                    "selected_item_poke": False,
                },
                {"success": True, "room": int(snap.screen), "triforce_0x0671": tf1, "tf_l5_bit": True, "whistle_0x065C": 1},
            )
        print("DONE", rec["ok"], "tf", rec["tf_l5_bit"], "room", rec["room"], hex(tf1), flush=True)
        return rec
    finally:
        env.close()


if __name__ == "__main__":
    r = main()
    print("OK", r.get("ok"), "TF", r.get("tf_l5_bit"), "ROOM", r.get("room"))
