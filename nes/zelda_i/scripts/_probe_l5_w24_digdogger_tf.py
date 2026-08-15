"""From Level5Whistle24: one Recorder song -> small Digdogger 0x18, sword, HC, TF 0x14.

Honest. No door/key pokes. Not a Clean STATUS claim.
"""
from __future__ import annotations

import importlib.util
from pathlib import Path

from retro_harness.env import state_path, write_state_bytes
from retro_harness.nes import nes_action
from retro_harness.segment_runner import configure_headless
from zelda_i.dungeon_ops import idle, push_dir
from zelda_i.dungeon_trace import write_state_provenance
from zelda_i.level5_path import select_b_item_menu, walk_axis
from zelda_i.paths import GAME, GAME_DIR, RECORDINGS_DIR
from zelda_i.ram import ADDR_HEALTH, ADDR_TRIFORCE, ADDR_WHISTLE, PLAY_MODE, read_snapshot, read_u8

HERE = Path(__file__).resolve()
_spec = importlib.util.spec_from_file_location("l5whistle", HERE.parent / "_probe_l5_whistle_path.py")
w = importlib.util.module_from_spec(_spec)
assert _spec.loader is not None
_spec.loader.exec_module(w)

TF_BIT = 0x10
BIG = 0x38
SMALL = 0x18
HC = 0x1A
SKIP = {0, 0xFF, 0x55, 0x4E, 0x40, 0x1A}


def dump(env):
    return w.dump_live(read_snapshot(env.get_ram()), env.get_ram())


def live_of(env, *types):
    s = read_snapshot(env.get_ram())
    return [o for o in s.objects if 1 <= o.slot <= 12 and o.type_id in types and o.hp > 0]


def leftovers(env):
    s = read_snapshot(env.get_ram())
    return [o for o in s.objects if 1 <= o.slot <= 12 and o.type_id not in SKIP and o.hp > 0]


def play_whistle(env, assist, total):
    menu = select_b_item_menu(env, assist, total, 5)
    idle(env, assist, total, 16)
    walk_axis(env, assist, total, "y", 141, max_f=300)
    walk_axis(env, assist, total, "x", 120, max_f=400)
    idle(env, assist, total, 12)
    for _ in range(16):
        w.step(env, assist, total, nes_action("B"))
    # Full recorder melody; Digdogger 0x38 -> 0x18 around 180f.
    for _ in range(8):
        idle(env, assist, total, 30)
        if live_of(env, SMALL) and not live_of(env, BIG):
            break
    idle(env, assist, total, 20)
    return menu


def take_north_14(env, assist, total):
    room0 = read_snapshot(env.get_ram()).screen
    walk_axis(env, assist, total, "y", 141, max_f=300)
    walk_axis(env, assist, total, "x", 120, max_f=300)
    walk_axis(env, assist, total, "y", 93, max_f=300)
    walk_axis(env, assist, total, "x", 120, max_f=200)
    push_dir(env, assist, total, "UP", frames=300)
    idle(env, assist, total, 16)
    for _ in range(260):
        s = read_snapshot(env.get_ram())
        if s.mode == PLAY_MODE and s.screen == 0x14:
            break
        w.step(env, assist, total, nes_action("UP"))
    idle(env, assist, total, 16)
    s = read_snapshot(env.get_ram())
    return {
        "from": f"0x{room0:02x}",
        "dest": f"0x{s.screen:02x}",
        "mode": s.mode,
        "xy": [s.link_x, s.link_y],
        "ok": s.screen == 0x14 and s.mode == PLAY_MODE,
        "doors": int(s.cur_opened_doors),
    }


def main():
    configure_headless()
    RECORDINGS_DIR.mkdir(parents=True, exist_ok=True)
    env, assist, _ = w.open_env("Level5Whistle24")
    total = [1]
    try:
        idle(env, assist, total, 16)
        start = dump(env)
        health0 = int(read_u8(env.get_ram(), ADDR_HEALTH))
        tf0 = int(read_u8(env.get_ram(), ADDR_TRIFORCE))
        print(
            "START",
            start.get("room_hex"),
            "whistle",
            start.get("whistle_0x065C"),
            "tf",
            hex(tf0),
            "boss",
            [(o.type_id, o.hp) for o in live_of(env, BIG, SMALL)],
            flush=True,
        )
        w.shot(env, assist, total, "l5_w24_start")
        if start.get("room") != 0x24 or int(start.get("whistle_0x065C") or 0) < 1:
            rec = {"ok": False, "reason": "not_24_with_whistle", "start": start, "pokes": False}
            w.write_dump("l5_24_whistle_boss", rec)
            return rec

        menu = play_whistle(env, assist, total)
        after_b = dump(env)
        small = live_of(env, SMALL)
        big = live_of(env, BIG)
        print(
            "WHISTLE",
            menu,
            "big",
            len(big),
            "small",
            [(o.hp, o.x, o.y) for o in small],
            flush=True,
        )
        w.shot(env, assist, total, "l5_w24_after_whistle")

        fight = None
        if small:
            fight = w.fight_type(env, assist, total, 0x24, SMALL, expected=len(small))
            idle(env, assist, total, 16)
            print("FIGHT18", {k: fight[k] for k in fight if k != "controller"}, flush=True)
        elif big:
            # Song missed; try once more then fight whatever remains.
            menu = play_whistle(env, assist, total)
            small = live_of(env, SMALL)
            if small:
                fight = w.fight_type(env, assist, total, 0x24, SMALL, expected=len(small))
            else:
                fight = w.fight_type(env, assist, total, 0x24, BIG, expected=len(big))
            idle(env, assist, total, 16)
            print("FIGHT_RETRY", None if fight is None else {k: fight[k] for k in fight if k != "controller"}, flush=True)

        extra = None
        left = leftovers(env)
        if left:
            extra = w.fight_type(env, assist, total, 0x24, left[0].type_id, expected=len(left))
            idle(env, assist, total, 12)
            print("EXTRA", {k: extra[k] for k in extra if k != "controller"}, flush=True)

        health1 = int(read_u8(env.get_ram(), ADDR_HEALTH))
        for tx, ty in ((120, 141), (144, 141), (96, 141), (160, 141), (80, 141), (120, 125), (120, 157), (224, 141), (120, 109)):
            walk_axis(env, assist, total, "y", ty, max_f=220)
            walk_axis(env, assist, total, "x", tx, max_f=220)
            idle(env, assist, total, 8)
            health1 = int(read_u8(env.get_ram(), ADDR_HEALTH))
            if (health1 >> 4) > (health0 >> 4):
                break
        after_heart = dump(env)
        heart = {
            "health_in": health0,
            "health_out": health1,
            "containers_in": health0 >> 4,
            "containers_out": health1 >> 4,
            "got_container": (health1 >> 4) > (health0 >> 4),
            "item": after_heart.get("room_item_id"),
            "boss_left": len(live_of(env, BIG, SMALL)),
            "all_dead": after_heart.get("room_all_dead"),
            "doors": after_heart.get("doors"),
        }
        print("HEART", heart, flush=True)
        w.shot(env, assist, total, "l5_w24_after_heart")

        north = take_north_14(env, assist, total)
        print("NORTH", north, flush=True)
        tf_walk = None
        snap = read_snapshot(env.get_ram())
        if snap.screen == 0x14 or snap.room_item_id == 0x1B:
            tf_walk = w.hunt_item(env, assist, total, ADDR_TRIFORCE)
            idle(env, assist, total, 20)
        tf1 = int(read_u8(env.get_ram(), ADDR_TRIFORCE))
        if not (tf1 & TF_BIT) and read_snapshot(env.get_ram()).screen == 0x14:
            for tx, ty in ((120, 141), (120, 125), (120, 157), (96, 141), (144, 141), (120, 109)):
                walk_axis(env, assist, total, "y", ty, max_f=200)
                walk_axis(env, assist, total, "x", tx, max_f=200)
                idle(env, assist, total, 12)
                tf1 = int(read_u8(env.get_ram(), ADDR_TRIFORCE))
                if tf1 & TF_BIT:
                    break
        idle(env, assist, total, 16)
        final = dump(env)
        tf1 = int(read_u8(env.get_ram(), ADDR_TRIFORCE))
        tf_ok = bool(tf1 & TF_BIT)
        png = w.shot(env, assist, total, "l5_14_triforce")
        ckpt = None
        if tf_ok:
            write_state_bytes(state_path(GAME_DIR, GAME, "Level5Complete"), env.em.get_state())
            write_state_provenance(
                GAME_DIR / "custom_integrations" / GAME / "Level5Complete.state",
                source_state_path=GAME_DIR / "custom_integrations" / GAME / "Level5Whistle24.state",
                request={
                    "segment": "Level5Complete",
                    "via": "Level5Whistle24 one Recorder song, sword 0x18, heart, north 0x14 TF 0x10",
                    "key_poke": False,
                    "door_poke": False,
                    "bomb_count_poke": False,
                    "selected_item_poke": False,
                },
                selected_trial={
                    "success": True,
                    "room": int(final.get("room") or 0),
                    "triforce_0x0671": tf1,
                    "tf_l5_bit": True,
                    "whistle_0x065C": 1,
                    "heart": heart,
                },
                natural_entry=False,
            )
            ckpt = "Level5Complete"
            print("SAVED Level5Complete", flush=True)
        rec = {
            "ok": tf_ok,
            "pokes": False,
            "status_claim": None,
            "start": start,
            "menu": menu,
            "after_whistle_room": after_b.get("room_hex"),
            "fight": None if fight is None else {k: fight[k] for k in fight if k != "controller"},
            "extra": None if extra is None else {k: extra[k] for k in extra if k != "controller"},
            "heart": heart,
            "north": north,
            "tf_walk": None if tf_walk is None else {k: v for k, v in tf_walk.items() if k != "hits"},
            "tf_in": tf0,
            "tf_out": tf1,
            "tf_l5_bit": tf_ok,
            "final": final,
            "checkpoint": ckpt,
            "screenshot": png,
            "whistle_0x065C": int(read_u8(env.get_ram(), ADDR_WHISTLE)),
            "reason": None if tf_ok else ("north_not_0x14" if not north.get("ok") else "tf_bit_0x10_not_set"),
        }
        w.write_dump("l5_24_whistle_boss", rec)
        w.write_dump("l5_w24_digdogger_tf", rec)
        print("FINAL", "tf", hex(tf1), "bit", tf_ok, "room", final.get("room_hex"), "heart", heart, flush=True)
        return rec
    finally:
        env.close()


if __name__ == "__main__":
    r = main()
    print("OK", r.get("ok"))
    print("TF", r.get("tf_l5_bit"), hex(r.get("tf_out") or 0))
    print("HEART", r.get("heart"))
    print("NORTH", r.get("north"))
    print("FIGHT", r.get("fight"))
    print("CKPT", r.get("checkpoint"))
    print("FAILED", r.get("reason"))
    print("status_claim", None)
