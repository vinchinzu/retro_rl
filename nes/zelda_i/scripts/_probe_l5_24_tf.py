"""From Level5Whistle24: walk off east door, one recorder song, sword small Digdoggers, TF."""
from __future__ import annotations

from retro_harness.env import make_env, reset_obs
from retro_harness.nes import nes_action, nes_idle_action
from retro_harness.segment_runner import configure_headless, write_json_report
from zelda_i.assist import UnlimitedHealthAssist
from zelda_i.dungeon_ops import idle, push_dir
from zelda_i.level5_path import walk_axis
from zelda_i.paths import GAME, GAME_DIR, RECORDINGS_DIR
from zelda_i.ram import ADDR_SELECTED_ITEM, ADDR_TRIFORCE, ADDR_WHISTLE, PLAY_MODE, read_snapshot, read_u8
from zelda_i.scripts._probe_l5_whistle_path import (
    dump_and_save_room,
    dump_live,
    fight_type,
    hunt_item,
    select_whistle_menu,
    shot,
    step,
    wait_play,
)

STATE = "Level5Whistle24"


def objs(snap):
    return [
        {"t": f"0x{o.type_id:02x}", "hp": o.hp, "xy": [o.x, o.y]}
        for o in snap.objects
        if 1 <= o.slot <= 12 and o.type_id not in (0, 0xFF)
    ]


def play_recorder(env, assist, total) -> list:
    samples = []
    # Hold B a few frames, then wait out the song.
    for _ in range(8):
        step(env, assist, total, nes_action("B"))
    for i in range(12):
        idle(env, assist, total, 30)
        snap = read_snapshot(env.get_ram())
        rec = {
            "f": (i + 1) * 30,
            "mode": snap.mode,
            "xy": [snap.link_x, snap.link_y],
            "sel": int(read_u8(env.get_ram(), ADDR_SELECTED_ITEM)),
            "objs": objs(snap),
        }
        samples.append(rec)
        print("SONG", rec, flush=True)
        types = {o["t"] for o in rec["objs"]}
        if "0x18" in types or ( "0x38" not in types and any(o["t"] not in ("0x55", "0x40", "0x4E") for o in rec["objs"])):
            break
    return samples


def main() -> dict:
    configure_headless()
    RECORDINGS_DIR.mkdir(parents=True, exist_ok=True)
    env = make_env(GAME, STATE, GAME_DIR, render_mode="rgb_array")
    assist = UnlimitedHealthAssist(enabled=True)
    total = [1]
    try:
        reset_obs(env)
        env.step(nes_idle_action())
        assist.apply_env(env, frame=0)
        idle(env, assist, total, 16)
        start = dump_live(read_snapshot(env.get_ram()), env.get_ram())
        print("START24", start.get("room_hex"), [start.get("x"), start.get("y")], "sel", int(read_u8(env.get_ram(), ADDR_SELECTED_ITEM)), "whistle", start.get("whistle_0x065C"), "objs", objs(read_snapshot(env.get_ram())), flush=True)

        # Off the east door into the room before using the item.
        walk_axis(env, assist, total, "x", 176, max_f=300)
        walk_axis(env, assist, total, "y", 141, max_f=200)
        walk_axis(env, assist, total, "x", 120, max_f=300)
        idle(env, assist, total, 12)
        mid = dump_live(read_snapshot(env.get_ram()), env.get_ram())
        print("MID", [mid.get("x"), mid.get("y")], objs(read_snapshot(env.get_ram())), flush=True)
        shot(env, assist, total, "l5_24_before_song")

        menu = select_whistle_menu(env, assist, total)
        print("MENU", menu, "sel", int(read_u8(env.get_ram(), ADDR_SELECTED_ITEM)), flush=True)
        idle(env, assist, total, 20)
        samples = play_recorder(env, assist, total)
        after = dump_live(read_snapshot(env.get_ram()), env.get_ram())
        print("AFTER_SONG", objs(read_snapshot(env.get_ram())), flush=True)
        shot(env, assist, total, "l5_24_after_song")

        # If still big, try one more song from a different stand.
        snap = read_snapshot(env.get_ram())
        big = [o for o in snap.objects if 1 <= o.slot <= 12 and o.type_id == 0x38 and o.hp > 0]
        if big:
            walk_axis(env, assist, total, "x", 80, max_f=300)
            walk_axis(env, assist, total, "y", 157, max_f=200)
            idle(env, assist, total, 8)
            samples2 = play_recorder(env, assist, total)
            samples.extend(samples2)
            print("AFTER_SONG2", objs(read_snapshot(env.get_ram())), flush=True)
            shot(env, assist, total, "l5_24_after_song2")

        snap = read_snapshot(env.get_ram())
        fights = []
        for type_id in (0x18, 0x38, 0x1A):
            live = [o for o in snap.objects if 1 <= o.slot <= 12 and o.type_id == type_id and o.hp > 0]
            if not live:
                continue
            if type_id == 0x38 and len(live) == 1 and live[0].hp >= 200:
                continue  # still big; sword will not work
            f = fight_type(env, assist, total, 0x24, type_id, expected=len(live))
            fights.append({"type": f"0x{type_id:02x}", **{k: f[k] for k in f if k not in ("progress", "controller")}})
            print("FIGHT", fights[-1], flush=True)
            idle(env, assist, total, 16)
            snap = read_snapshot(env.get_ram())

        leftovers = [
            o
            for o in read_snapshot(env.get_ram()).objects
            if 1 <= o.slot <= 12 and o.type_id not in (0, 0xFF, 0x40, 0x4E, 0x55, 0x68) and o.hp > 0
        ]
        if leftovers:
            extra = fight_type(env, assist, total, 0x24, leftovers[0].type_id, expected=len(leftovers))
            fights.append({"type": f"0x{leftovers[0].type_id:02x}_extra", "ok": extra.get("ok"), "end_n": extra.get("end_n")})
            idle(env, assist, total, 12)

        # Heart 0x1A then north shutter to 0x14.
        walk_axis(env, assist, total, "y", 141, max_f=300)
        walk_axis(env, assist, total, "x", 120, max_f=300)
        idle(env, assist, total, 16)
        after_heart = dump_live(read_snapshot(env.get_ram()), env.get_ram())
        room0 = read_snapshot(env.get_ram()).screen
        walk_axis(env, assist, total, "x", 120, max_f=300)
        walk_axis(env, assist, total, "y", 93, max_f=400)
        push_dir(env, assist, total, "UP", frames=260)
        idle(env, assist, total, 16)
        wait_play(env, assist, total, max_f=240)
        snap = read_snapshot(env.get_ram())
        tf_dump = None
        if snap.screen != room0:
            tf_dump = dump_and_save_room(
                env, assist, total, f"l5_{snap.screen:02x}_triforce", "Level5Triforce", STATE, "0x24 north after Digdogger"
            )
        tf0 = int(read_u8(env.get_ram(), ADDR_TRIFORCE))
        tf_walk = None
        snap = read_snapshot(env.get_ram())
        if snap.room_item_id == 0x1B or snap.screen == 0x14:
            tf_walk = hunt_item(env, assist, total, ADDR_TRIFORCE)
            idle(env, assist, total, 24)
        ram = env.get_ram()
        snap = read_snapshot(ram)
        final = dump_live(snap, ram)
        png = shot(env, assist, total, "l5_24_tf_final")
        rec = {
            "ok": bool(int(final.get("triforce_0x0671") or 0) & 0x10),
            "start": start,
            "mid": {"xy": [mid.get("x"), mid.get("y")]},
            "menu": menu,
            "samples": samples,
            "after_song": after,
            "fights": fights,
            "after_heart": after_heart,
            "tf": tf_dump,
            "tf_walk": None if tf_walk is None else {k: v for k, v in tf_walk.items() if k != "hits"},
            "tf_in": tf0,
            "tf_out": int(final.get("triforce_0x0671") or 0),
            "tf_l5": bool(int(final.get("triforce_0x0671") or 0) & 0x10),
            "final": final,
            "screenshot": png,
            "pokes": False,
            "status_claim": None,
            "l6_l8": False,
            "whistle_0x065C": int(read_u8(ram, ADDR_WHISTLE)),
        }
        write_json_report(RECORDINGS_DIR / "l5_24_tf.json", rec)
        print("FINAL", rec["ok"], "tf", hex(rec["tf_out"]), "room", final.get("room_hex"), flush=True)
        return rec
    finally:
        env.close()


if __name__ == "__main__":
    r = main()
    print("OK", r.get("ok"), "tf", r.get("tf_l5"), flush=True)
