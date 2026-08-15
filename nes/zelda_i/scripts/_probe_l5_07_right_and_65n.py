"""1) Level5Entered07 raw right-mouth -> 0x06. 2) Level5Whistle65 NORTH probe."""
from retro_harness.nes import nes_action
from retro_harness.segment_runner import configure_headless
from zelda_i.dungeon_ops import idle, push_dir
from zelda_i.level9_stairs import dest_report
from zelda_i.paths import RECORDINGS_DIR
from zelda_i.ram import PLAY_MODE, read_snapshot
from zelda_i.scripts._probe_l5_whistle_path import (
    dump_and_save_room,
    dump_live,
    open_env,
    rom_room,
    shot,
    step,
    wait_play,
    write_dump,
)


def raw_axis(env, assist, total, axis, tgt, max_f=500):
    last = None
    stall = 0
    for _ in range(max_f):
        snap = read_snapshot(env.get_ram())
        if axis == "x":
            if abs(snap.link_x - tgt) <= 1:
                return [snap.link_x, snap.link_y]
            step(env, assist, total, nes_action("RIGHT" if snap.link_x < tgt else "LEFT"))
        else:
            if abs(snap.link_y - tgt) <= 1:
                return [snap.link_x, snap.link_y]
            step(env, assist, total, nes_action("DOWN" if snap.link_y < tgt else "UP"))
        snap2 = read_snapshot(env.get_ram())
        pos = (snap2.link_x, snap2.link_y)
        if pos == last:
            stall += 1
            if stall >= 35:
                return [snap2.link_x, snap2.link_y]
        else:
            stall = 0
        last = pos
    snap = read_snapshot(env.get_ram())
    return [snap.link_x, snap.link_y]


def right_mouth():
    env, assist, _ = open_env("Level5Entered07")
    total = [1]
    log = []
    try:
        idle(env, assist, total, 10)
        start = dump_live(read_snapshot(env.get_ram()), env.get_ram())
        print("07START", [start.get("x"), start.get("y")], start.get("mode"), flush=True)
        for axis, tgt in (("y", 165), ("x", 192), ("y", 61), ("x", 176)):
            xy = raw_axis(env, assist, total, axis, tgt)
            snap = read_snapshot(env.get_ram())
            rec = {"axis": axis, "tgt": tgt, "xy": xy, "mode": snap.mode, "room": f"0x{snap.screen:02x}"}
            log.append(rec)
            print("07WALK", rec, flush=True)
            if snap.mode == PLAY_MODE and snap.screen != 0x07:
                break
        push_dir(env, assist, total, "UP", frames=200)
        idle(env, assist, total, 16)
        wait_play(env, assist, total, max_f=280)
        idle(env, assist, total, 16)
        snap = read_snapshot(env.get_ram())
        dest = dump_live(snap, env.get_ram())
        png = shot(env, assist, total, "l5_06_arrive")
        ok = snap.screen == 0x06
        body = {
            "ok": ok,
            "failed_room": None if ok else "0x07",
            "pokes": False,
            "status_claim": None,
            "start": [start.get("x"), start.get("y")],
            "log": log,
            "dump": dest,
            "dest": dest_report(snap),
            "screenshot": png,
            "whistle_0x065C": dest.get("whistle_0x065C"),
            "rom": rom_room(int(snap.screen)),
        }
        write_dump("l5_06_arrive", body)
        print("07DEST", dest.get("room_hex"), dest.get("mode"), [dest.get("x"), dest.get("y")], "ok", ok, flush=True)
        if ok:
            dump_and_save_room(env, assist, total, "l5_06_arrive", "Level5Entered06", "Level5Entered07", "0x07 right mouth y165 raw")
        return body
    finally:
        env.close()


def north65():
    env, assist, _ = open_env("Level5Whistle65")
    total = [1]
    try:
        idle(env, assist, total, 10)
        start = dump_live(read_snapshot(env.get_ram()), env.get_ram())
        print("65START", start.get("room_hex"), [start.get("x"), start.get("y")], "doors", start.get("doors"), "whistle", start.get("whistle_0x065C"), flush=True)
        push_dir(env, assist, total, "UP", frames=240)
        idle(env, assist, total, 16)
        snap = read_snapshot(env.get_ram())
        if snap.screen != 0x65:
            wait_play(env, assist, total, snap.screen, max_f=240)
        idle(env, assist, total, 12)
        snap = read_snapshot(env.get_ram())
        dest = dump_live(snap, env.get_ram())
        png = shot(env, assist, total, "l5_65_north")
        ok = snap.screen != 0x65
        body = {
            "ok": ok,
            "pokes": False,
            "status_claim": None,
            "start": start,
            "dump": dest,
            "screenshot": png,
            "whistle_0x065C": dest.get("whistle_0x065C"),
        }
        write_dump("l5_65_north", body)
        print("65DEST", dest.get("room_hex"), dest.get("mode"), [dest.get("x"), dest.get("y")], "ok", ok, "whistle", dest.get("whistle_0x065C"), flush=True)
        return body
    finally:
        env.close()


def main():
    configure_headless()
    RECORDINGS_DIR.mkdir(parents=True, exist_ok=True)
    r07 = right_mouth()
    r65 = north65()
    out = {"pokes": False, "status_claim": None, "right_mouth": {"ok": r07.get("ok"), "dest": (r07.get("dump") or {}).get("room_hex")}, "north65": {"ok": r65.get("ok"), "dest": (r65.get("dump") or {}).get("room_hex"), "whistle": r65.get("whistle_0x065C")}}
    write_dump("l5_07_right_and_65n", out)
    print("SUMMARY", out, flush=True)
    return out


if __name__ == "__main__":
    main()
