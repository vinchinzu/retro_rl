"""Settle 0x07 then y165/x192 right mouth. Re-enter 0x65 from 0x64 to spawn/open north shutter."""
from retro_harness.nes import nes_action
from retro_harness.segment_runner import configure_headless
from zelda_i.dungeon_ops import idle, push_dir
from zelda_i.paths import RECORDINGS_DIR
from zelda_i.ram import PLAY_MODE, read_snapshot
from zelda_i.scripts._probe_l5_whistle_path import (
    dump_and_save_room,
    dump_live,
    fight_darknuts,
    live_darknuts,
    open_env,
    rom_room,
    shot,
    step,
    wait_play,
    walk_axis,
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
        idle(env, assist, total, 40)
        snap = read_snapshot(env.get_ram())
        start = dump_live(snap, env.get_ram())
        print("07START", [snap.link_x, snap.link_y], "mode", snap.mode, flush=True)
        # force a DOWN then re-read so fake (128,141) resolves
        for _ in range(20):
            step(env, assist, total, nes_action("DOWN"))
        snap = read_snapshot(env.get_ram())
        print("07AFTERDOWN", [snap.link_x, snap.link_y], flush=True)
        for axis, tgt in (("y", 165), ("x", 192), ("y", 61)):
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
        write_dump(
            "l5_06_arrive",
            {
                "ok": ok,
                "failed_room": None if ok else "0x07",
                "pokes": False,
                "status_claim": None,
                "start": [start.get("x"), start.get("y")],
                "log": log,
                "dump": dest,
                "screenshot": png,
                "whistle_0x065C": dest.get("whistle_0x065C"),
                "rom": rom_room(int(snap.screen)),
            },
        )
        print("07DEST", dest.get("room_hex"), dest.get("mode"), [dest.get("x"), dest.get("y")], "ok", ok, flush=True)
        if ok:
            dump_and_save_room(env, assist, total, "l5_06_arrive", "Level5Entered06", "Level5Entered07", "0x07 right mouth settle+y165")
        return {"ok": ok, "dest": dest.get("room_hex"), "xy": [dest.get("x"), dest.get("y")]}
    finally:
        env.close()


def reenter65():
    env, assist, _ = open_env("Level5Whistle65")
    total = [1]
    log = []
    try:
        idle(env, assist, total, 10)
        start = dump_live(read_snapshot(env.get_ram()), env.get_ram())
        print("65A", start.get("room_hex"), [start.get("x"), start.get("y")], start.get("doors"), flush=True)
        # west to 0x64 through bomb hole
        walk_axis(env, assist, total, "y", 141, max_f=400)
        walk_axis(env, assist, total, "x", 32, max_f=400)
        push_dir(env, assist, total, "LEFT", frames=220)
        idle(env, assist, total, 16)
        wait_play(env, assist, total, 0x64, max_f=240)
        idle(env, assist, total, 12)
        snap = read_snapshot(env.get_ram())
        at64 = dump_live(snap, env.get_ram())
        log.append({"tag": "at64", "room": at64.get("room_hex"), "xy": [at64.get("x"), at64.get("y")], "objs": [(o.get("type_hex"), o.get("hp")) for o in at64.get("objects") or []]})
        print("AT64", log[-1], flush=True)
        if snap.screen != 0x64:
            rec = {"ok": False, "failed_room": "0x65", "reason": "west_not_0x64", "log": log, "pokes": False, "status_claim": None}
            write_dump("l5_65_reenter", rec)
            return rec
        # back east into 0x65
        walk_axis(env, assist, total, "y", 141, max_f=300)
        walk_axis(env, assist, total, "x", 224, max_f=400)
        push_dir(env, assist, total, "RIGHT", frames=220)
        idle(env, assist, total, 16)
        wait_play(env, assist, total, 0x65, max_f=240)
        idle(env, assist, total, 16)
        snap = read_snapshot(env.get_ram())
        back = dump_live(snap, env.get_ram())
        log.append({
            "tag": "re65",
            "room": back.get("room_hex"),
            "xy": [back.get("x"), back.get("y")],
            "doors": back.get("doors"),
            "objs": [(o.get("type_hex"), o.get("type_name"), o.get("hp")) for o in back.get("objects") or []],
        })
        print("RE65", log[-1], flush=True)
        # if enemies, we do not invent a new room; fight in 0x65 only if they are here
        n = len(live_darknuts(snap)) + len([
            o for o in snap.objects
            if 1 <= o.slot <= 12 and o.type_id not in (0, 0xFF, 0x68, 0x55, 0x4E, 0x40) and o.hp > 0
        ])
        fight = None
        if n:
            # reuse darknut fighter only if darknuts; otherwise just report types
            dns = live_darknuts(snap)
            if dns:
                fight = fight_darknuts(env, assist, total, 0x65, expected=len(dns), source=0x64)
                print("FIGHT65", fight.get("ok"), fight.get("end_n"), flush=True)
        idle(env, assist, total, 12)
        snap = read_snapshot(env.get_ram())
        after = dump_live(snap, env.get_ram())
        walk_axis(env, assist, total, "x", 120, max_f=300)
        walk_axis(env, assist, total, "y", 93, max_f=400)
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
        rec = {
            "ok": ok,
            "failed_room": None if ok else "0x65",
            "reason": None if ok else "north_shutter_still_closed",
            "pokes": False,
            "status_claim": None,
            "start": {"room": start.get("room_hex"), "doors": start.get("doors"), "n_obj": len(start.get("objects") or [])},
            "log": log,
            "fight": None if fight is None else {k: fight[k] for k in fight if k != "controller"},
            "after_reenter": {"room": after.get("room_hex"), "doors": after.get("doors"), "objs": [(o.get("type_hex"), o.get("hp")) for o in after.get("objects") or []]},
            "dest": dest.get("room_hex"),
            "dest_xy": [dest.get("x"), dest.get("y")],
            "dest_doors": dest.get("doors"),
            "whistle_0x065C": dest.get("whistle_0x065C"),
            "screenshot": png,
        }
        write_dump("l5_65_reenter", rec)
        print("65FINAL", dest.get("room_hex"), dest.get("doors"), "ok", ok, flush=True)
        return rec
    finally:
        env.close()


def main():
    configure_headless()
    RECORDINGS_DIR.mkdir(parents=True, exist_ok=True)
    r07 = right_mouth()
    r65 = reenter65()
    out = {"pokes": False, "status_claim": None, "right_mouth": r07, "reenter65": {k: r65[k] for k in r65 if k != "log"}}
    write_dump("l5_07_cross_and_65re", out)
    print("SUMMARY", out, flush=True)


if __name__ == "__main__":
    main()
