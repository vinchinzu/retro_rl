"""0x07 right mouth at y=169 (below 0x68 at y=160). One try, then stop."""
from retro_harness.nes import nes_action
from retro_harness.segment_runner import configure_headless
from zelda_i.dungeon_ops import idle, push_dir
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


def main():
    configure_headless()
    RECORDINGS_DIR.mkdir(parents=True, exist_ok=True)
    env, assist, _ = open_env("Level5Entered07")
    total = [1]
    log = []
    try:
        idle(env, assist, total, 20)
        for _ in range(16):
            step(env, assist, total, nes_action("DOWN"))
        snap = read_snapshot(env.get_ram())
        print("START", [snap.link_x, snap.link_y], "mode", snap.mode, "objs",
              [(o.type_id, o.x, o.y) for o in snap.objects if 1 <= o.slot <= 12 and o.type_id not in (0, 0xFF)], flush=True)
        for axis, tgt in (("y", 169), ("x", 192), ("y", 61)):
            xy = raw_axis(env, assist, total, axis, tgt)
            snap = read_snapshot(env.get_ram())
            rec = {"axis": axis, "tgt": tgt, "xy": xy, "mode": snap.mode, "room": f"0x{snap.screen:02x}"}
            log.append(rec)
            print("WALK", rec, flush=True)
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
        write_dump("l5_06_arrive", {"ok": ok, "failed_room": None if ok else "0x07", "pokes": False, "status_claim": None, "log": log, "dump": dest, "screenshot": png, "whistle_0x065C": dest.get("whistle_0x065C"), "rom": rom_room(int(snap.screen))})
        print("DEST", dest.get("room_hex"), dest.get("mode"), [dest.get("x"), dest.get("y")], "ok", ok, flush=True)
        if ok:
            dump_and_save_room(env, assist, total, "l5_06_arrive", "Level5Entered06", "Level5Entered07", "0x07 right mouth y169")
        return ok
    finally:
        env.close()


if __name__ == "__main__":
    main()
