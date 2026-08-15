"""Exit 0x07 via right mouth: raw-step y=165, x=192, y=61, UP. Dump dest."""
from __future__ import annotations

import importlib.util
from pathlib import Path

from retro_harness.nes import nes_action
from retro_harness.segment_runner import configure_headless
from zelda_i.dungeon_ops import idle, push_dir
from zelda_i.level9_stairs import dest_report
from zelda_i.paths import RECORDINGS_DIR
from zelda_i.ram import PLAY_MODE, read_snapshot

HERE = Path(__file__).resolve()
_spec = importlib.util.spec_from_file_location("l5whistle", HERE.parent / "_probe_l5_whistle_path.py")
w = importlib.util.module_from_spec(_spec)
assert _spec.loader is not None
_spec.loader.exec_module(w)


def raw_axis(env, assist, total, axis, tgt, max_f=500) -> list:
    last = None
    stall = 0
    for _ in range(max_f):
        snap = read_snapshot(env.get_ram())
        if snap.mode == PLAY_MODE and snap.screen != 0x07:
            return [snap.link_x, snap.link_y]
        if axis == "x":
            if abs(snap.link_x - tgt) <= 1:
                return [snap.link_x, snap.link_y]
            w.step(env, assist, total, nes_action("RIGHT" if snap.link_x < tgt else "LEFT"))
        else:
            if abs(snap.link_y - tgt) <= 1:
                return [snap.link_x, snap.link_y]
            w.step(env, assist, total, nes_action("DOWN" if snap.link_y < tgt else "UP"))
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


def main() -> dict:
    configure_headless()
    RECORDINGS_DIR.mkdir(parents=True, exist_ok=True)
    env, assist, _ = w.open_env("Level5Entered07")
    total = [1]
    log = []
    try:
        idle(env, assist, total, 10)
        start = w.dump_live(read_snapshot(env.get_ram()), env.get_ram())
        print("START", [start.get("x"), start.get("y")], start.get("mode"), flush=True)
        for axis, tgt in (("y", 165), ("x", 192), ("y", 61), ("x", 176)):
            xy = raw_axis(env, assist, total, axis, tgt)
            snap = read_snapshot(env.get_ram())
            rec = {"axis": axis, "tgt": tgt, "xy": xy, "mode": snap.mode, "room": f"0x{snap.screen:02x}"}
            log.append(rec)
            print("WALK", rec, flush=True)
            if snap.mode == PLAY_MODE and snap.screen != 0x07:
                break
        push_dir(env, assist, total, "UP", frames=200)
        idle(env, assist, total, 16)
        w.wait_play(env, assist, total, max_f=280)
        idle(env, assist, total, 16)
        snap = read_snapshot(env.get_ram())
        dest = w.dump_live(snap, env.get_ram())
        png = w.shot(env, assist, total, "l5_06_arrive")
        ok = snap.screen == 0x06 and snap.mode == PLAY_MODE
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
            "rom": w.rom_room(int(snap.screen)),
        }
        w.write_dump("l5_06_arrive", body)
        print("DEST", dest.get("room_hex"), dest.get("mode"), [dest.get("x"), dest.get("y")], "ok", ok, flush=True)
        if ok:
            w.dump_and_save_room(env, assist, total, "l5_06_arrive", "Level5Entered06", "Level5Entered07", "0x07 right mouth y165")
        return body
    finally:
        env.close()


if __name__ == "__main__":
    r = main()
    print("OK", r.get("ok"), "ROOM", (r.get("dump") or {}).get("room_hex"), (r.get("dump") or {}).get("mode"))
