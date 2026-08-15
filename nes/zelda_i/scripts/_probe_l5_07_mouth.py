"""From Level5Entered07, walk RIGHT mouth via cellar_exit_step. Dump dest."""
from __future__ import annotations

import importlib.util
from pathlib import Path

from retro_harness.segment_runner import configure_headless
from zelda_i.dungeon_ops import idle
from zelda_i.level9_stairs import cellar_exit_step, dest_report
from zelda_i.paths import RECORDINGS_DIR
from zelda_i.ram import PLAY_MODE, read_snapshot

HERE = Path(__file__).resolve()
_spec = importlib.util.spec_from_file_location("l5whistle", HERE.parent / "_probe_l5_whistle_path.py")
w = importlib.util.module_from_spec(_spec)
assert _spec.loader is not None
_spec.loader.exec_module(w)


def run_side(env, assist, total, side: str, max_f: int = 900) -> dict:
    log = []
    last = None
    for i in range(max_f):
        snap = read_snapshot(env.get_ram())
        rec = {"f": i, "xy": [snap.link_x, snap.link_y], "mode": snap.mode, "room": f"0x{snap.screen:02x}"}
        if last is None or rec["xy"] != last["xy"] or rec["mode"] != last["mode"] or rec["room"] != last["room"]:
            if i % 20 == 0 or rec["mode"] != (last or {}).get("mode") or rec["room"] != (last or {}).get("room"):
                log.append(rec)
                print("F", rec, flush=True)
            last = rec
        if snap.mode == PLAY_MODE and snap.screen != 0x07:
            break
        if snap.mode == PLAY_MODE and snap.level == 0:
            break
        frame = cellar_exit_step(snap, side=side)
        w.step(env, assist, total, frame.action)
    idle(env, assist, total, 16)
    w.wait_play(env, assist, total, max_f=200)
    idle(env, assist, total, 12)
    snap = read_snapshot(env.get_ram())
    return {"side": side, "log": log[-20:], "end": w.dump_live(snap, env.get_ram()), "dest": dest_report(snap)}


def main() -> dict:
    configure_headless()
    RECORDINGS_DIR.mkdir(parents=True, exist_ok=True)
    env, assist, _ = w.open_env("Level5Entered07")
    total = [1]
    try:
        idle(env, assist, total, 12)
        start = w.dump_live(read_snapshot(env.get_ram()), env.get_ram())
        print("START", start.get("room_hex"), start.get("mode"), [start.get("x"), start.get("y")], flush=True)
        w.shot(env, assist, total, "l5_07_cellar")
        # Right mouth first (spawn is left / 0x64).
        right = run_side(env, assist, total, "right")
        snap = read_snapshot(env.get_ram())
        print("RIGHT_DEST", f"0x{snap.screen:02x}", "mode", snap.mode, [snap.link_x, snap.link_y], flush=True)
        png = w.shot(env, assist, total, "l5_07_right_mouth")
        ok = snap.screen == 0x06
        body = {
            "ok": ok,
            "failed_room": None if ok else "0x07",
            "pokes": False,
            "status_claim": None,
            "start": {"xy": [start.get("x"), start.get("y")], "mode": start.get("mode"), "room": start.get("room_hex")},
            "right": {"dest": f"0x{snap.screen:02x}", "mode": snap.mode, "xy": [snap.link_x, snap.link_y], "log": right["log"]},
            "dump": right["end"],
            "screenshot": png,
        }
        w.write_dump("l5_07_mouth", body)
        if ok:
            w.dump_and_save_room(env, assist, total, "l5_06_arrive", "Level5Entered06", "Level5Entered07", "0x07 right mouth")
        return body
    finally:
        env.close()


if __name__ == "__main__":
    r = main()
    print("OK", r.get("ok"), "RIGHT", r.get("right"))
