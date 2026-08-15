"""From Level5Whistle right ladder, find Y that crosses left, then exit UP."""
from __future__ import annotations

import importlib.util
from pathlib import Path

from retro_harness.nes import nes_action
from retro_harness.segment_runner import configure_headless
from zelda_i.dungeon_ops import idle, push_dir
from zelda_i.paths import RECORDINGS_DIR
from zelda_i.ram import ADDR_WHISTLE, PLAY_MODE, read_snapshot, read_u8

HERE = Path(__file__).resolve()
_spec = importlib.util.spec_from_file_location("l5whistle", HERE.parent / "_probe_l5_whistle_path.py")
w = importlib.util.module_from_spec(_spec)
assert _spec.loader is not None
_spec.loader.exec_module(w)


def raw_axis(env, assist, total, axis, tgt, max_f=400) -> list:
    last = None
    stall = 0
    for _ in range(max_f):
        snap = read_snapshot(env.get_ram())
        if snap.mode == PLAY_MODE and snap.screen != 0x04:
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
            if stall >= 30:
                return [snap2.link_x, snap2.link_y]
        else:
            stall = 0
        last = pos
    snap = read_snapshot(env.get_ram())
    return [snap.link_x, snap.link_y]


def main() -> dict:
    configure_headless()
    rows = []
    for ytgt in (165, 169, 173, 177, 181, 189):
        env, assist, _ = w.open_env("Level5Whistle")
        total = [1]
        try:
            idle(env, assist, total, 4)
            raw_axis(env, assist, total, "x", 176)
            raw_axis(env, assist, total, "y", ytgt)
            after_y = [read_snapshot(env.get_ram()).link_x, read_snapshot(env.get_ram()).link_y]
            raw_axis(env, assist, total, "x", 48)
            snap = read_snapshot(env.get_ram())
            rec = {"ytgt": ytgt, "after_y": after_y, "after_x": [snap.link_x, snap.link_y], "crossed": snap.link_x <= 64}
            print("Y", rec, flush=True)
            rows.append(rec)
        finally:
            env.close()
    # If any crossed, do full exit at that Y
    good = next((r["ytgt"] for r in rows if r["crossed"]), None)
    result = {"rows": rows, "good_y": good, "pokes": False, "status_claim": None}
    if good is None:
        w.write_dump("l5_04_exit2", result)
        return result
    env, assist, _ = w.open_env("Level5Whistle")
    total = [1]
    try:
        idle(env, assist, total, 6)
        raw_axis(env, assist, total, "x", 176)
        raw_axis(env, assist, total, "y", good)
        raw_axis(env, assist, total, "x", 48)
        raw_axis(env, assist, total, "y", 61)
        push_dir(env, assist, total, "UP", frames=220)
        idle(env, assist, total, 16)
        w.wait_play(env, assist, total, max_f=280)
        idle(env, assist, total, 12)
        snap = read_snapshot(env.get_ram())
        dest = w.dump_live(snap, env.get_ram())
        ok = snap.screen == 0x05
        result.update({"ok": ok, "dest": dest, "whistle_0x065C": int(read_u8(env.get_ram(), ADDR_WHISTLE))})
        print("DEST", dest.get("room_hex"), dest.get("mode"), [dest.get("x"), dest.get("y")], "ok", ok, flush=True)
        w.shot(env, assist, total, "l5_04_exit")
        if ok:
            w.save_ckpt(env, "Level5Whistle05", "Level5Whistle", {"segment": "Level5Whistle05", "via": "0x04 left mouth", "key_poke": False, "door_poke": False}, {"success": True, "room": 0x05, "whistle_0x065C": 1})
    finally:
        env.close()
    w.write_dump("l5_04_exit2", result)
    return result


if __name__ == "__main__":
    r = main()
    print("GOOD_Y", r.get("good_y"), "OK", r.get("ok"), "WHISTLE", r.get("whistle_0x065C"))
