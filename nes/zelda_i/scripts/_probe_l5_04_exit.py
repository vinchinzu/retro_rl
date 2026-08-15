"""Exit Level5Whistle 0x04 via right ladder down, cross y=165, left mouth UP -> 0x05."""
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


def raw_axis(env, assist, total, axis, tgt, max_f=500) -> list:
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
    env, assist, _ = w.open_env("Level5Whistle")
    total = [1]
    log = []
    try:
        idle(env, assist, total, 8)
        print("START", [read_snapshot(env.get_ram()).link_x, read_snapshot(env.get_ram()).link_y], "whistle", int(read_u8(env.get_ram(), ADDR_WHISTLE)), flush=True)
        for axis, tgt in (("x", 176), ("y", 165), ("x", 48), ("y", 61)):
            xy = raw_axis(env, assist, total, axis, tgt)
            snap = read_snapshot(env.get_ram())
            rec = {"axis": axis, "tgt": tgt, "xy": xy, "mode": snap.mode, "room": f"0x{snap.screen:02x}"}
            log.append(rec)
            print("WALK", rec, flush=True)
            if snap.mode == PLAY_MODE and snap.screen != 0x04:
                break
        push_dir(env, assist, total, "UP", frames=220)
        idle(env, assist, total, 16)
        w.wait_play(env, assist, total, max_f=280)
        idle(env, assist, total, 12)
        snap = read_snapshot(env.get_ram())
        dest = w.dump_live(snap, env.get_ram())
        png = w.shot(env, assist, total, "l5_04_exit")
        ok = snap.screen == 0x05 and snap.mode == PLAY_MODE
        body = {"ok": ok, "failed_room": None if ok else "0x04", "log": log, "dump": dest, "screenshot": png, "whistle_0x065C": int(read_u8(env.get_ram(), ADDR_WHISTLE)), "pokes": False, "status_claim": None}
        w.write_dump("l5_04_exit", body)
        print("DEST", dest.get("room_hex"), dest.get("mode"), [dest.get("x"), dest.get("y")], "whistle", body["whistle_0x065C"], "ok", ok, flush=True)
        if ok:
            w.save_ckpt(env, "Level5Whistle05", "Level5Whistle", {"segment": "Level5Whistle05", "via": "0x04 left mouth", "key_poke": False, "door_poke": False}, {"success": True, "room": 0x05, "whistle_0x065C": 1})
        return body
    finally:
        env.close()


if __name__ == "__main__":
    r = main()
    print("OK", r.get("ok"), "FAILED", r.get("failed_room"), "WHISTLE", r.get("whistle_0x065C"))
