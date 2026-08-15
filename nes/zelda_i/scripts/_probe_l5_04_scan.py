"""Dump 0x04 objects and sweep mid-pit / right-ladder stands for whistle."""
from __future__ import annotations

import importlib.util
from pathlib import Path

from retro_harness.nes import nes_action
from retro_harness.segment_runner import configure_headless
from zelda_i.dungeon_ops import idle
from zelda_i.paths import RECORDINGS_DIR
from zelda_i.ram import ADDR_WHISTLE, read_snapshot, read_u8

HERE = Path(__file__).resolve()
_spec = importlib.util.spec_from_file_location("l5whistle", HERE.parent / "_probe_l5_whistle_path.py")
w = importlib.util.module_from_spec(_spec)
assert _spec.loader is not None
_spec.loader.exec_module(w)


def raw_axis(env, assist, total, axis, tgt, max_f=400) -> list:
    last = None
    stall = 0
    for _ in range(max_f):
        if int(read_u8(env.get_ram(), ADDR_WHISTLE)) >= 1:
            snap = read_snapshot(env.get_ram())
            return [snap.link_x, snap.link_y]
        snap = read_snapshot(env.get_ram())
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


def objs(snap):
    return [
        {"slot": o.slot, "type": f"0x{o.type_id:02x}", "x": o.x, "y": o.y, "hp": o.hp, "state": o.state}
        for o in snap.objects if 1 <= o.slot <= 12
    ]


def main() -> dict:
    configure_headless()
    RECORDINGS_DIR.mkdir(parents=True, exist_ok=True)
    env, assist, _ = w.open_env("Level5Entered04")
    total = [1]
    log = []
    try:
        idle(env, assist, total, 8)
        snap = read_snapshot(env.get_ram())
        print("START objs", objs(snap), "item", snap.room_item_id, "whistle", int(read_u8(env.get_ram(), ADDR_WHISTLE)), flush=True)
        # Down left ladder, across at 165, up right ladder, sweep mid Y.
        path = (("y", 165), ("x", 192), ("y", 165), ("x", 176), ("y", 141), ("x", 176), ("y", 120), ("x", 176), ("y", 93), ("x", 176), ("y", 61), ("x", 176))
        for axis, tgt in path:
            xy = raw_axis(env, assist, total, axis, tgt)
            snap = read_snapshot(env.get_ram())
            rec = {"axis": axis, "tgt": tgt, "xy": xy, "whistle": int(read_u8(env.get_ram(), ADDR_WHISTLE)), "mode": snap.mode, "room": f"0x{snap.screen:02x}", "item": snap.room_item_id, "objs": objs(snap)}
            log.append({k: rec[k] for k in rec if k != "objs"})
            print("WALK", {k: rec[k] for k in rec if k != "objs"}, "objs", rec["objs"], flush=True)
            if rec["whistle"] >= 1:
                break
        w.shot(env, assist, total, "l5_04_scan")
        # Also try center x at several Y from right ladder
        for ty in (165, 157, 149, 141, 133, 125, 117, 109):
            raw_axis(env, assist, total, "y", ty)
            raw_axis(env, assist, total, "x", 120)
            snap = read_snapshot(env.get_ram())
            val = int(read_u8(env.get_ram(), ADDR_WHISTLE))
            print("MID", ty, [snap.link_x, snap.link_y], "whistle", val, "objs", objs(snap), flush=True)
            log.append({"mid_y": ty, "xy": [snap.link_x, snap.link_y], "whistle": val})
            if val >= 1:
                break
            raw_axis(env, assist, total, "x", 176)
        whistle = int(read_u8(env.get_ram(), ADDR_WHISTLE))
        snap = read_snapshot(env.get_ram())
        body = {"ok": whistle >= 1, "failed_room": None if whistle >= 1 else "0x04", "pokes": False, "status_claim": None, "log": log, "whistle_0x065C": whistle, "final": w.dump_live(snap, env.get_ram())}
        w.write_dump("l5_04_scan", body)
        png = w.shot(env, assist, total, "l5_04_whistle")
        if whistle >= 1:
            w.save_ckpt(env, "Level5Whistle", "Level5Entered04", {"segment": "Level5Whistle", "via": "0x04 scan", "key_poke": False, "door_poke": False}, {"success": True, "whistle_0x065C": whistle})
        return body
    finally:
        env.close()


if __name__ == "__main__":
    r = main()
    print("OK", r.get("ok"), "WHISTLE", r.get("whistle_0x065C"))
