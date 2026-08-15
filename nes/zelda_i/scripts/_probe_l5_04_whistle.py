"""From Level5Entered04: raw-walk cellar floor for recorder. walk_axis bails on mode 9."""
from __future__ import annotations

import importlib.util
from pathlib import Path

from retro_harness.nes import nes_action
from retro_harness.segment_runner import configure_headless
from zelda_i.dungeon_ops import idle
from zelda_i.level9_stairs import dest_report
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


def main() -> dict:
    configure_headless()
    RECORDINGS_DIR.mkdir(parents=True, exist_ok=True)
    env, assist, _ = w.open_env("Level5Entered04")
    total = [1]
    hits = []
    try:
        idle(env, assist, total, 10)
        w0 = int(read_u8(env.get_ram(), ADDR_WHISTLE))
        start = w.dump_live(read_snapshot(env.get_ram()), env.get_ram())
        print("START04", [start.get("x"), start.get("y")], "mode", start.get("mode"), "item", start.get("room_item_id"), "whistle", w0, flush=True)
        w.shot(env, assist, total, "l5_04_arrive")
        # Down to pit, sweep the floor, including center where recorder usually sits.
        stands = (
            ("y", 165),
            ("x", 80),
            ("x", 120),
            ("x", 160),
            ("x", 192),
            ("x", 120),
            ("y", 173),
            ("x", 80),
            ("x", 160),
            ("y", 157),
            ("x", 120),
            ("y", 181),
            ("x", 120),
            ("x", 64),
            ("x", 176),
            ("y", 165),
            ("x", 120),
        )
        for axis, tgt in stands:
            xy = raw_axis(env, assist, total, axis, tgt)
            val = int(read_u8(env.get_ram(), ADDR_WHISTLE))
            snap = read_snapshot(env.get_ram())
            rec = {"axis": axis, "tgt": tgt, "xy": xy, "whistle": val, "mode": snap.mode, "room": f"0x{snap.screen:02x}", "tile": int(snap.colliding_tile)}
            hits.append(rec)
            print("WALK", rec, flush=True)
            if val >= 1:
                break
        idle(env, assist, total, 12)
        whistle = int(read_u8(env.get_ram(), ADDR_WHISTLE))
        snap = read_snapshot(env.get_ram())
        final = w.dump_live(snap, env.get_ram())
        png = w.shot(env, assist, total, "l5_04_whistle")
        ok = whistle >= 1
        body = {
            "ok": ok,
            "failed_room": None if ok else "0x04",
            "pokes": False,
            "status_claim": None,
            "start": [start.get("x"), start.get("y")],
            "hits": hits,
            "final": final,
            "dest": dest_report(snap),
            "screenshot": png,
            "whistle_0x065C": whistle,
        }
        w.write_dump("l5_04_whistle", body)
        print("WHISTLE", w0, "->", whistle, "xy", [snap.link_x, snap.link_y], flush=True)
        if ok:
            w.save_ckpt(
                env,
                "Level5Whistle",
                "Level5Entered04",
                {"segment": "Level5Whistle", "via": "0x04 cellar floor raw-walk", "key_poke": False, "door_poke": False, "bomb_count_poke": False, "selected_item_poke": False},
                {"success": True, "room": int(snap.screen), "whistle_0x065C": whistle, "bombs": int(snap.bombs), "keys": int(snap.keys)},
            )
        return body
    finally:
        env.close()


if __name__ == "__main__":
    r = main()
    print("OK", r.get("ok"), "WHISTLE", r.get("whistle_0x065C"), "FAILED", r.get("failed_room"))
