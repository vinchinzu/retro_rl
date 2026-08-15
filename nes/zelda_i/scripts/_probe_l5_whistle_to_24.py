"""From Level5Whistle: exit 0x04 -> 0x05 -> 0x06 -> cellar -> 0x64 -> 0x65, then toward 0x24."""
from __future__ import annotations

import importlib.util
from pathlib import Path

from retro_harness.nes import nes_action
from retro_harness.segment_runner import configure_headless
from zelda_i.dungeon_ops import idle, push_dir
from zelda_i.level9_stairs import dest_report, on_stair_tile
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
        if snap.mode == PLAY_MODE and snap.screen not in (0x04, 0x07):
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


def exit_cellar_left(env, assist, total) -> dict:
    """Both 0x04 mouths -> 0x05. Climb left mouth and UP."""
    raw_axis(env, assist, total, "y", 165)
    raw_axis(env, assist, total, "x", 48)
    raw_axis(env, assist, total, "y", 61)
    push_dir(env, assist, total, "UP", frames=200)
    idle(env, assist, total, 16)
    w.wait_play(env, assist, total, max_f=280)
    idle(env, assist, total, 12)
    snap = read_snapshot(env.get_ram())
    return w.dump_live(snap, env.get_ram())


def take_06_stairs(env, assist, total) -> dict:
    """Same diamond as 0x64: north around, push left block north, walk (120,141)."""
    w.walk_axis(env, assist, total, "y", 93, max_f=300)
    w.walk_axis(env, assist, total, "x", 64, max_f=300)
    w.walk_axis(env, assist, total, "y", 160, max_f=300)
    w.walk_axis(env, assist, total, "x", 96, max_f=300)
    push_dir(env, assist, total, "UP", frames=120)
    idle(env, assist, total, 8)
    w.walk_axis(env, assist, total, "y", 141, max_f=300)
    w.walk_axis(env, assist, total, "x", 120, max_f=300)
    for d in ("UP", "DOWN", "LEFT", "RIGHT"):
        push_dir(env, assist, total, d, frames=60)
        snap = read_snapshot(env.get_ram())
        if snap.mode in (9, 10, 11, 16) or snap.screen == 0x07:
            break
    w.wait_play(env, assist, total, max_f=240)
    idle(env, assist, total, 12)
    return w.dump_live(read_snapshot(env.get_ram()), env.get_ram())


def cellar_07_to_64(env, assist, total) -> dict:
    """From 0x07, go to LEFT mouth (0x64). Spawn may be right (from 0x06)."""
    snap = read_snapshot(env.get_ram())
    # Cross at y=165 to left, climb, UP.
    raw_axis(env, assist, total, "y", 165)
    raw_axis(env, assist, total, "x", 48)
    raw_axis(env, assist, total, "y", 61)
    push_dir(env, assist, total, "UP", frames=200)
    idle(env, assist, total, 16)
    w.wait_play(env, assist, total, max_f=280)
    idle(env, assist, total, 12)
    return w.dump_live(read_snapshot(env.get_ram()), env.get_ram())


def main() -> dict:
    configure_headless()
    RECORDINGS_DIR.mkdir(parents=True, exist_ok=True)
    hops = []
    env, assist, _ = w.open_env("Level5Whistle")
    total = [1]
    try:
        idle(env, assist, total, 10)
        snap = read_snapshot(env.get_ram())
        print("START", f"0x{snap.screen:02x}", snap.mode, [snap.link_x, snap.link_y], "whistle", int(read_u8(env.get_ram(), ADDR_WHISTLE)), flush=True)
        if int(read_u8(env.get_ram(), ADDR_WHISTLE)) < 1:
            rec = {"ok": False, "failed_room": "0x04", "reason": "whistle_save_lost"}
            w.write_dump("l5_24_whistle_boss", rec)
            return rec

        d05 = exit_cellar_left(env, assist, total)
        hops.append({"hop": "0x04_exit", "dest": d05.get("room_hex"), "ok": d05.get("room") == 0x05})
        print("EXIT04", hops[-1], [d05.get("x"), d05.get("y")], flush=True)
        if d05.get("room") != 0x05:
            rec = {"ok": False, "failed_room": "0x04", "reason": "exit_not_0x05", "hops": hops, "dump": d05, "pokes": False, "status_claim": None}
            w.write_dump("l5_24_whistle_boss", rec)
            return rec

        # East back to 0x06
        w.walk_axis(env, assist, total, "y", 141, max_f=300)
        w.walk_axis(env, assist, total, "x", 224, max_f=400)
        push_dir(env, assist, total, "RIGHT", frames=220)
        idle(env, assist, total, 12)
        w.wait_play(env, assist, total, 0x06, max_f=240)
        snap = read_snapshot(env.get_ram())
        hops.append({"hop": "0x05_east", "dest": f"0x{snap.screen:02x}", "ok": snap.screen == 0x06})
        print("EAST05", hops[-1], flush=True)
        if snap.screen != 0x06:
            rec = {"ok": False, "failed_room": "0x05", "reason": "east_not_0x06", "hops": hops, "pokes": False, "status_claim": None}
            w.write_dump("l5_24_whistle_boss", rec)
            return rec

        d07 = take_06_stairs(env, assist, total)
        hops.append({"hop": "0x06_stairs", "dest": d07.get("room_hex"), "mode": d07.get("mode"), "ok": d07.get("room") == 0x07 or d07.get("mode") in (9, 10)})
        print("STAIRS06", hops[-1], flush=True)
        if not (d07.get("room") == 0x07 or d07.get("mode") in (9, 10, 11, 16)):
            rec = {"ok": False, "failed_room": "0x06", "reason": "stairs_not_0x07", "hops": hops, "dump": d07, "pokes": False, "status_claim": None}
            w.write_dump("l5_24_whistle_boss", rec)
            return rec

        d64 = cellar_07_to_64(env, assist, total)
        hops.append({"hop": "0x07_to_64", "dest": d64.get("room_hex"), "ok": d64.get("room") == 0x64})
        print("TO64", hops[-1], flush=True)
        if d64.get("room") != 0x64:
            rec = {"ok": False, "failed_room": "0x07", "reason": "left_mouth_not_0x64", "hops": hops, "dump": d64, "pokes": False, "status_claim": None}
            w.write_dump("l5_24_whistle_boss", rec)
            return rec

        # East bomb hole 0x64 -> 0x65
        w.walk_axis(env, assist, total, "y", 93, max_f=300)
        w.walk_axis(env, assist, total, "x", 192, max_f=300)
        w.walk_axis(env, assist, total, "y", 141, max_f=300)
        w.walk_axis(env, assist, total, "x", 224, max_f=300)
        push_dir(env, assist, total, "RIGHT", frames=220)
        idle(env, assist, total, 12)
        w.wait_play(env, assist, total, 0x65, max_f=240)
        snap = read_snapshot(env.get_ram())
        hops.append({"hop": "0x64_east", "dest": f"0x{snap.screen:02x}", "ok": snap.screen == 0x65, "whistle": int(read_u8(env.get_ram(), ADDR_WHISTLE))})
        print("TO65", hops[-1], flush=True)
        if snap.screen != 0x65:
            rec = {"ok": False, "failed_room": "0x64", "reason": "east_not_0x65", "hops": hops, "pokes": False, "status_claim": None}
            w.write_dump("l5_24_whistle_boss", rec)
            return rec

        # North shutter 0x65 -> 0x55
        w.walk_axis(env, assist, total, "x", 120, max_f=300)
        w.walk_axis(env, assist, total, "y", 93, max_f=300)
        push_dir(env, assist, total, "UP", frames=220)
        idle(env, assist, total, 12)
        w.wait_play(env, assist, total, 0x55, max_f=240)
        snap = read_snapshot(env.get_ram())
        hops.append({"hop": "0x65_north", "dest": f"0x{snap.screen:02x}", "ok": snap.screen == 0x55})
        print("TO55", hops[-1], flush=True)
        png = w.shot(env, assist, total, "l5_whistle_backtrack")
        rec = {
            "ok": False,
            "failed_room": None if snap.screen == 0x55 else "0x65",
            "reason": "backtrack_reached_" + f"0x{snap.screen:02x}" + "_need_path_to_0x24",
            "pokes": False,
            "status_claim": None,
            "hops": hops,
            "whistle_0x065C": int(read_u8(env.get_ram(), ADDR_WHISTLE)),
            "now": w.dump_live(snap, env.get_ram()),
            "screenshot": png,
        }
        w.write_dump("l5_24_whistle_boss", rec)
        print("STOP", rec["reason"], "whistle", rec["whistle_0x065C"], flush=True)
        return rec
    finally:
        env.close()


if __name__ == "__main__":
    r = main()
    print("OK", r.get("ok"), "FAILED", r.get("failed_room"), "HOPS", r.get("hops"), "WHISTLE", r.get("whistle_0x065C"))
