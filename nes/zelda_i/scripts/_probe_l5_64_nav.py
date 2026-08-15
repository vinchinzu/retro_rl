"""From Level5Cleared64: north-around diamond, west, push left block, stairs.

No re-fight. No pokes. Dest must be cellar 0x07.
"""
from __future__ import annotations

import importlib.util
from pathlib import Path

from retro_harness.nes import nes_action
from retro_harness.segment_runner import configure_headless
from zelda_i.dungeon_ops import idle, push_dir
from zelda_i.level9_stairs import dest_report, on_stair_tile, walk_to_step
from zelda_i.paths import RECORDINGS_DIR
from zelda_i.ram import PLAY_MODE, read_snapshot

HERE = Path(__file__).resolve()
_spec = importlib.util.spec_from_file_location("l5whistle", HERE.parent / "_probe_l5_whistle_path.py")
w = importlib.util.module_from_spec(_spec)
assert _spec.loader is not None
_spec.loader.exec_module(w)

ROOM_64 = 0x64
ROOM_07 = 0x07
CELLAR_MODES = (9, 10, 11, 16)


def in_cellar(snap) -> bool:
    return snap.mode in CELLAR_MODES or snap.screen == ROOM_07


def walk(env, assist, total, axis, tgt, max_f=320) -> dict:
    ok = w.walk_axis(env, assist, total, axis, tgt, max_f=max_f)
    snap = read_snapshot(env.get_ram())
    rec = {
        "axis": axis,
        "tgt": tgt,
        "ok": ok,
        "xy": [snap.link_x, snap.link_y],
        "tile": int(snap.colliding_tile),
        "stair": bool(on_stair_tile(snap)),
        "mode": snap.mode,
        "room": f"0x{snap.screen:02x}",
    }
    print("WALK", rec, flush=True)
    return rec


def unstick_east(env, assist, total) -> None:
    """If pressed against the diamond, back up east then north or south."""
    snap = read_snapshot(env.get_ram())
    if snap.link_x >= 168:
        return
    walk(env, assist, total, "x", 192, max_f=200)


def main() -> dict:
    configure_headless()
    RECORDINGS_DIR.mkdir(parents=True, exist_ok=True)
    env, assist, _ = w.open_env("Level5Cleared64")
    total = [1]
    log = []
    try:
        idle(env, assist, total, 16)
        snap = read_snapshot(env.get_ram())
        start = w.dump_live(snap, env.get_ram())
        print("START", start.get("room_hex"), "xy", [start.get("x"), start.get("y")], "n_dn", len(w.live_darknuts(snap)), flush=True)
        if snap.screen != ROOM_64 or w.live_darknuts(snap):
            rec = {"ok": False, "failed_room": "0x64", "reason": "cleared64_not_ready", "start": start}
            w.write_dump("l5_64_nav", rec)
            return rec

        # 1. North corridor around the diamond: y=93, west past x=160.
        log.append(walk(env, assist, total, "y", 93))
        log.append(walk(env, assist, total, "x", 64))
        snap = read_snapshot(env.get_ram())
        if snap.link_x > 140:
            # North blocked; try south corridor.
            print("NORTH_WEST_FAIL try south", flush=True)
            unstick_east(env, assist, total)
            log.append(walk(env, assist, total, "y", 189))
            log.append(walk(env, assist, total, "x", 64))
        west_xy = [read_snapshot(env.get_ram()).link_x, read_snapshot(env.get_ram()).link_y]
        print("WEST_SIDE", west_xy, flush=True)
        w.shot(env, assist, total, "l5_64_west_side")

        # 2. Drop to door-band y=141 on the west, approach left block (96,144).
        log.append(walk(env, assist, total, "y", 141))
        log.append(walk(env, assist, total, "x", 80))
        at_west = w.dump_live(read_snapshot(env.get_ram()), env.get_ram())
        print("AT_WEST", at_west.get("x"), at_west.get("y"), "tile", at_west.get("colliding_tile"), "blocks", at_west.get("blocks_0x68"), flush=True)

        # 3. Push leftmost block from the west / south / east of it.
        push_log = []
        for tx, ty, direction in (
            (80, 144, "RIGHT"),
            (96, 160, "UP"),
            (96, 128, "DOWN"),
            (112, 144, "LEFT"),
            (80, 141, "RIGHT"),
            (64, 144, "RIGHT"),
        ):
            log.append(walk(env, assist, total, "y", ty))
            log.append(walk(env, assist, total, "x", tx))
            push_dir(env, assist, total, direction, frames=110)
            idle(env, assist, total, 10)
            snap = read_snapshot(env.get_ram())
            rec = {
                "stand": [tx, ty],
                "dir": direction,
                "xy": [snap.link_x, snap.link_y],
                "tile": int(snap.colliding_tile),
                "stair": bool(on_stair_tile(snap)),
                "mode": snap.mode,
                "blocks": [{"x": o.x, "y": o.y} for o in snap.objects if 1 <= o.slot <= 12 and o.type_id == 0x68],
            }
            push_log.append(rec)
            print("PUSH", rec, flush=True)
            if in_cellar(snap) or on_stair_tile(snap):
                break

        # 4. Walk onto center stairs from west.
        for tx, ty in ((96, 141), (104, 141), (112, 141), (120, 141), (128, 141), (120, 144), (120, 137)):
            snap = read_snapshot(env.get_ram())
            if in_cellar(snap):
                break
            for _ in range(200):
                snap = read_snapshot(env.get_ram())
                if in_cellar(snap) or (on_stair_tile(snap) and 80 < snap.link_x < 160):
                    break
                frame = walk_to_step(snap, tx, ty, y_first=False, tol=2)
                if frame.reason == "walk_arrived":
                    idle(env, assist, total, 8)
                    break
                w.step(env, assist, total, frame.action)
            snap = read_snapshot(env.get_ram())
            print("STAND", [tx, ty], "at", [snap.link_x, snap.link_y], "tile", snap.colliding_tile, "stair", on_stair_tile(snap), "mode", snap.mode, flush=True)
            if in_cellar(snap) or (on_stair_tile(snap) and snap.colliding_tile != 0x24):
                for direction in ("UP", "DOWN", "LEFT", "RIGHT"):
                    push_dir(env, assist, total, direction, frames=70)
                    if in_cellar(read_snapshot(env.get_ram())):
                        break
                break

        w.wait_play(env, assist, total, max_f=240)
        idle(env, assist, total, 16)
        snap = read_snapshot(env.get_ram())
        dest = w.dump_live(snap, env.get_ram())
        png = w.shot(env, assist, total, "l5_64_nav")
        ok = in_cellar(snap)
        body = {
            "ok": ok,
            "failed_room": None if ok else "0x64",
            "reason": None if ok else "west_around_stairs_failed",
            "pokes": False,
            "status_claim": None,
            "start": {"xy": [start.get("x"), start.get("y")], "room": start.get("room_hex")},
            "west_xy": west_xy,
            "at_west": {"xy": [at_west.get("x"), at_west.get("y")], "tile": at_west.get("colliding_tile"), "blocks": at_west.get("blocks_0x68")},
            "log": log,
            "push_log": push_log,
            "dump": dest,
            "dest": dest_report(snap),
            "screenshot": png,
            "whistle_0x065C": dest.get("whistle_0x065C"),
        }
        w.write_dump("l5_64_nav", body)
        if ok:
            w.dump_and_save_room(env, assist, total, "l5_07_arrive", "Level5Entered07", "Level5Cleared64", "0x64 stairs via west")
        print("DEST", dest.get("room_hex"), "mode", dest.get("mode"), "xy", [dest.get("x"), dest.get("y")], "ok", ok, flush=True)
        return body
    finally:
        env.close()


if __name__ == "__main__":
    r = main()
    print("OK", r.get("ok"), "FAILED", r.get("failed_room"), "WEST", r.get("west_xy"))
