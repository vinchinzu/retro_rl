"""Level5Cleared64: try north-center stairs, then push LEFT (not onto stairs)."""
from __future__ import annotations

import importlib.util
from pathlib import Path

from retro_harness.segment_runner import configure_headless
from zelda_i.dungeon_ops import idle, push_dir
from zelda_i.level9_stairs import dest_report, on_stair_tile, walk_to_step
from zelda_i.paths import RECORDINGS_DIR
from zelda_i.ram import read_snapshot

HERE = Path(__file__).resolve()
_spec = importlib.util.spec_from_file_location("l5whistle", HERE.parent / "_probe_l5_whistle_path.py")
w = importlib.util.module_from_spec(_spec)
assert _spec.loader is not None
_spec.loader.exec_module(w)

CELLAR = (9, 10, 11, 16)


def cellar(snap) -> bool:
    return snap.mode in CELLAR or snap.screen == 0x07


def walk(env, assist, total, axis, tgt, max_f=320) -> dict:
    ok = w.walk_axis(env, assist, total, axis, tgt, max_f=max_f)
    snap = read_snapshot(env.get_ram())
    rec = {
        "a": axis, "t": tgt, "ok": ok,
        "xy": [snap.link_x, snap.link_y],
        "tile": int(snap.colliding_tile),
        "stair": bool(on_stair_tile(snap)),
        "mode": snap.mode,
    }
    print("WALK", rec, flush=True)
    return rec


def nudge_stairs(env, assist, total) -> bool:
    snap = read_snapshot(env.get_ram())
    if cellar(snap):
        return True
    if on_stair_tile(snap) and snap.colliding_tile != 0x24:
        for d in ("UP", "DOWN", "LEFT", "RIGHT"):
            push_dir(env, assist, total, d, frames=80)
            if cellar(read_snapshot(env.get_ram())):
                return True
    return cellar(read_snapshot(env.get_ram()))


def try_path(env, assist, total, name, steps) -> dict:
    print("PATH", name, flush=True)
    log = []
    for axis, tgt in steps:
        log.append(walk(env, assist, total, axis, tgt))
        snap = read_snapshot(env.get_ram())
        if cellar(snap) or (on_stair_tile(snap) and snap.colliding_tile != 0x24):
            nudge_stairs(env, assist, total)
            break
    snap = read_snapshot(env.get_ram())
    return {"name": name, "log": log, "cellar": cellar(snap), "xy": [snap.link_x, snap.link_y], "tile": int(snap.colliding_tile), "mode": snap.mode}


def main() -> dict:
    configure_headless()
    RECORDINGS_DIR.mkdir(parents=True, exist_ok=True)
    env, assist, _ = w.open_env("Level5Cleared64")
    total = [1]
    tries = []
    try:
        idle(env, assist, total, 12)
        start = w.dump_live(read_snapshot(env.get_ram()), env.get_ram())
        print("START", [start.get("x"), start.get("y")], flush=True)

        # A: north corridor to x=120 then south into diamond
        tries.append(try_path(env, assist, total, "A_n120_south", (
            ("y", 93), ("x", 120), ("y", 125), ("y", 141), ("x", 120),
        )))
        if cellar(read_snapshot(env.get_ram())):
            return finish(env, assist, total, tries, start, "A")

        # B: north to x=104 (west of north block) then SE into center
        tries.append(try_path(env, assist, total, "B_n104_se", (
            ("y", 93), ("x", 104), ("y", 125), ("x", 120), ("y", 141),
        )))
        if cellar(read_snapshot(env.get_ram())):
            return finish(env, assist, total, tries, start, "B")

        # C: north to x=136 then SW into center
        tries.append(try_path(env, assist, total, "C_n136_sw", (
            ("y", 93), ("x", 136), ("y", 125), ("x", 120), ("y", 141),
        )))
        if cellar(read_snapshot(env.get_ram())):
            return finish(env, assist, total, tries, start, "C")

        # Reset to west via north, then get EAST of left block and push LEFT.
        tries.append(try_path(env, assist, total, "D_west_then_east_of_block", (
            ("y", 93), ("x", 64), ("y", 160), ("x", 112), ("y", 144),
        )))
        snap = read_snapshot(env.get_ram())
        print("PREPUSH", [snap.link_x, snap.link_y], "blocks", [(o.x, o.y) for o in snap.objects if 1 <= o.slot <= 12 and o.type_id == 0x68], flush=True)
        push_dir(env, assist, total, "LEFT", frames=120)
        idle(env, assist, total, 12)
        snap = read_snapshot(env.get_ram())
        blocks = [(o.x, o.y) for o in snap.objects if 1 <= o.slot <= 12 and o.type_id == 0x68]
        print("PUSHED_LEFT", [snap.link_x, snap.link_y], "tile", snap.colliding_tile, "blocks", blocks, "stair", on_stair_tile(snap), flush=True)
        w.shot(env, assist, total, "l5_64_pushed_left")
        tries.append({"name": "D_push_left", "xy": [snap.link_x, snap.link_y], "tile": int(snap.colliding_tile), "blocks": blocks, "stair": bool(on_stair_tile(snap)), "mode": snap.mode})

        # Walk onto center / old block tile
        for tx, ty in ((96, 144), (96, 141), (104, 141), (112, 141), (120, 141), (120, 144), (128, 141), (88, 144), (80, 144)):
            for _ in range(180):
                snap = read_snapshot(env.get_ram())
                if cellar(snap):
                    break
                frame = walk_to_step(snap, tx, ty, y_first=True, tol=2)
                if frame.reason == "walk_arrived":
                    idle(env, assist, total, 6)
                    break
                w.step(env, assist, total, frame.action)
            snap = read_snapshot(env.get_ram())
            print("STAND", [tx, ty], "at", [snap.link_x, snap.link_y], "tile", snap.colliding_tile, "stair", on_stair_tile(snap), flush=True)
            if cellar(snap) or (on_stair_tile(snap) and snap.colliding_tile != 0x24):
                nudge_stairs(env, assist, total)
                break
            for d in ("UP", "DOWN", "LEFT", "RIGHT"):
                push_dir(env, assist, total, d, frames=40)
                if cellar(read_snapshot(env.get_ram())):
                    break
            if cellar(read_snapshot(env.get_ram())):
                break

        return finish(env, assist, total, tries, start, "D")
    finally:
        env.close()


def finish(env, assist, total, tries, start, via) -> dict:
    w.wait_play(env, assist, total, max_f=200)
    idle(env, assist, total, 12)
    snap = read_snapshot(env.get_ram())
    dest = w.dump_live(snap, env.get_ram())
    png = w.shot(env, assist, total, "l5_64_nav2")
    ok = cellar(snap)
    body = {
        "ok": ok,
        "failed_room": None if ok else "0x64",
        "via": via,
        "pokes": False,
        "status_claim": None,
        "start": [start.get("x"), start.get("y")],
        "tries": tries,
        "dump": dest,
        "dest": dest_report(snap),
        "screenshot": png,
        "whistle_0x065C": dest.get("whistle_0x065C"),
    }
    w.write_dump("l5_64_nav2", body)
    if ok:
        w.dump_and_save_room(env, assist, total, "l5_07_arrive", "Level5Entered07", "Level5Cleared64", f"0x64 stairs {via}")
    print("DEST", dest.get("room_hex"), dest.get("mode"), [dest.get("x"), dest.get("y")], "ok", ok, "via", via, flush=True)
    return body


if __name__ == "__main__":
    r = main()
    print("OK", r.get("ok"), "VIA", r.get("via"), "FAILED", r.get("failed_room"))
