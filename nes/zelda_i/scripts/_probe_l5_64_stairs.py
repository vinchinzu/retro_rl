"""L5 0x64 stairs: kill 5 Blue Darknuts, push left 0x68, corner-walk center stairs.

Resume from Level5Entered64. Dest must be cellar 0x07 (mode 9/10/11/16).
Door-hole tile 0x24 is NOT stairs. No pokes. No Clean STATUS.
"""
from __future__ import annotations

import importlib.util
from pathlib import Path

from retro_harness.nes import nes_action
from retro_harness.segment_runner import configure_headless, write_json_report
from zelda_i.dungeon_ops import idle, push_dir
from zelda_i.level9_stairs import (
    CELLAR_MODE,
    ITEM_CELLAR_MODE,
    STAIR_STANDS,
    dest_report,
    on_stair_tile,
    stair_transition_modes,
    walk_to_step,
)
from zelda_i.paths import RECORDINGS_DIR
from zelda_i.ram import PLAY_MODE, read_snapshot, read_u8, ADDR_WHISTLE

HERE = Path(__file__).resolve()
_spec = importlib.util.spec_from_file_location("l5whistle", HERE.parent / "_probe_l5_whistle_path.py")
w = importlib.util.module_from_spec(_spec)
assert _spec.loader is not None
_spec.loader.exec_module(w)

ROOM_64 = 0x64
ROOM_07 = 0x07
CELLAR_MODES = (9, 10, 11, 16)

# Diamond corner approaches. Cardinals hit the four blocks; corners slip in.
CORNER_PATHS = (
    ("se", (("y", 165), ("x", 144), ("y", 157), ("x", 128), ("y", 149), ("x", 120), ("y", 141))),
    ("sw", (("y", 165), ("x", 96), ("y", 157), ("x", 112), ("y", 149), ("x", 120), ("y", 141))),
    ("ne", (("y", 117), ("x", 144), ("y", 125), ("x", 128), ("y", 133), ("x", 120), ("y", 141))),
    ("nw", (("y", 117), ("x", 96), ("y", 125), ("x", 112), ("y", 133), ("x", 120), ("y", 141))),
)

# After push, also try walking onto known stair stands (center of diamond).
CENTER_STANDS = (
    (120, 141),
    (128, 141),
    (112, 141),
    (120, 144),
    (120, 137),
    (124, 141),
    (116, 141),
    (120, 149),
    (120, 133),
    (104, 141),
    (136, 141),
) + tuple(STAIR_STANDS[:8])


def in_cellar(snap) -> bool:
    if snap.mode in CELLAR_MODES:
        return True
    if stair_transition_modes(snap.mode) and snap.screen in (ROOM_07, ROOM_64):
        return True
    return False


def on_center_stairs(snap) -> bool:
    if not on_stair_tile(snap):
        return False
    # Exclude east bomb-hole 0x24 and door columns.
    if snap.colliding_tile == 0x24:
        return False
    return 80 < snap.link_x < 160 and 120 < snap.link_y < 165


def walk_path(env, assist, total, steps) -> dict:
    log = []
    for axis, tgt in steps:
        ok = w.walk_axis(env, assist, total, axis, tgt, max_f=280)
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
        log.append(rec)
        if in_cellar(snap) or on_center_stairs(snap):
            break
    return {"log": log, "end": w.dump_live(read_snapshot(env.get_ram()), env.get_ram())}


def push_left_block(env, assist, total) -> dict:
    """Wiki: after all-dead, push the leftmost diamond block (live 0x68 @ 96,144)."""
    snap = read_snapshot(env.get_ram())
    blocks = [
        o for o in snap.objects
        if 1 <= o.slot <= 12 and o.type_id == 0x68
    ]
    log = []
    stands = [(112, 144, "LEFT"), (96, 160, "UP"), (80, 144, "RIGHT"), (96, 128, "DOWN")]
    if blocks:
        bx, by = blocks[0].x, blocks[0].y
        stands = [
            (bx + 16, by, "LEFT"),
            (bx, by + 16, "UP"),
            (bx - 16, by, "RIGHT"),
            (bx, by - 16, "DOWN"),
        ] + stands
    for tx, ty, direction in stands:
        w.walk_axis(env, assist, total, "y", ty, max_f=240)
        w.walk_axis(env, assist, total, "x", tx, max_f=240)
        push_dir(env, assist, total, direction, frames=100)
        idle(env, assist, total, 10)
        snap = read_snapshot(env.get_ram())
        rec = {
            "stand": [tx, ty],
            "dir": direction,
            "xy": [snap.link_x, snap.link_y],
            "tile": int(snap.colliding_tile),
            "stair": bool(on_stair_tile(snap)),
            "mode": snap.mode,
            "room": f"0x{snap.screen:02x}",
            "blocks": [{"slot": b.slot, "x": b.x, "y": b.y} for b in [
                o for o in snap.objects if 1 <= o.slot <= 12 and o.type_id == 0x68
            ]],
        }
        log.append(rec)
        if in_cellar(snap) or on_center_stairs(snap):
            break
    return {
        "blocks_in": [{"slot": b.slot, "x": b.x, "y": b.y} for b in blocks],
        "log": log,
        "end": w.dump_live(read_snapshot(env.get_ram()), env.get_ram()),
    }


def hunt_center_stairs(env, assist, total) -> dict:
    hits = []
    for name, steps in CORNER_PATHS:
        snap = read_snapshot(env.get_ram())
        if in_cellar(snap):
            return {"took": True, "via": name, "hits": hits, "dest": w.dump_live(snap, env.get_ram())}
        walked = walk_path(env, assist, total, steps)
        hits.append({"path": name, "walk": walked["log"]})
        snap = read_snapshot(env.get_ram())
        if in_cellar(snap) or on_center_stairs(snap):
            break
    for tx, ty in CENTER_STANDS:
        snap = read_snapshot(env.get_ram())
        if in_cellar(snap):
            break
        for _ in range(240):
            snap = read_snapshot(env.get_ram())
            if in_cellar(snap) or on_center_stairs(snap):
                break
            frame = walk_to_step(snap, tx, ty, y_first=True, tol=2)
            if frame.reason == "walk_arrived":
                idle(env, assist, total, 8)
                break
            w.step(env, assist, total, frame.action)
        snap = read_snapshot(env.get_ram())
        hits.append({
            "stand": [tx, ty],
            "xy": [snap.link_x, snap.link_y],
            "tile": int(snap.colliding_tile),
            "stair": bool(on_stair_tile(snap)),
            "mode": snap.mode,
            "room": f"0x{snap.screen:02x}",
        })
        if in_cellar(snap) or on_center_stairs(snap):
            break
        for direction in ("UP", "DOWN", "LEFT", "RIGHT"):
            for _ in range(12):
                snap = read_snapshot(env.get_ram())
                if in_cellar(snap):
                    break
                w.step(env, assist, total, nes_action(direction))
            if in_cellar(read_snapshot(env.get_ram())):
                break
        if in_cellar(read_snapshot(env.get_ram())):
            break
    snap = read_snapshot(env.get_ram())
    return {
        "took": in_cellar(snap) or on_center_stairs(snap),
        "hits": hits,
        "end": w.dump_live(snap, env.get_ram()),
    }


def trigger_stairs(env, assist, total) -> dict:
    """If on stair tile, nudge into the warp. Then wait for cellar."""
    snap = read_snapshot(env.get_ram())
    if in_cellar(snap):
        w.wait_play(env, assist, total, max_f=280)
        idle(env, assist, total, 16)
        snap = read_snapshot(env.get_ram())
        return {"already": True, "dump": w.dump_live(snap, env.get_ram())}
    for direction in ("UP", "DOWN", "LEFT", "RIGHT"):
        push_dir(env, assist, total, direction, frames=80)
        idle(env, assist, total, 8)
        snap = read_snapshot(env.get_ram())
        if in_cellar(snap):
            break
    w.wait_play(env, assist, total, max_f=280)
    idle(env, assist, total, 20)
    snap = read_snapshot(env.get_ram())
    return {"already": False, "dump": w.dump_live(snap, env.get_ram())}


def main() -> dict:
    configure_headless()
    RECORDINGS_DIR.mkdir(parents=True, exist_ok=True)
    env, assist, _obs = w.open_env("Level5Entered64")
    total = [1]
    try:
        idle(env, assist, total, 16)
        snap = read_snapshot(env.get_ram())
        start = w.dump_live(snap, env.get_ram())
        print(
            "START64",
            start.get("room_hex"),
            "mode",
            start.get("mode"),
            "xy",
            [start.get("x"), start.get("y")],
            "bombs",
            start.get("bombs"),
            "n_dn",
            len(w.live_darknuts(snap)),
            flush=True,
        )
        if snap.screen != ROOM_64 or snap.mode != PLAY_MODE:
            rec = {"ok": False, "failed_room": "0x64", "reason": "entered64_not_play_0x64", "start": start}
            w.write_dump("l5_64_stairs", rec)
            return rec

        n_dn = len(w.live_darknuts(snap))
        fight = None
        if n_dn:
            fight = w.fight_darknuts(env, assist, total, ROOM_64, expected=max(5, n_dn), source=0x65)
            idle(env, assist, total, 20)
            print("FIGHT64", fight.get("ok"), "end_n", fight.get("end_n"), "start", fight.get("start_n"), flush=True)
            snap = read_snapshot(env.get_ram())
            if snap.screen != ROOM_64:
                rec = {
                    "ok": False,
                    "failed_room": "0x64",
                    "reason": "fight_left_0x64",
                    "fight": {k: fight[k] for k in fight if k != "controller"},
                    "now": w.dump_live(snap, env.get_ram()),
                }
                w.write_dump("l5_64_stairs", rec)
                return rec
            if w.live_darknuts(snap):
                rec = {
                    "ok": False,
                    "failed_room": "0x64",
                    "reason": "darknuts_not_cleared",
                    "fight": {k: fight[k] for k in fight if k != "controller"},
                    "now": w.dump_live(snap, env.get_ram()),
                }
                w.write_dump("l5_64_stairs", rec)
                return rec
        cleared = w.dump_and_save_room(
            env, assist, total, "l5_64_cleared", "Level5Cleared64", "Level5Entered64", "0x64 5/5 darknuts"
        )

        pushed = push_left_block(env, assist, total)
        print("PUSH64", "blocks", pushed.get("blocks_in"), "end_xy", (pushed.get("end") or {}).get("x"), (pushed.get("end") or {}).get("y"), flush=True)
        snap = read_snapshot(env.get_ram())
        hunt = None
        if not in_cellar(snap):
            hunt = hunt_center_stairs(env, assist, total)
            print("HUNT64 took", hunt.get("took"), "hits", len(hunt.get("hits") or []), flush=True)
        trig = trigger_stairs(env, assist, total)
        snap = read_snapshot(env.get_ram())
        dest = w.dump_live(snap, env.get_ram())
        png = w.shot(env, assist, total, "l5_64_stairs")
        ok = in_cellar(snap) or snap.screen == ROOM_07
        body = {
            "ok": ok,
            "failed_room": None if ok else "0x64",
            "reason": None if ok else "stairs_in_0x64_not_taken",
            "pokes": False,
            "status_claim": None,
            "start": start,
            "fight": None if fight is None else {k: fight[k] for k in fight if k != "controller"},
            "cleared": cleared.get("dump"),
            "push": {k: v for k, v in pushed.items() if k != "log"},
            "push_log": pushed.get("log"),
            "hunt": None if hunt is None else {k: v for k, v in hunt.items() if k != "hits"},
            "hunt_hits": None if hunt is None else hunt.get("hits"),
            "trigger": trig,
            "dest": dest_report(snap),
            "dump": dest,
            "screenshot": png,
            "whistle_0x065C": dest.get("whistle_0x065C"),
            "rom": w.rom_room(int(snap.screen)),
        }
        w.write_dump("l5_64_stairs", body)
        if ok:
            w.dump_and_save_room(
                env, assist, total, "l5_07_arrive", "Level5Entered07", "Level5Entered64", "0x64 stairs"
            )
        print(
            "DEST64",
            dest.get("room_hex"),
            "mode",
            dest.get("mode"),
            "xy",
            [dest.get("x"), dest.get("y")],
            "ok",
            ok,
            flush=True,
        )
        return body
    finally:
        env.close()


if __name__ == "__main__":
    r = main()
    print("OK", r.get("ok"), "FAILED_ROOM", r.get("failed_room"), "DEST", (r.get("dump") or {}).get("room_hex"), (r.get("dump") or {}).get("mode"))
