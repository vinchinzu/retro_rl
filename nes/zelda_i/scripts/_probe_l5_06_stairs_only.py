"""Reload-per-path 0x06 center-stairs hunt. 0x64 diamond: y=117, x=120, y=141, RIGHT."""
from __future__ import annotations

from retro_harness.env import make_env, reset_obs
from retro_harness.nes import nes_action, nes_idle_action
from retro_harness.segment_runner import configure_headless, write_json_report
from zelda_i.assist import UnlimitedHealthAssist
from zelda_i.dungeon_ops import idle, push_dir
from zelda_i.level5_path import walk_axis
from zelda_i.level9_stairs import on_stair_tile, stair_transition_modes
from zelda_i.paths import GAME, GAME_DIR, RECORDINGS_DIR
from zelda_i.ram import ADDR_WHISTLE, PLAY_MODE, read_snapshot, read_u8
from zelda_i.scripts._probe_l5_whistle_path import (
    dest_report,
    dump_and_save_room,
    dump_live,
    shot,
    step,
    wait_play,
    write_dump,
)

STATE = "Level5Whistle06"
ROOM = 0x06


def open_env(state=STATE):
    env = make_env(GAME, state, GAME_DIR, render_mode="rgb_array")
    assist = UnlimitedHealthAssist(enabled=True)
    reset_obs(env)
    env.step(nes_idle_action())
    assist.apply_env(env, frame=0)
    return env, assist


def in_cellar(snap) -> bool:
    if stair_transition_modes(snap.mode):
        return True
    return snap.level == 5 and snap.screen == 0x07


def rec(snap, **extra):
    d = {
        "xy": [snap.link_x, snap.link_y],
        "tile": int(snap.colliding_tile),
        "stair": bool(on_stair_tile(snap)),
        "mode": snap.mode,
        "room": f"0x{snap.screen:02x}",
    }
    d.update(extra)
    return d


def try_path(name, steps, nudges=("DOWN", "RIGHT", "UP")):
    env, assist = open_env()
    total = [1]
    try:
        idle(env, assist, total, 10)
        start = rec(read_snapshot(env.get_ram()))
        print("PATH", name, "start", start, "whistle", int(read_u8(env.get_ram(), ADDR_WHISTLE)), flush=True)
        log = [start]
        for axis, tgt in steps:
            ok = walk_axis(env, assist, total, axis, tgt, max_f=400)
            snap = read_snapshot(env.get_ram())
            log.append(rec(snap, axis=axis, tgt=tgt, ok=ok))
            print("  STEP", log[-1], flush=True)
            if in_cellar(snap) or (snap.screen != ROOM and snap.mode in (PLAY_MODE, 9, 10, 11, 16)):
                break
        for direction in nudges:
            for _ in range(36):
                snap = read_snapshot(env.get_ram())
                if in_cellar(snap) or snap.screen != ROOM:
                    break
                if snap.link_y >= 200:
                    step(env, assist, total, nes_action("UP"))
                    continue
                if snap.link_x <= 24:
                    step(env, assist, total, nes_action("RIGHT"))
                    continue
                if snap.link_x > 200:
                    step(env, assist, total, nes_action("LEFT"))
                    continue
                step(env, assist, total, nes_action(direction))
            snap = read_snapshot(env.get_ram())
            log.append(rec(snap, nudge=direction))
            print("  NUDGE", log[-1], flush=True)
            if in_cellar(snap) or snap.screen != ROOM:
                break
        for _ in range(180):
            snap = read_snapshot(env.get_ram())
            if in_cellar(snap) or (snap.mode == PLAY_MODE and snap.screen != ROOM):
                break
            step(env, assist, total, nes_idle_action())
        idle(env, assist, total, 12)
        snap = read_snapshot(env.get_ram())
        png = shot(env, assist, total, f"l5_06_p_{name}")
        took = in_cellar(snap) or (snap.screen != ROOM and snap.screen not in (0x05, 0x16))
        out = {"name": name, "start": start, "log": log, "end": rec(snap), "took": took, "png": png}
        if took:
            dump_and_save_room(env, assist, total, "l5_07_from06", "Level5Whistle07", STATE, f"0x06 {name}")
        print("END", name, "took", took, out["end"], flush=True)
        return out
    finally:
        env.close()


def main():
    configure_headless()
    RECORDINGS_DIR.mkdir(parents=True, exist_ok=True)
    paths = [
        ("n117_r120_d141", (("y", 117), ("x", 120), ("y", 141))),
        ("n109_r120_d141", (("y", 109), ("x", 120), ("y", 141))),
        ("n93_r120_d141", (("y", 93), ("x", 120), ("y", 141))),
        ("n117_r128_d141", (("y", 117), ("x", 128), ("y", 141))),
        ("n117_r112_d141", (("y", 117), ("x", 112), ("y", 141))),
        ("s189_l80_n141_r120", (("y", 189), ("x", 80), ("y", 141), ("x", 120))),
        ("s189_l64_n141_r120", (("y", 189), ("x", 64), ("y", 141), ("x", 120))),
        ("off48_n117_r120_d141", (("x", 48), ("y", 117), ("x", 120), ("y", 141))),
        ("off48_n109_r120_d141", (("x", 48), ("y", 109), ("x", 120), ("y", 141))),
        ("n117_r104_d133_r120", (("y", 117), ("x", 104), ("y", 133), ("x", 120))),
        ("n125_r120", (("y", 125), ("x", 120))),
        ("n117_r136_d141", (("y", 117), ("x", 136), ("y", 141))),
        ("s189_r160_n141_l120", (("y", 189), ("x", 160), ("y", 141), ("x", 120))),
        ("n93_r80_d141_r120", (("y", 93), ("x", 80), ("y", 141), ("x", 120))),
    ]
    results = []
    hit = None
    for name, steps in paths:
        out = try_path(name, steps)
        results.append({k: out[k] for k in out if k != "log"})
        results[-1]["log"] = out["log"]
        if out["took"]:
            hit = out
            break
    body = {
        "ok": hit is not None,
        "hit": None if hit is None else {k: hit[k] for k in hit if k != "log"},
        "results": results,
        "pokes": False,
        "status_claim": None,
        "l6_l8": False,
    }
    write_json_report(RECORDINGS_DIR / "l5_06_stairs_only.json", body)
    print("OK", body["ok"], "hit", None if hit is None else hit["name"], flush=True)


if __name__ == "__main__":
    main()
