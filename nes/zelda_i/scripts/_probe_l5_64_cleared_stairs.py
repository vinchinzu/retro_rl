"""From Level5Cleared64 (no enemies): walk into diamond to center stairs."""
from retro_harness.env import make_env, reset_obs
from retro_harness.nes import nes_action, nes_idle_action
from retro_harness.segment_runner import configure_headless, save_rgb_png, write_json_report
from zelda_i.assist import UnlimitedHealthAssist
from zelda_i.dungeon_ops import idle
from zelda_i.level9_stairs import on_stair_tile, stair_transition_modes
from zelda_i.paths import GAME, GAME_DIR, RECORDINGS_DIR
from zelda_i.ram import PLAY_MODE, read_snapshot
from zelda_i.scripts._probe_l5_whistle_path import step, walk_axis

STATE = "Level5Cleared64"


def open_env():
    env = make_env(GAME, STATE, GAME_DIR, render_mode="rgb_array")
    assist = UnlimitedHealthAssist(enabled=True)
    obs, _ = reset_obs(env)
    obs, *_ = env.step(nes_idle_action())
    assist.apply_env(env, frame=0)
    return env, assist, obs


def rec(snap, **extra):
    d = {"xy": [snap.link_x, snap.link_y], "tile": int(snap.colliding_tile), "stair": bool(on_stair_tile(snap)), "mode": snap.mode, "room": f"0x{snap.screen:02x}"}
    d.update(extra)
    return d


def try_path(name, steps):
    env, assist, obs = open_env()
    total = [1]
    try:
        idle(env, assist, total, 8)
        log = []
        for axis, tgt in steps:
            ok = walk_axis(env, assist, total, axis, tgt, max_f=350)
            snap = read_snapshot(env.get_ram())
            log.append(rec(snap, axis=axis, tgt=tgt, ok=ok))
            if stair_transition_modes(snap.mode) or (snap.screen != 0x64 and snap.mode == PLAY_MODE):
                break
        for direction in ("DOWN", "RIGHT", "UP", "LEFT"):
            for _ in range(40):
                snap = read_snapshot(env.get_ram())
                if stair_transition_modes(snap.mode) or (snap.screen != 0x64 and snap.mode == PLAY_MODE):
                    break
                if snap.link_x > 190:
                    step(env, assist, total, nes_action("LEFT"))
                    continue
                step(env, assist, total, nes_action(direction))
            snap = read_snapshot(env.get_ram())
            log.append(rec(snap, nudge=direction))
            if stair_transition_modes(snap.mode) or snap.screen != 0x64:
                break
        snap = read_snapshot(env.get_ram())
        png = RECORDINGS_DIR / f"l5_64_c_{name}.png"
        obs, *_ = env.step(nes_idle_action())
        save_rgb_png(obs, png)
        out = {"name": name, "log": log, "end": rec(snap), "took": stair_transition_modes(snap.mode) or (snap.screen not in (0x64, 0x65))}
        print(name, "took", out["took"], "end", out["end"], "steps", [(s.get("axis"), s.get("tgt"), s.get("ok"), s["xy"]) for s in log if "axis" in s], flush=True)
        return out
    finally:
        env.close()


def main():
    configure_headless()
    # Cleared64 spawn is ~ (72, 104) NW of diamond.
    paths = [
        ("r120_d141", (("x", 120), ("y", 141))),
        ("d117_r120_d141", (("y", 117), ("x", 120), ("y", 141))),
        ("d125_r120", (("y", 125), ("x", 120))),
        ("r96_d125_r120", (("x", 96), ("y", 125), ("x", 120), ("y", 141))),
        ("r104_d125_r120", (("x", 104), ("y", 125), ("x", 120))),
        ("r112_d125_r120", (("x", 112), ("y", 125), ("x", 120))),
        ("r88_d117_r104_d133_r120", (("x", 88), ("y", 117), ("x", 104), ("y", 133), ("x", 120))),
        ("r80_d141_r120", (("x", 80), ("y", 141), ("x", 120))),
        ("y93_x120_d141", (("y", 93), ("x", 120), ("y", 141))),
        ("y93_x128_d141", (("y", 93), ("x", 128), ("y", 141))),
        ("y93_x112_d141", (("y", 93), ("x", 112), ("y", 141))),
        ("d109_r136_d141", (("y", 109), ("x", 136), ("y", 141))),
        ("d109_r144_d141", (("y", 109), ("x", 144), ("y", 141))),
        ("r160_d141_l120", (("x", 160), ("y", 141), ("x", 120))),
        ("diag_r96_d117_r112_d133_r120", (("x", 96), ("y", 117), ("x", 112), ("y", 133), ("x", 120), ("y", 141))),
    ]
    results = [try_path(n, s) for n, s in paths]
    hits = [r for r in results if r["took"] or r["end"].get("stair") or (100 < r["end"]["xy"][0] < 150 and 120 < r["end"]["xy"][1] < 160)]
    write_json_report(RECORDINGS_DIR / "l5_64_cleared_stairs.json", {"results": results, "hits": hits})
    print("HITS", [(r["name"], r["end"]) for r in hits], flush=True)


if __name__ == "__main__":
    main()
