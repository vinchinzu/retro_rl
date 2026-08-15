"""From Entered64: north-wall to (80,141), then into center stairs."""
from retro_harness.env import make_env, reset_obs
from retro_harness.nes import nes_action, nes_idle_action
from retro_harness.segment_runner import configure_headless, save_rgb_png, write_json_report
from zelda_i.assist import UnlimitedHealthAssist
from zelda_i.dungeon_ops import idle
from zelda_i.level9_stairs import on_stair_tile, stair_transition_modes
from zelda_i.paths import GAME, GAME_DIR, RECORDINGS_DIR
from zelda_i.ram import PLAY_MODE, read_snapshot
from zelda_i.scripts._probe_l5_whistle_path import step, walk_axis

STATE = "Level5Entered64"


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
        idle(env, assist, total, 10)
        log = []
        for axis, tgt in steps:
            ok = walk_axis(env, assist, total, axis, tgt, max_f=400)
            snap = read_snapshot(env.get_ram())
            log.append(rec(snap, axis=axis, tgt=tgt, ok=ok))
            if stair_transition_modes(snap.mode) or (snap.screen != 0x64 and snap.mode == PLAY_MODE):
                break
        for direction in ("DOWN", "UP", "RIGHT", "LEFT"):
            for _ in range(36):
                snap = read_snapshot(env.get_ram())
                if stair_transition_modes(snap.mode) or (snap.screen != 0x64 and snap.mode == PLAY_MODE):
                    break
                if snap.link_x > 180:
                    step(env, assist, total, nes_action("LEFT"))
                    continue
                step(env, assist, total, nes_action(direction))
            snap = read_snapshot(env.get_ram())
            log.append(rec(snap, nudge=direction))
            if stair_transition_modes(snap.mode) or snap.screen != 0x64:
                break
        snap = read_snapshot(env.get_ram())
        png = RECORDINGS_DIR / f"l5_64_80_{name}.png"
        obs, *_ = env.step(nes_idle_action())
        save_rgb_png(obs, png)
        out = {"name": name, "log": log, "end": rec(snap), "took": stair_transition_modes(snap.mode) or (snap.screen not in (0x64, 0x65))}
        print(name, "took", out["took"], "end", out["end"], "steps", [(s.get("axis"), s.get("tgt"), s.get("ok"), s["xy"]) for s in log if "axis" in s], flush=True)
        return out
    finally:
        env.close()


def main():
    configure_headless()
    base = (("y", 93), ("x", 80), ("y", 141))
    paths = [
        ("to80", base),
        ("80_x96", base + (("x", 96),)),
        ("80_x104", base + (("x", 104),)),
        ("80_x112", base + (("x", 112),)),
        ("80_x120", base + (("x", 120),)),
        ("80_y125_x120", base + (("y", 125), ("x", 120))),
        ("80_y117_x120", base + (("y", 117), ("x", 120))),
        ("80_y157_x120", base + (("y", 157), ("x", 120))),
        ("80_y109_x120", base + (("y", 109), ("x", 120), ("y", 141))),
        ("96_y125_x120", (("y", 93), ("x", 96), ("y", 125), ("x", 120), ("y", 141))),
        ("sw_32_y125_x120", (("y", 189), ("x", 32), ("y", 141), ("y", 125), ("x", 120))),
        ("sw_32_x96_y125", (("y", 189), ("x", 32), ("y", 141), ("x", 96), ("y", 125), ("x", 120))),
        ("80_x96_y157_x120", base + (("x", 96), ("y", 157), ("x", 120))),
        ("80_x96_y117_x120", base + (("x", 96), ("y", 117), ("x", 120), ("y", 141))),
    ]
    results = [try_path(n, s) for n, s in paths]
    hits = [r for r in results if r["took"] or r["end"].get("stair") or (100 < r["end"]["xy"][0] < 150 and 120 < r["end"]["xy"][1] < 165)]
    write_json_report(RECORDINGS_DIR / "l5_64_from80.json", {"results": results, "hits": hits})
    print("HITS", [(r["name"], r["end"]) for r in hits], flush=True)


if __name__ == "__main__":
    main()
