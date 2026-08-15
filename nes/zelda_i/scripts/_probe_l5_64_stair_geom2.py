"""Finer 0x64 stair-gap probe from north/east channels. No pokes."""
from retro_harness.env import make_env, reset_obs
from retro_harness.nes import nes_action, nes_idle_action
from retro_harness.segment_runner import configure_headless, save_rgb_png, write_json_report
from zelda_i.assist import UnlimitedHealthAssist
from zelda_i.dungeon_ops import idle
from zelda_i.level9_stairs import on_stair_tile, stair_transition_modes
from zelda_i.paths import GAME, GAME_DIR, RECORDINGS_DIR
from zelda_i.ram import PLAY_MODE, read_snapshot

STATE = "Level5Entered64"


def open_env():
    env = make_env(GAME, STATE, GAME_DIR, render_mode="rgb_array")
    assist = UnlimitedHealthAssist(enabled=True)
    obs, _ = reset_obs(env)
    obs, *_ = env.step(nes_idle_action())
    assist.apply_env(env, frame=0)
    return env, assist, obs


def step(env, assist, total, action):
    obs, *_ = env.step(action)
    total[0] += 1
    assist.apply_env(env, frame=total[0])
    return obs


def walk_axis(env, assist, total, axis, target, max_f=350):
    last = None
    stall = 0
    for _ in range(max_f):
        snap = read_snapshot(env.get_ram())
        if stair_transition_modes(snap.mode) or (snap.screen != 0x64 and snap.mode == PLAY_MODE):
            return True, snap
        if axis == "x":
            if abs(snap.link_x - target) <= 1:
                return True, snap
            step(env, assist, total, nes_action("RIGHT" if snap.link_x < target else "LEFT"))
        else:
            if abs(snap.link_y - target) <= 1:
                return True, snap
            step(env, assist, total, nes_action("DOWN" if snap.link_y < target else "UP"))
        snap2 = read_snapshot(env.get_ram())
        pos = (snap2.link_x, snap2.link_y)
        if pos == last:
            stall += 1
            if stall >= 28:
                return False, snap2
        else:
            stall = 0
        last = pos
    return False, read_snapshot(env.get_ram())


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


def try_path(name, steps, nudges=("DOWN", "UP")):
    env, assist, obs = open_env()
    total = [1]
    try:
        idle(env, assist, total, 10)
        log = []
        for axis, tgt in steps:
            ok, snap = walk_axis(env, assist, total, axis, tgt)
            log.append(rec(snap, axis=axis, tgt=tgt, ok=ok))
            if snap.screen != 0x64 or stair_transition_modes(snap.mode):
                break
        for direction in nudges:
            for _ in range(50):
                snap = read_snapshot(env.get_ram())
                if stair_transition_modes(snap.mode) or snap.screen != 0x64:
                    break
                step(env, assist, total, nes_action(direction))
            snap = read_snapshot(env.get_ram())
            log.append(rec(snap, nudge=direction))
            if stair_transition_modes(snap.mode) or snap.screen != 0x64:
                break
        snap = read_snapshot(env.get_ram())
        png = RECORDINGS_DIR / f"l5_64_g2_{name}.png"
        obs, *_ = env.step(nes_idle_action())
        save_rgb_png(obs, png)
        out = {
            "name": name,
            "steps": log,
            "end": rec(snap),
            "took": stair_transition_modes(snap.mode) or (snap.screen not in (0x64, 0x65) and snap.mode != 7),
        }
        print(name, "took", out["took"], "end", out["end"], "steps", [(s.get("axis"), s.get("tgt"), s.get("ok"), s["xy"]) for s in log if "axis" in s], flush=True)
        return out
    finally:
        env.close()


def main():
    configure_headless()
    xs = (80, 96, 104, 112, 120, 128, 136, 144, 160, 176)
    paths = []
    for x in xs:
        paths.append((f"n109_x{x}", (("y", 109), ("x", x), ("y", 141))))
        paths.append((f"n117_x{x}", (("y", 117), ("x", x), ("y", 141))))
        paths.append((f"e125_x{x}", (("y", 125), ("x", x))))
    paths += [
        ("e_y125_x120", (("y", 125), ("x", 120), ("y", 141))),
        ("e_y117_x120", (("y", 117), ("x", 120), ("y", 141))),
        ("e_y133_x120", (("y", 133), ("x", 120))),
        ("e_y149_x120", (("y", 149), ("x", 120))),
        ("e_y157_x120", (("y", 157), ("x", 120))),
        ("e_y165_x120", (("y", 165), ("x", 120))),
    ]
    results = [try_path(n, s) for n, s in paths]
    hits = [
        r
        for r in results
        if r["took"]
        or r["end"].get("stair")
        or (r["end"]["xy"][0] < 170 and 115 < r["end"]["xy"][1] < 170)
    ]
    write_json_report(RECORDINGS_DIR / "l5_64_stair_geom2.json", {"results": results, "near": hits})
    print("HITS", len(hits), [(r["name"], r["end"]) for r in hits], flush=True)


if __name__ == "__main__":
    main()
