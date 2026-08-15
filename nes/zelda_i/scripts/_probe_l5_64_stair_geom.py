"""Probe 0x64 center-stairs approach columns from Level5Entered64. No pokes."""
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


def try_path(name, steps):
    env, assist, obs = open_env()
    total = [1]
    try:
        idle(env, assist, total, 12)
        log = []
        for axis, tgt in steps:
            ok, snap = walk_axis(env, assist, total, axis, tgt)
            log.append(rec(snap, axis=axis, tgt=tgt, ok=ok))
            if snap.screen != 0x64 or stair_transition_modes(snap.mode):
                break
        # nudge UP/DOWN
        for direction in ("UP", "DOWN"):
            for _ in range(40):
                snap = read_snapshot(env.get_ram())
                if stair_transition_modes(snap.mode) or snap.screen != 0x64:
                    break
                step(env, assist, total, nes_action(direction))
            snap = read_snapshot(env.get_ram())
            log.append(rec(snap, nudge=direction))
            if stair_transition_modes(snap.mode) or snap.screen != 0x64:
                break
        snap = read_snapshot(env.get_ram())
        png = RECORDINGS_DIR / f"l5_64_geom_{name}.png"
        obs, *_ = env.step(nes_idle_action())
        save_rgb_png(obs, png)
        out = {
            "name": name,
            "steps": log,
            "end": rec(snap),
            "took": stair_transition_modes(snap.mode) or (snap.screen != 0x64 and snap.screen != 0x65),
            "screenshot": str(png),
        }
        print(name, "took", out["took"], "end", out["end"], flush=True)
        return out
    finally:
        env.close()


def main():
    configure_headless()
    paths = [
        ("south120", (("y", 173), ("x", 120), ("y", 141))),
        ("south112", (("y", 173), ("x", 112), ("y", 141))),
        ("south128", (("y", 173), ("x", 128), ("y", 141))),
        ("south104", (("y", 173), ("x", 104), ("y", 141))),
        ("south136", (("y", 173), ("x", 136), ("y", 141))),
        ("south96", (("y", 173), ("x", 96), ("y", 141))),
        ("south144", (("y", 173), ("x", 144), ("y", 141))),
        ("mid160", (("y", 141), ("x", 160), ("x", 128))),
        ("mid80", (("y", 141), ("x", 80), ("x", 112))),
        ("north120", (("y", 109), ("x", 120), ("y", 141))),
        ("north112", (("y", 109), ("x", 112), ("y", 141))),
        ("north128", (("y", 109), ("x", 128), ("y", 141))),
        ("se_diag", (("y", 173), ("x", 160), ("y", 157), ("x", 128), ("y", 141))),
        ("sw_diag", (("y", 173), ("x", 80), ("y", 157), ("x", 112), ("y", 141))),
    ]
    results = [try_path(n, s) for n, s in paths]
    hits = [r for r in results if r["took"] or r["end"].get("stair") or r["end"]["xy"][1] < 165]
    write_json_report(RECORDINGS_DIR / "l5_64_stair_geom.json", {"results": results, "near": hits})
    print("NEAR", [(r["name"], r["end"]) for r in hits], flush=True)


if __name__ == "__main__":
    main()
