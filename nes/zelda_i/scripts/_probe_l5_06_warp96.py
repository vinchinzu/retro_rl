"""From Level5Whistle05 east into 0x06, stand emerge (96,157) and take warp."""
from __future__ import annotations

from retro_harness.env import make_env, reset_obs
from retro_harness.nes import nes_action, nes_idle_action
from retro_harness.segment_runner import configure_headless, save_rgb_png, write_json_report
from zelda_i.assist import UnlimitedHealthAssist
from zelda_i.dungeon_ops import idle, push_dir
from zelda_i.level5_path import walk_axis
from zelda_i.level9_stairs import on_stair_tile, on_warp_tile, stair_transition_modes
from zelda_i.paths import GAME, GAME_DIR, RECORDINGS_DIR
from zelda_i.ram import ADDR_WHISTLE, PLAY_MODE, read_snapshot, read_u8

STATE = "Level5Whistle05"


def step(env, assist, total, a):
    env.step(a)
    total[0] += 1
    assist.apply_env(env, frame=total[0])


def dump(env):
    s = read_snapshot(env.get_ram())
    return {
        "sc": f"0x{s.screen:02x}",
        "mode": s.mode,
        "xy": [s.link_x, s.link_y],
        "whistle": int(read_u8(env.get_ram(), ADDR_WHISTLE)),
        "stair": bool(on_stair_tile(s)),
        "warp": bool(on_warp_tile(s)),
        "tile": int(s.colliding_tile),
    }


def wait_play(env, assist, total):
    for _ in range(200):
        s = read_snapshot(env.get_ram())
        if s.mode == PLAY_MODE and not s.transitioning:
            idle(env, assist, total, 6)
            return
        step(env, assist, total, nes_action("UP"))


def main():
    configure_headless()
    env = None
    total = [1]
    log = []
    try:
        env = make_env(GAME, STATE, GAME_DIR, render_mode="rgb_array")
        assist = UnlimitedHealthAssist(enabled=True)
        reset_obs(env)
        env.step(nes_idle_action())
        assist.apply_env(env, frame=0)
        idle(env, assist, total, 12)
        walk_axis(env, assist, total, "y", 141, max_f=300)
        walk_axis(env, assist, total, "x", 224, max_f=400)
        push_dir(env, assist, total, "RIGHT", frames=200)
        idle(env, assist, total, 10)
        wait_play(env, assist, total)
        print("AT06", dump(env), flush=True)
        walk_axis(env, assist, total, "x", 64, max_f=300)
        walk_axis(env, assist, total, "y", 157, max_f=300)
        walk_axis(env, assist, total, "x", 96, max_f=300)
        walk_axis(env, assist, total, "y", 157, max_f=200)
        print("AT96157", dump(env), flush=True)
        idle(env, assist, total, 40)
        print("IDLE", dump(env), flush=True)
        for d in ("UP", "DOWN", "LEFT", "RIGHT", "UP"):
            for _ in range(12):
                s = read_snapshot(env.get_ram())
                if stair_transition_modes(s.mode) or s.screen != 0x06:
                    print("TOOK", d, dump(env), flush=True)
                    wait_play(env, assist, total)
                    write_json_report(RECORDINGS_DIR / "l5_06_warp96.json", {"ok": True, "final": dump(env), "pokes": False})
                    save_rgb_png(env.step(nes_idle_action())[0], RECORDINGS_DIR / "l5_06_warp96.png")
                    print("FINAL", dump(env), flush=True)
                    return
                step(env, assist, total, nes_action(d))
            print("AFTER", d, dump(env), flush=True)
            walk_axis(env, assist, total, "x", 96, max_f=80)
            walk_axis(env, assist, total, "y", 157, max_f=80)
        write_json_report(RECORDINGS_DIR / "l5_06_warp96.json", {"ok": False, "final": dump(env), "pokes": False})
        save_rgb_png(env.step(nes_idle_action())[0], RECORDINGS_DIR / "l5_06_warp96.png")
        print("FINAL", dump(env), flush=True)
    finally:
        if env is not None:
            env.close()


if __name__ == "__main__":
    main()
