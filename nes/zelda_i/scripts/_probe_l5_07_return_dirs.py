"""From 0x07 return spawn: idle settle, try each dir, then y165 cross."""
from retro_harness.env import make_env, reset_obs
from retro_harness.nes import nes_action, nes_idle_action
from retro_harness.segment_runner import configure_headless, save_rgb_png, write_json_report
from zelda_i.assist import UnlimitedHealthAssist
from zelda_i.dungeon_ops import idle
from zelda_i.level5_path import (
    _cellar_walk_axis,
    _step,
    exit_whistle_04,
    take_block_stairs_06,
    walk_east_from_05,
)
from zelda_i.paths import GAME, GAME_DIR, RECORDINGS_DIR
from zelda_i.ram import ADDR_WHISTLE, PLAY_MODE, read_snapshot, read_u8

def dump(env):
    s = read_snapshot(env.get_ram())
    return {
        "sc": f"0x{s.screen:02x}",
        "mode": s.mode,
        "sub": int(s.submode),
        "xy": [s.link_x, s.link_y],
        "tile": int(s.colliding_tile),
        "whistle": int(read_u8(env.get_ram(), ADDR_WHISTLE)),
    }

def main():
    configure_headless()
    env = make_env(GAME, "Level5Whistle", GAME_DIR, render_mode="rgb_array")
    assist = UnlimitedHealthAssist(enabled=True)
    n = [1]
    log = []
    try:
        reset_obs(env)
        env.step(nes_idle_action())
        assist.apply_env(env, frame=0)
        idle(env, assist, n, 8)
        exit_whistle_04(env, assist, n)
        walk_east_from_05(env, assist, n)
        take_block_stairs_06(env, assist, n)
        log.append({"tag": "arrive", **dump(env)})
        print("ARRIVE", log[-1], flush=True)
        for i in range(8):
            idle(env, assist, n, 20)
            log.append({"tag": f"idle{i}", **dump(env)})
            print("IDLE", log[-1], flush=True)
        save_rgb_png(env.step(nes_idle_action())[0], RECORDINGS_DIR / "l5_07_return_settle.png")

        # Fresh dirs from a saved mental start: try RIGHT first (off stairs toward 0x06 mouth)
        for d in ("RIGHT", "LEFT", "DOWN", "UP"):
            s0 = dump(env)
            for _ in range(80):
                s = read_snapshot(env.get_ram())
                if s.mode == PLAY_MODE and s.screen != 0x07:
                    break
                _step(env, assist, n, nes_action(d))
            rec = {"tag": f"dir_{d}", "from": s0, **dump(env)}
            log.append(rec)
            print("DIR", rec, flush=True)
            if read_snapshot(env.get_ram()).mode == PLAY_MODE:
                break

        if read_snapshot(env.get_ram()).mode != PLAY_MODE:
            ok = _cellar_walk_axis(env, assist, n, "y", 165, max_f=500)
            log.append({"tag": "y165", "ok": ok, **dump(env)})
            print("Y165", log[-1], flush=True)
            ok = _cellar_walk_axis(env, assist, n, "x", 48, max_f=600)
            log.append({"tag": "x48", "ok": ok, **dump(env)})
            print("X48", log[-1], flush=True)
            for _ in range(240):
                s = read_snapshot(env.get_ram())
                if s.mode == PLAY_MODE and s.screen != 0x07:
                    break
                _step(env, assist, n, nes_action("UP"))
            idle(env, assist, n, 16)
            log.append({"tag": "after_up", **dump(env)})
            print("AFTER_UP", log[-1], flush=True)

        save_rgb_png(env.step(nes_idle_action())[0], RECORDINGS_DIR / "l5_07_return_dirs.png")
        write_json_report(RECORDINGS_DIR / "l5_07_return_dirs.json", {"log": log, "pokes": False})
    finally:
        env.close()

if __name__ == "__main__":
    main()
