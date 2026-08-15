"""From Level5Whistle: 04→05→06 stairs→07, dump both mouths."""
from retro_harness.env import make_env, reset_obs
from retro_harness.nes import nes_action, nes_idle_action
from retro_harness.segment_runner import configure_headless, save_rgb_png, write_json_report
from zelda_i.assist import UnlimitedHealthAssist
from zelda_i.dungeon_ops import idle, push_dir
from zelda_i.level5_path import exit_whistle_04, take_block_stairs_06, walk_axis, walk_east_from_05
from zelda_i.paths import GAME, GAME_DIR, RECORDINGS_DIR
from zelda_i.ram import ADDR_WHISTLE, PLAY_MODE, read_snapshot, read_u8

def dump(env):
    s = read_snapshot(env.get_ram())
    return {
        "sc": f"0x{s.screen:02x}",
        "mode": s.mode,
        "xy": [s.link_x, s.link_y],
        "tile": int(s.colliding_tile),
        "whistle": int(read_u8(env.get_ram(), ADDR_WHISTLE)),
        "keys": int(s.keys),
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
        st = take_block_stairs_06(env, assist, n)
        idle(env, assist, n, 8)
        log.append({"tag": "in07", "stairs": {k: st[k] for k in st if k != "log"}, **dump(env)})
        print("IN07", log[-1], flush=True)
        save_rgb_png(env.step(nes_idle_action())[0], RECORDINGS_DIR / "l5_07_return_arrive.png")

        # Floor, then try LEFT, record, if back in 0x06 abort this mouth.
        walk_axis(env, assist, n, "y", 189, max_f=400)
        log.append({"tag": "floor", **dump(env)})
        print("FLOOR", log[-1], flush=True)
        walk_axis(env, assist, n, "x", 48, max_f=500)
        log.append({"tag": "left_col", **dump(env)})
        print("LEFTCOL", log[-1], flush=True)
        walk_axis(env, assist, n, "y", 65, max_f=400)
        log.append({"tag": "left_climb", **dump(env)})
        print("LEFTCLIMB", log[-1], flush=True)
        push_dir(env, assist, n, "UP", frames=240)
        idle(env, assist, n, 16)
        for _ in range(200):
            s = read_snapshot(env.get_ram())
            if s.mode == PLAY_MODE and s.screen != 0x07:
                break
            env.step(nes_action("UP"))
            n[0] += 1
            assist.apply_env(env, frame=n[0])
        idle(env, assist, n, 12)
        log.append({"tag": "after_left", **dump(env)})
        print("AFTER_LEFT", log[-1], flush=True)
        save_rgb_png(env.step(nes_idle_action())[0], RECORDINGS_DIR / "l5_07_return_left.png")
        write_json_report(RECORDINGS_DIR / "l5_07_return_mouths.json", {"log": log, "pokes": False})
    finally:
        env.close()

if __name__ == "__main__":
    main()
