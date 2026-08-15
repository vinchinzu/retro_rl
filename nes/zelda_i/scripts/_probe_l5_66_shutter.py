"""At 0x66 via whistle return: try south-reenter then north shutter."""
from retro_harness.env import make_env, reset_obs
from retro_harness.nes import nes_idle_action
from retro_harness.segment_runner import configure_headless, save_rgb_png, write_json_report
from zelda_i.assist import UnlimitedHealthAssist
from zelda_i.dungeon_ops import idle, push_dir
from zelda_i.level5_path import (
    bomb_east_from_65,
    cellar_07_to_64,
    exit_whistle_04,
    take_block_stairs_06,
    walk_axis,
    walk_east_from_05,
)
from zelda_i.paths import GAME, GAME_DIR, RECORDINGS_DIR
from zelda_i.ram import ADDR_WHISTLE, read_snapshot, read_u8

def dump(env):
    s = read_snapshot(env.get_ram())
    return {
        "sc": f"0x{s.screen:02x}", "mode": s.mode, "xy": [s.link_x, s.link_y],
        "doors": int(s.cur_opened_doors), "mask": int(s.open_doorway_mask),
        "keys": int(s.keys), "whistle": int(read_u8(env.get_ram(), ADDR_WHISTLE)),
    }

def main():
    configure_headless()
    env = make_env(GAME, "Level5Whistle", GAME_DIR, render_mode="rgb_array")
    assist = UnlimitedHealthAssist(enabled=True)
    n = [1]
    log = []
    try:
        reset_obs(env); env.step(nes_idle_action()); assist.apply_env(env, frame=0)
        idle(env, assist, n, 8)
        exit_whistle_04(env, assist, n)
        walk_east_from_05(env, assist, n)
        take_block_stairs_06(env, assist, n)
        cellar_07_to_64(env, assist, n)
        walk_axis(env, assist, n, "y", 189, max_f=400)
        walk_axis(env, assist, n, "x", 208, max_f=400)
        walk_axis(env, assist, n, "y", 141, max_f=300)
        walk_axis(env, assist, n, "x", 224, max_f=200)
        push_dir(env, assist, n, "RIGHT", frames=240)
        idle(env, assist, n, 12)
        bomb_east_from_65(env, assist, n)
        log.append({"tag": "at66_west", **dump(env)})
        print("AT66", log[-1], flush=True)
        # south to 76
        walk_axis(env, assist, n, "x", 120, max_f=300)
        walk_axis(env, assist, n, "y", 205, max_f=300)
        push_dir(env, assist, n, "DOWN", frames=240)
        idle(env, assist, n, 16)
        log.append({"tag": "at76", **dump(env)})
        print("AT76", log[-1], flush=True)
        save_rgb_png(env.step(nes_idle_action())[0], RECORDINGS_DIR / "l5_76_from66.png")
        # north back into 66
        walk_axis(env, assist, n, "x", 120, max_f=300)
        walk_axis(env, assist, n, "y", 93, max_f=300)
        push_dir(env, assist, n, "UP", frames=240)
        idle(env, assist, n, 16)
        log.append({"tag": "at66_south", **dump(env)})
        print("AT66S", log[-1], flush=True)
        # try north
        walk_axis(env, assist, n, "x", 120, max_f=300)
        walk_axis(env, assist, n, "y", 93, max_f=300)
        idle(env, assist, n, 40)
        push_dir(env, assist, n, "UP", frames=260)
        idle(env, assist, n, 16)
        log.append({"tag": "after_north", **dump(env)})
        print("NORTH", log[-1], flush=True)
        save_rgb_png(env.step(nes_idle_action())[0], RECORDINGS_DIR / "l5_66_shutter.png")
        write_json_report(RECORDINGS_DIR / "l5_66_shutter.json", {"log": log, "pokes": False})
    finally:
        env.close()

if __name__ == "__main__":
    main()
