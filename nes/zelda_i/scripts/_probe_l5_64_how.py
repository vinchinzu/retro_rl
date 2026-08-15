"""How do center stairs work on already-cleared 0x64?"""
from retro_harness.env import make_env, reset_obs
from retro_harness.nes import nes_idle_action
from retro_harness.segment_runner import configure_headless, save_rgb_png
from zelda_i.assist import UnlimitedHealthAssist
from zelda_i.dungeon_ops import idle
from zelda_i.level5_path import take_center_stairs_64, walk_axis
from zelda_i.paths import GAME, GAME_DIR, RECORDINGS_DIR
from zelda_i.ram import ADDR_WHISTLE, read_snapshot, read_u8


def dump(env):
    s = read_snapshot(env.get_ram())
    blocks = [{"x": o.x, "y": o.y} for o in s.objects if 1 <= o.slot <= 12 and o.type_id == 0x68]
    return {
        "sc": f"0x{s.screen:02x}", "mode": s.mode, "xy": [s.link_x, s.link_y],
        "tile": int(s.colliding_tile), "whistle": int(read_u8(env.get_ram(), ADDR_WHISTLE)),
        "blocks": blocks, "doors": s.cur_opened_doors,
    }


def main():
    configure_headless()
    env = make_env(GAME, "Level5Cleared64", GAME_DIR, render_mode="rgb_array")
    assist = UnlimitedHealthAssist(enabled=True)
    total = [1]
    reset_obs(env)
    env.step(nes_idle_action())
    assist.apply_env(env, frame=0)
    idle(env, assist, total, 16)
    print("START64", dump(env), flush=True)
    save_rgb_png(env.step(nes_idle_action())[0], RECORDINGS_DIR / "l5_64_cleared_look.png")
    rec = take_center_stairs_64(env, assist, total)
    print("STAIRS", {k: rec[k] for k in rec if k != "log"}, flush=True)
    print("END", dump(env), flush=True)
    save_rgb_png(env.step(nes_idle_action())[0], RECORDINGS_DIR / "l5_64_cleared_stairs.png")
    env.close()


if __name__ == "__main__":
    main()
