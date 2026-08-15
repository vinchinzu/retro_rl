"""At y=141, test UP/DOWN at several x to find 0x04 ladders."""
from __future__ import annotations

from retro_harness.env import make_env, reset_obs
from retro_harness.nes import nes_action, nes_idle_action
from retro_harness.segment_runner import configure_headless, save_rgb_png, write_json_report
from zelda_i.assist import UnlimitedHealthAssist
from zelda_i.dungeon_ops import idle
from zelda_i.paths import GAME, GAME_DIR, RECORDINGS_DIR
from zelda_i.ram import ADDR_WHISTLE, PLAY_MODE, read_snapshot, read_u8

STATE = "Level5Whistle"


def step(env, assist, total, a):
    env.step(a)
    total[0] += 1
    assist.apply_env(env, frame=total[0])


def go_x(env, assist, total, tx, n=200):
    for _ in range(n):
        s = read_snapshot(env.get_ram())
        if abs(s.link_x - tx) <= 1:
            return True
        step(env, assist, total, nes_action("RIGHT" if s.link_x < tx else "LEFT"))
    return False


def main():
    configure_headless()
    env = None
    total = [1]
    results = []
    try:
        env = make_env(GAME, STATE, GAME_DIR, render_mode="rgb_array")
        assist = UnlimitedHealthAssist(enabled=True)
        reset_obs(env)
        env.step(nes_idle_action())
        assist.apply_env(env, frame=0)
        idle(env, assist, total, 80)
        for _ in range(16):
            step(env, assist, total, nes_action("LEFT"))
        idle(env, assist, total, 8)

        for tx in (112, 120, 128, 144, 160, 176, 184, 192, 200, 208):
            go_x(env, assist, total, tx)
            s0 = read_snapshot(env.get_ram())
            # DOWN
            for _ in range(90):
                step(env, assist, total, nes_action("DOWN"))
            sd = read_snapshot(env.get_ram())
            # back up if we moved
            if sd.link_y != s0.link_y:
                for _ in range(90):
                    step(env, assist, total, nes_action("UP"))
            go_x(env, assist, total, tx)
            # UP
            s1 = read_snapshot(env.get_ram())
            for _ in range(90):
                step(env, assist, total, nes_action("UP"))
            su = read_snapshot(env.get_ram())
            rec = {
                "tx": tx,
                "at": [s0.link_x, s0.link_y],
                "down": [sd.link_x, sd.link_y, sd.mode, f"0x{sd.screen:02x}"],
                "up": [su.link_x, su.link_y, su.mode, f"0x{su.screen:02x}"],
                "whistle": int(read_u8(env.get_ram(), ADDR_WHISTLE)),
            }
            results.append(rec)
            print(rec, flush=True)
            # recover to y=141 corridor
            if su.link_y < 130:
                for _ in range(80):
                    step(env, assist, total, nes_action("DOWN"))
            if su.screen != 0x04 or su.mode == PLAY_MODE:
                break

        write_json_report(RECORDINGS_DIR / "l5_04_ladders.json", {"results": results, "pokes": False})
        save_rgb_png(env.step(nes_idle_action())[0], RECORDINGS_DIR / "l5_04_ladders.png")
        print("DONE", flush=True)
    finally:
        if env is not None:
            env.close()


if __name__ == "__main__":
    main()
