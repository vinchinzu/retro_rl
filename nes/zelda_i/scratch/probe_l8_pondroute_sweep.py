"""rr-6o7.4 scratch: sweep one screen of the reverse L7-pond -> L8 route.

Usage: --screen 0x55 [--right | --down | --up | --left]
Drives Level7PondToLevel8BushController from OW_L7Pond until settled on the
target screen, then for a set of staged poses pushes the sweep direction and
reports the terminal cell / any screen transition. Re-stages via the
controller whenever Link leaves the target screen.
"""

from __future__ import annotations

import argparse

from retro_harness.env import make_env, reset_obs
from retro_harness.nes import nes_action, nes_idle_action
from retro_harness.segment_runner import configure_headless, save_rgb_png
from zelda_i.level8.overworld import Level7PondToLevel8BushController
from zelda_i.paths import GAME, GAME_DIR, RECORDINGS_DIR
from zelda_i.ram import read_snapshot


def snap(env):
    return read_snapshot(env.get_ram())


def step(env, btn):
    obs, *_ = env.step(nes_action(btn) if btn else nes_idle_action())
    return obs


def stage(env, scr):
    reset_obs(env)
    ctl = Level7PondToLevel8BushController()
    for _ in range(40000):
        s = snap(env)
        if s.screen == scr and s.mode == 5 and not s.transitioning:
            return True
        if s.screen not in (0x42, 0x52, 0x53, 0x54, 0x64, 0x65, scr) and s.mode == 5:
            return False
        env.step(ctl.step(s).action)
    return False


def goto(env, scr, tx, ty, budget=400):
    last, stuck = None, 0
    for _ in range(budget):
        s = snap(env)
        if s.screen != scr or s.mode != 5:
            return s
        x, y = s.link_x, s.link_y
        if abs(x - tx) <= 3 and abs(y - ty) <= 3:
            return s
        if (x, y) == last:
            stuck += 1
        else:
            stuck = 0
        last = (x, y)
        if stuck > 20:
            return s
        if abs(x - tx) >= abs(y - ty):
            step(env, "LEFT" if x > tx else "RIGHT")
        else:
            step(env, "UP" if y > ty else "DOWN")
    return snap(env)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--screen", type=lambda v: int(v, 0), required=True)
    ap.add_argument("--dir", default="RIGHT")
    args = ap.parse_args()
    scr = args.screen
    configure_headless()
    env = make_env(GAME, "OW_L7Pond", GAME_DIR, render_mode="rgb_array")
    if not stage(env, scr):
        print(f"could not stage on 0x{scr:02X}")
        return
    s = snap(env)
    print(f"staged 0x{scr:02X} at ({s.link_x},{s.link_y})")
    save_rgb_png(step(env, None), RECORDINGS_DIR / f"l8_sweep_{scr:02x}.png")

    d = args.dir
    if d in ("RIGHT", "LEFT"):
        axis = [(x, "y") for x in range(72, 210, 12)]
        poses = [(120, y) for y in range(72, 200, 12)]
    else:
        poses = [(x, 120) for x in range(16, 232, 12)]
    for px, py in poses:
        if snap(env).screen != scr:
            if not stage(env, scr):
                print("restage failed")
                break
        s0 = goto(env, scr, px, py)
        if s0.screen != scr:
            print(f"stage ({px},{py}): drifted to 0x{s0.screen:02X}")
            continue
        last, stuck = None, 0
        path_min_x = path_max_x = s0.link_x
        path_min_y = path_max_y = s0.link_y
        for _ in range(300):
            s = snap(env)
            if s.screen != scr:
                break
            path_min_x = min(path_min_x, s.link_x)
            path_max_x = max(path_max_x, s.link_x)
            path_min_y = min(path_min_y, s.link_y)
            path_max_y = max(path_max_y, s.link_y)
            if (s.link_x, s.link_y) == last:
                stuck += 1
            else:
                stuck = 0
            last = (s.link_x, s.link_y)
            if stuck > 16:
                break
            step(env, d)
        s = snap(env)
        tag = f"-> 0x{s.screen:02X}" if s.screen != scr else ""
        print(
            f"from ({s0.link_x},{s0.link_y}) push {d}: "
            f"end ({s.link_x},{s.link_y}) x[{path_min_x}-{path_max_x}] "
            f"y[{path_min_y}-{path_max_y}] {tag}"
        )
    env.close()


if __name__ == "__main__":
    main()
