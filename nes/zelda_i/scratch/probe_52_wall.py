"""Find the 0x52 boulder-wall gap: per-x, push UP from the mid-band (y~110)
and record where Link stalls / which column funnels through to pond 0x42.

    uv run python nes/zelda_i/scratch/probe_52_wall.py --tag wall
"""

from __future__ import annotations

import argparse

from retro_harness.env import make_env, reset_obs
from retro_harness.nes import nes_action, nes_idle_action
from retro_harness.segment_runner import configure_headless, save_rgb_png
from zelda_i.level7.overworld import OverworldToLevel7PondController
from zelda_i.paths import GAME, GAME_DIR, RECORDINGS_DIR
from zelda_i.ram import read_snapshot
from zelda_i.runner import add_common_args, make_assist, write_report

ON = 0x52


def _to(env, assist, tx, ty, *, tol=4, max_f=500):
    for _ in range(max_f):
        s = read_snapshot(env.get_ram())
        if s.screen != ON:
            return s
        dx, dy = tx - s.link_x, ty - s.link_y
        if abs(dx) <= tol and abs(dy) <= tol:
            return s
        act = nes_action("RIGHT" if dx > 0 else "LEFT") if abs(dx) > tol else nes_action("DOWN" if dy > 0 else "UP")
        env.step(act)
        if assist:
            assist.apply_env(env, frame=0)
    return read_snapshot(env.get_ram())


def main() -> None:
    p = argparse.ArgumentParser()
    add_common_args(p, default_state="PostSwordStart", default_tag="sky")
    args = p.parse_args()
    configure_headless()
    assist = make_assist(args.infinite_life)
    env = make_env(GAME, args.from_state, GAME_DIR, render_mode="rgb_array")
    RECORDINGS_DIR.mkdir(parents=True, exist_ok=True)
    sky = {}
    try:
        obs, _ = reset_obs(env)
        env.step(nes_idle_action())
        ctl = OverworldToLevel7PondController()
        for _ in range(26000):
            s = read_snapshot(env.get_ram())
            if s.screen == ON and s.mode == 5 and not s.transitioning:
                break
            obs, *_ = env.step(ctl.step(s).action)
            if assist:
                assist.apply_env(env, frame=0)
            if ctl.failed and s.screen != ON:
                break
        a = read_snapshot(env.get_ram())
        print(f"arrived 0x{a.screen:02x} ({a.link_x},{a.link_y})")
        if a.screen != ON:
            return
        for x in list(range(40,180,4)):
            _to(env, assist, 40, 120)
            s = _to(env, assist, x, 110)
            if s.screen != ON:
                sky[x] = f"restage->0x{s.screen:02x}"
                continue
            miny = s.link_y
            for _ in range(220):
                env.step(nes_action("UP"))
                if assist:
                    assist.apply_env(env, frame=0)
                s = read_snapshot(env.get_ram())
                if s.screen != ON:
                    miny = f"EXIT->0x{s.screen:02x}@x{s.link_x}"
                    break
                miny = min(miny, s.link_y)
            sky[x] = miny
            print(f"  x={x}: {miny}")
        save_rgb_png(obs, RECORDINGS_DIR / f"{args.tag}_final.png")
        out = write_report("probe_52_wall", {"arrived": [a.link_x, a.link_y], "skyline": sky}, tag=args.tag)
        print(out)
        print(sky)
    finally:
        env.close()


if __name__ == "__main__":
    main()
