"""Recon: reach 0x6A via the green prefix, then y-scan for an east path.

For each y band (top wall to bottom), walk Link onto that band and push RIGHT
for a fixed window, recording the max x reached.  Dark room, so this maps the
invisible interior collision.  Do not poke.

    QT_QPA_PLATFORM=offscreen uv run python \
        nes/zelda_i/scratch/sweep_l7_room6a.py
"""

from __future__ import annotations

import argparse
import json

from retro_harness.env import make_env, reset_obs
from retro_harness.nes import nes_action, nes_idle_action
from retro_harness.segment_runner import configure_headless, save_rgb_png
from zelda_i.level7.path import (
    LEVEL7,
    ROOM_6A,
    EntryNorthDoorController,
    Room69EastController,
)
from zelda_i.paths import GAME, GAME_DIR, RECORDINGS_DIR
from zelda_i.ram import PLAY_MODE, read_snapshot
from zelda_i.runner import add_common_args, make_assist


def _drive_to_6a(env, assist) -> tuple[bool, int]:
    north = EntryNorthDoorController()
    east69 = Room69EastController()
    chain = [north, east69]
    idx = 0
    ctl = chain[idx]
    budget = sum(c.max_frames for c in chain)
    for frame in range(budget):
        snap = read_snapshot(env.get_ram())
        act = ctl.step(snap)
        env.step(act.action)
        if assist is not None:
            assist.apply_env(env, frame=frame)
        if ctl.success and idx < len(chain) - 1:
            idx += 1
            ctl = chain[idx]
            continue
        snap2 = read_snapshot(env.get_ram())
        if snap2.screen == ROOM_6A and snap2.mode == PLAY_MODE and not snap2.transitioning:
            return True, frame
        if ctl.failed:
            return False, frame
    return False, budget


def _hold(env, assist, base_frame, button, frames):
    for i in range(frames):
        env.step(nes_action(button) if button else nes_idle_action())
        if assist is not None:
            assist.apply_env(env, frame=base_frame + i)
    return base_frame + frames


def main() -> None:
    parser = argparse.ArgumentParser()
    add_common_args(parser, default_state="Level7Entrance", default_tag="sweep6a")
    args = parser.parse_args()
    configure_headless()
    assist = make_assist(args.infinite_life)
    env = make_env(GAME, args.from_state, GAME_DIR, render_mode="rgb_array")
    RECORDINGS_DIR.mkdir(parents=True, exist_ok=True)
    results = []
    try:
        reset_obs(env)
        ok, f = _drive_to_6a(env, assist)
        snap = read_snapshot(env.get_ram())
        print(f"reached_6a={ok} frame={f} xy=({snap.link_x},{snap.link_y})")
        if not ok:
            return
        base = f
        # sweep target y bands
        for ty in range(72, 190, 8):
            # move to band: press UP or DOWN until close
            for _ in range(120):
                s = read_snapshot(env.get_ram())
                if s.screen != ROOM_6A:
                    break
                dy = int(s.link_y) - ty
                if abs(dy) <= 3:
                    break
                env.step(nes_action("UP" if dy > 0 else "DOWN"))
                if assist is not None:
                    assist.apply_env(env, frame=base)
                base += 1
            s0 = read_snapshot(env.get_ram())
            if s0.screen != ROOM_6A:
                results.append({"ty": ty, "note": f"left_room_to_0x{s0.screen:02x}", "at": [s0.link_x, s0.link_y]})
                break
            x_start = int(s0.link_x)
            max_x = x_start
            for _ in range(90):
                s = read_snapshot(env.get_ram())
                if s.screen != ROOM_6A:
                    break
                env.step(nes_action("RIGHT"))
                if assist is not None:
                    assist.apply_env(env, frame=base)
                base += 1
                s = read_snapshot(env.get_ram())
                max_x = max(max_x, int(s.link_x))
            s1 = read_snapshot(env.get_ram())
            results.append(
                {
                    "ty": ty,
                    "y_actual": int(s0.link_y),
                    "x_start": x_start,
                    "max_x": max_x,
                    "end_xy": [int(s1.link_x), int(s1.link_y)],
                    "end_screen": f"0x{s1.screen:02x}",
                    "tile": int(s1.colliding_tile),
                }
            )
            print(results[-1])
            if s1.screen != ROOM_6A:
                break
            # walk back west to x~48 for next band
            for _ in range(120):
                s = read_snapshot(env.get_ram())
                if int(s.link_x) <= 40 or s.screen != ROOM_6A:
                    break
                env.step(nes_action("LEFT"))
                if assist is not None:
                    assist.apply_env(env, frame=base)
                base += 1
        obs = env.render()
        save_rgb_png(env.render(), RECORDINGS_DIR / f"{args.tag}_final.png")
        out = RECORDINGS_DIR / f"{args.tag}.json"
        out.write_text(json.dumps({"reached": ok, "results": results}, indent=1))
        print("wrote", out)
        for r in results:
            print(r)
    finally:
        env.close()


if __name__ == "__main__":
    main()
