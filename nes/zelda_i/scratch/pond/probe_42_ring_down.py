"""rr-6o7.1: west or east column around the refilled pond, then south gap.

LEFT to x=24 then DOWN, or RIGHT to x=216 then DOWN. Grade whether y
crosses the pond (y>175) and whether 0x52 opens.

    QT_QPA_PLATFORM=offscreen uv run python \
        nes/zelda_i/scratch/pond/probe_42_ring_down.py --side west --tag l8_42W
"""

from __future__ import annotations

import argparse

from retro_harness.env import make_env, reset_obs
from retro_harness.nes import nes_action, nes_idle_action
from retro_harness.segment_runner import configure_headless, save_rgb_png
from zelda_i.paths import GAME, GAME_DIR, RECORDINGS_DIR
from zelda_i.ram import PLAY_MODE, read_snapshot
from zelda_i.runner import add_common_args, make_assist

POND_SCREEN = 0x42
EXIT_MAX = 800
WEST_X = 24
EAST_X = 216
ALIGN_MAX = 200
DOWN_MAX = 400


def main() -> None:
    parser = argparse.ArgumentParser()
    add_common_args(parser, default_state="Level7Entrance", default_tag="l8_42ringd")
    parser.add_argument("--side", choices=["west", "east"], default="west")
    args = parser.parse_args()
    configure_headless()
    assist = make_assist(args.infinite_life)
    env = make_env(GAME, args.from_state, GAME_DIR, render_mode="rgb_array")
    obs_box: list = [None]
    total = [0]

    def step(btn: str | None):
        obs, *_ = env.step(nes_idle_action() if not btn else nes_action(btn))
        obs_box[0] = obs
        total[0] += 1
        if assist is not None:
            assist.apply_env(env, frame=total[0])
        return read_snapshot(env.get_ram())

    try:
        obs, _ = reset_obs(env)
        obs_box[0] = obs
        snap = step(None)
        for _ in range(EXIT_MAX):
            snap = read_snapshot(env.get_ram())
            if (
                snap.level == 0
                and snap.mode == PLAY_MODE
                and snap.screen == POND_SCREEN
                and not snap.transitioning
            ):
                break
            snap = step("DOWN")
        target_x = WEST_X if args.side == "west" else EAST_X
        align = "LEFT" if args.side == "west" else "RIGHT"
        print(f"settled ({snap.link_x},{snap.link_y}) align {align} to x={target_x}")
        RECORDINGS_DIR.mkdir(parents=True, exist_ok=True)
        save_rgb_png(obs_box[0], RECORDINGS_DIR / f"{args.tag}_settled.png")
        for _ in range(ALIGN_MAX):
            snap = step(align)
            if abs(snap.link_x - target_x) <= 2:
                break
        print(f"aligned ({snap.link_x},{snap.link_y}) f={total[0]}")
        save_rgb_png(obs_box[0], RECORDINGS_DIR / f"{args.tag}_aligned.png")
        for i in range(DOWN_MAX):
            snap = step("DOWN")
            if i % 50 == 49 or snap.screen != POND_SCREEN:
                print(
                    f"down f={total[0]} ({snap.link_x},{snap.link_y}) "
                    f"s=0x{snap.screen:02x} m={snap.mode}"
                )
            if snap.screen != POND_SCREEN:
                break
        save_rgb_png(obs_box[0], RECORDINGS_DIR / f"{args.tag}_south.png")
        print(f"south ({snap.link_x},{snap.link_y}) f={total[0]}")
        gap_dir = "RIGHT" if args.side == "west" else "LEFT"
        gap_x = 112
        for _ in range(ALIGN_MAX):
            snap = step(gap_dir)
            if abs(snap.link_x - gap_x) <= 3:
                break
        print(f"gap ({snap.link_x},{snap.link_y}) f={total[0]}")
        save_rgb_png(obs_box[0], RECORDINGS_DIR / f"{args.tag}_gap.png")
        for i in range(DOWN_MAX):
            snap = step("DOWN")
            if i % 20 == 19:
                print(
                    f"exit f={total[0]} ({snap.link_x},{snap.link_y}) "
                    f"s=0x{snap.screen:02x} m={snap.mode}"
                )
            if (
                snap.screen != POND_SCREEN
                and snap.mode == PLAY_MODE
                and not snap.transitioning
            ):
                break
        save_rgb_png(obs_box[0], RECORDINGS_DIR / f"{args.tag}_final.png")
        print(
            f"grade: side={args.side} end=({snap.link_x},{snap.link_y}) "
            f"s=0x{snap.screen:02x} m={snap.mode} trans={snap.transitioning}"
        )
    finally:
        env.close()


if __name__ == "__main__":
    main()
