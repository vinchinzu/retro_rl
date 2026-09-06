"""rr-6o7.1: one cardinal push from the refilled 0x42 north strip.

Claim: holding --dir from Level7Entrance leftover pose changes RAM x/y.
Grade after 200 frames. Do not occupancy-walk.

    QT_QPA_PLATFORM=offscreen uv run python \
        nes/zelda_i/scratch/pond/probe_42_push.py --dir LEFT --tag l8_42L
"""

from __future__ import annotations

import argparse

from retro_harness.env import make_env, reset_obs
from retro_harness.nes import nes_action, nes_idle_action
from retro_harness.segment_runner import configure_headless, save_rgb_png
from zelda_i.dungeon.tilemap import tile_at_screen
from zelda_i.paths import GAME, GAME_DIR, RECORDINGS_DIR
from zelda_i.ram import PLAY_MODE, read_snapshot
from zelda_i.runner import add_common_args, make_assist

POND_SCREEN = 0x42
EXIT_MAX = 800
PUSH_FRAMES = 200


def main() -> None:
    parser = argparse.ArgumentParser()
    add_common_args(parser, default_state="Level7Entrance", default_tag="l8_42push")
    parser.add_argument("--dir", default="LEFT", choices=["LEFT", "RIGHT", "UP", "DOWN"])
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
        start = (int(snap.link_x), int(snap.link_y), int(snap.screen), int(snap.mode))
        ram = env.get_ram()
        print(
            f"claim: {args.dir} from {start[:2]} tile="
            f"{tile_at_screen(ram, start[0], start[1]):02x}"
        )
        RECORDINGS_DIR.mkdir(parents=True, exist_ok=True)
        save_rgb_png(obs_box[0], RECORDINGS_DIR / f"{args.tag}_start.png")
        for i in range(PUSH_FRAMES):
            snap = step(args.dir)
            if i % 50 == 49 or snap.screen != POND_SCREEN or snap.mode != PLAY_MODE:
                print(
                    f"f={total[0]} ({snap.link_x},{snap.link_y}) "
                    f"s=0x{snap.screen:02x} m={snap.mode}"
                )
            if snap.screen != POND_SCREEN or snap.mode != PLAY_MODE:
                break
        save_rgb_png(obs_box[0], RECORDINGS_DIR / f"{args.tag}_final.png")
        dx = int(snap.link_x) - start[0]
        dy = int(snap.link_y) - start[1]
        moved = dx != 0 or dy != 0 or snap.screen != start[2]
        print(
            f"grade: start={start[:2]} end=({snap.link_x},{snap.link_y}) "
            f"d=({dx},{dy}) screen=0x{snap.screen:02x} moved={moved}"
        )
    finally:
        env.close()


if __name__ == "__main__":
    main()
