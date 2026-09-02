"""Recon: sweep Link across the 0x25 Armos grid, hunting a hidden staircase.

Starts from Level6ExitOverworld, walks the fixture-live 0x22->0x25 bait prefix
with OverworldToBaitShopController, then hands off to a waypoint sweep over the
Armos positions.  Screenshots on every mode/screen change and at each waypoint.
Bails as soon as mode enters 9/11/16 (passage / cave / cave-enter) or the screen
changes off 0x25.

    uv run python nes/zelda_i/scratch/sweep_25_armos.py --tag l7_25sweep
"""

from __future__ import annotations

import argparse

from retro_harness.env import make_env, reset_obs
from retro_harness.nes import nes_action, nes_idle_action
from retro_harness.segment_runner import configure_headless, save_rgb_png
from zelda_i.level7.entry import POST_L6_EXIT_STATE
from zelda_i.level7.overworld import OverworldToBaitShopController
from zelda_i.paths import GAME, GAME_DIR, RECORDINGS_DIR
from zelda_i.ram import read_snapshot
from zelda_i.runner import add_common_args, make_assist, write_report
from zelda_i.screen_glance import leftover_from_snapshot

# Armos grid on 0x25 (4 cols x 2 rows) plus the two top-wall gaps, swept in a
# boustrophedon so Link brushes every statue tile.  Coords are Link's target
# (x, y); the game's Armos hitbox is generous.
WAYPOINTS: tuple[tuple[int, int], ...] = (
    (48, 150), (48, 95), (48, 60),          # left column bottom->top, into N gap
    (96, 60), (96, 95), (96, 150),          # 2nd column
    (144, 150), (144, 95), (144, 60),       # 3rd column
    (192, 60), (192, 95), (192, 150),       # 4th column
    (176, 40),                              # right-of-centre top gap
    (128, 95),                              # dead centre
    (72, 95), (120, 95), (168, 95),         # inter-statue centres
)
WP_TOL = 6
BAIL_MODES = frozenset({9, 11, 16})

# After the sweep reveals the hidden staircase in the top wall, march into it.
ENTER_WAYPOINTS: tuple[tuple[int, int], ...] = (
    (208, 100), (208, 64), (208, 48), (208, 32), (208, 20),
)


def main() -> None:
    parser = argparse.ArgumentParser()
    add_common_args(parser, default_state=POST_L6_EXIT_STATE, default_tag="l7_25sweep")
    parser.add_argument("--max-frames", type=int, default=30000)
    args = parser.parse_args()
    configure_headless()

    prefix = OverworldToBaitShopController()  # walks 0x22 -> 0x25
    assist = make_assist(args.infinite_life)
    env = make_env(GAME, args.from_state, GAME_DIR, render_mode="rgb_array")

    wp = 0
    stuck = 0
    last_xy = (-1, -1)
    last_shot_key = None
    phase = "prefix"
    events: list[str] = []
    try:
        obs, _ = reset_obs(env)
        obs, *_ = env.step(nes_idle_action())
        for frame in range(args.max_frames):
            snap = read_snapshot(env.get_ram())
            key = (snap.screen, snap.mode)
            if key != last_shot_key:
                png = RECORDINGS_DIR / f"{args.tag}_f{frame}_s{snap.screen:02x}_m{snap.mode}.png"
                save_rgb_png(obs, png)
                last_shot_key = key
                events.append(f"f{frame} s{snap.screen:02x} m{snap.mode} ({snap.link_x},{snap.link_y})")

            if phase == "prefix":
                if prefix.success and snap.screen == 0x25:
                    phase = "sweep"
                    events.append(f"f{frame} sweep_start ({snap.link_x},{snap.link_y})")
                    action = nes_idle_action()
                elif prefix.failed:
                    events.append(f"f{frame} prefix_failed {prefix.report().get('notes', [])[-3:]}")
                    break
                else:
                    action = prefix.step(snap).action
                obs, *_ = env.step(action)
                if assist is not None:
                    assist.apply_env(env, frame=frame)
                continue

            # sweep phase
            if phase == "sweep" and (snap.screen != 0x25 or snap.mode in BAIL_MODES):
                png = RECORDINGS_DIR / f"{args.tag}_f{frame}_HIT_s{snap.screen:02x}_m{snap.mode}.png"
                save_rgb_png(obs, png)
                events.append(f"f{frame} HIT s{snap.screen:02x} m{snap.mode} ({snap.link_x},{snap.link_y}) wp{wp}")
                phase = "entered"
                break

            if phase == "sweep" and wp >= len(WAYPOINTS):
                events.append(f"f{frame} sweep_exhausted -> enter phase")
                phase = "enter"
                wp = 0
                stuck = 0

            if phase == "enter":
                if snap.screen != 0x25 or snap.mode in BAIL_MODES:
                    png = RECORDINGS_DIR / f"{args.tag}_f{frame}_ENTER_s{snap.screen:02x}_m{snap.mode}.png"
                    save_rgb_png(obs, png)
                    events.append(f"f{frame} ENTER s{snap.screen:02x} m{snap.mode} ({snap.link_x},{snap.link_y})")
                    # keep going ~400 frames to settle in the cave, screenshotting
                    for k in range(400):
                        snap = read_snapshot(env.get_ram())
                        if k % 40 == 0:
                            save_rgb_png(obs, RECORDINGS_DIR / f"{args.tag}_cave_k{k}_s{snap.screen:02x}_m{snap.mode}.png")
                        obs, *_ = env.step(nes_idle_action())
                        if assist is not None:
                            assist.apply_env(env, frame=frame + k)
                    snap = read_snapshot(env.get_ram())
                    save_rgb_png(obs, RECORDINGS_DIR / f"{args.tag}_cave_settled_s{snap.screen:02x}_m{snap.mode}.png")
                    events.append(f"settled s{snap.screen:02x} m{snap.mode} ({snap.link_x},{snap.link_y})")
                    break
                waypoints = ENTER_WAYPOINTS
            else:
                waypoints = WAYPOINTS

            if wp >= len(waypoints):
                events.append(f"f{frame} {phase}_exhausted")
                break

            tx, ty = waypoints[wp]
            dx, dy = tx - snap.link_x, ty - snap.link_y
            if abs(dx) <= WP_TOL and abs(dy) <= WP_TOL:
                png = RECORDINGS_DIR / f"{args.tag}_wp{wp}_f{frame}_x{snap.link_x}_y{snap.link_y}.png"
                save_rgb_png(obs, png)
                events.append(f"f{frame} wp{wp} reached ({snap.link_x},{snap.link_y})")
                wp += 1
                stuck = 0
                obs, *_ = env.step(nes_idle_action())
                continue

            cur_xy = (snap.link_x, snap.link_y)
            stuck = stuck + 1 if cur_xy == last_xy else 0
            last_xy = cur_xy
            if stuck > 90:
                events.append(f"f{frame} wp{wp} stuck at ({snap.link_x},{snap.link_y}), skip")
                png = RECORDINGS_DIR / f"{args.tag}_wp{wp}_STUCK_f{frame}.png"
                save_rgb_png(obs, png)
                wp += 1
                stuck = 0
                continue

            if abs(dx) >= abs(dy):
                btn = "RIGHT" if dx > 0 else "LEFT"
            else:
                btn = "DOWN" if dy > 0 else "UP"
            obs, *_ = env.step(nes_action(btn))
            if assist is not None:
                assist.apply_env(env, frame=frame)

        snap = read_snapshot(env.get_ram())
        leftover = leftover_from_snapshot(snap)
        png = RECORDINGS_DIR / f"{args.tag}_final_s{snap.screen:02x}_m{snap.mode}.png"
        save_rgb_png(obs, png)
        payload = {
            "phase": phase,
            "wp": wp,
            "leftover": leftover,
            "events": events,
            "screenshot": str(png),
        }
        out = write_report("l7_25sweep", payload, tag=args.tag)
        print(out)
        for e in events:
            print(e)
        print(f"final: s{snap.screen:02x} m{snap.mode} ({snap.link_x},{snap.link_y}) wp={wp}")
    finally:
        env.close()


if __name__ == "__main__":
    main()
