"""Recon: from 0x24 (bracelet Armos, grid E3) find the way to bait shop 0x34
(grid E4, one screen SOUTH).  External map (nesmaps Q1): L6=C3=0x22,
bracelet=E3=0x24, bait shop=E4=0x34.  The fixture prefix over-shot to 0x25
(east of 0x24); 0x25 south is solid mountain, so resume from 0x24.

Walks the fixture prefix up to 0x24, screenshots it, sweeps the Armos grid to
trip any hidden stair, then probes DOWN at a spread of x columns for a
0x24 -> 0x34 transition.

    uv run python nes/zelda_i/scratch/probe_24_to_shop.py --tag l7_24probe
"""

from __future__ import annotations

import argparse

from retro_harness.env import make_env, reset_obs
from retro_harness.nes import nes_action, nes_idle_action
from retro_harness.segment_runner import configure_headless, save_rgb_png
from zelda_i.level7.entry import POST_L6_EXIT_STATE
from zelda_i.level7.overworld import POST_L6_TO_BAIT_HOPS, OverworldToBaitShopController
from zelda_i.paths import GAME, GAME_DIR, RECORDINGS_DIR
from zelda_i.ram import read_snapshot
from zelda_i.runner import add_common_args, make_assist, write_report
from zelda_i.screen_glance import leftover_from_snapshot

# prefix ends at 0x24 (drop the trailing 0x25 RIGHT hop)
PREFIX_HOPS = POST_L6_TO_BAIT_HOPS[:-1]

# Armos grid sweep on 0x24 (reveal hidden stair), then south-edge probe.
ARMOS_WP: tuple[tuple[int, int], ...] = (
    (48, 150), (48, 90), (96, 90), (96, 150), (144, 150), (144, 90),
    (192, 90), (192, 150), (128, 90), (208, 90), (208, 150),
)
# DOWN probe columns along the south edge of 0x24
SOUTH_COLS = (40, 72, 104, 128, 152, 184, 216)
WP_TOL = 7
BAIL_MODES = frozenset({9, 11, 16})


def main() -> None:
    parser = argparse.ArgumentParser()
    add_common_args(parser, default_state=POST_L6_EXIT_STATE, default_tag="l7_24probe")
    parser.add_argument("--max-frames", type=int, default=40000)
    args = parser.parse_args()
    configure_headless()

    prefix = OverworldToBaitShopController(hops=PREFIX_HOPS)
    assist = make_assist(args.infinite_life)
    env = make_env(GAME, args.from_state, GAME_DIR, render_mode="rgb_array")

    phase = "prefix"
    wp = 0
    col = 0
    stuck = 0
    last_xy = (-1, -1)
    last_key = None
    events: list[str] = []

    def shot(obs, tag: str) -> None:
        s = read_snapshot(env.get_ram())
        save_rgb_png(obs, RECORDINGS_DIR / f"{args.tag}_{tag}_s{s.screen:02x}_m{s.mode}.png")

    try:
        obs, _ = reset_obs(env)
        obs, *_ = env.step(nes_idle_action())
        for frame in range(args.max_frames):
            snap = read_snapshot(env.get_ram())
            key = (snap.screen, snap.mode)
            if key != last_key:
                shot(obs, f"f{frame}")
                events.append(f"f{frame} s{snap.screen:02x} m{snap.mode} ({snap.link_x},{snap.link_y}) [{phase}]")
                last_key = key

            if phase == "prefix":
                if prefix.success and snap.screen == 0x24:
                    phase = "armos"
                    shot(obs, f"arrive24_f{frame}")
                    events.append(f"f{frame} arrive 0x24 ({snap.link_x},{snap.link_y})")
                    obs, *_ = env.step(nes_idle_action())
                    continue
                if prefix.failed:
                    events.append(f"f{frame} prefix_failed {prefix.report().get('notes', [])[-3:]}")
                    break
                obs, *_ = env.step(prefix.step(snap).action)
                if assist is not None:
                    assist.apply_env(env, frame=frame)
                continue

            # 0x24 exploration
            if snap.screen != 0x24 or snap.mode in BAIL_MODES:
                shot(obs, f"HIT_f{frame}")
                events.append(f"f{frame} HIT s{snap.screen:02x} m{snap.mode} ({snap.link_x},{snap.link_y}) phase={phase} wp={wp} col={col}")
                # settle & capture
                for k in range(360):
                    if k % 40 == 0:
                        shot(obs, f"after_k{k}")
                    obs, *_ = env.step(nes_idle_action())
                    if assist is not None:
                        assist.apply_env(env, frame=frame + k)
                shot(obs, "settled")
                s = read_snapshot(env.get_ram())
                events.append(f"settled s{s.screen:02x} m{s.mode} ({s.link_x},{s.link_y})")
                break

            targets = ARMOS_WP if phase == "armos" else None
            if phase == "armos":
                if wp >= len(ARMOS_WP):
                    phase = "south"
                    wp = 0
                    events.append(f"f{frame} armos sweep done -> south probe")
                    continue
                tx, ty = ARMOS_WP[wp]
            else:  # south probe: go to (col_x, 150) then push DOWN
                if col >= len(SOUTH_COLS):
                    events.append(f"f{frame} south probe exhausted, no 0x34")
                    break
                cx = SOUTH_COLS[col]
                if abs(snap.link_x - cx) > WP_TOL:
                    tx, ty = cx, 150
                else:
                    # aligned: push DOWN hard for a while
                    obs, *_ = env.step(nes_action("DOWN"))
                    if assist is not None:
                        assist.apply_env(env, frame=frame)
                    cur = (snap.link_x, snap.link_y)
                    stuck = stuck + 1 if cur == last_xy else 0
                    last_xy = cur
                    if stuck > 60:
                        shot(obs, f"southcol{col}_x{cx}_stuck_f{frame}")
                        events.append(f"f{frame} south col {col} x{cx} stuck at ({snap.link_x},{snap.link_y})")
                        col += 1
                        stuck = 0
                    continue

            dx, dy = tx - snap.link_x, ty - snap.link_y
            if abs(dx) <= WP_TOL and abs(dy) <= WP_TOL:
                if phase == "armos":
                    shot(obs, f"armos_wp{wp}_f{frame}")
                    events.append(f"f{frame} armos wp{wp} ({snap.link_x},{snap.link_y})")
                    wp += 1
                stuck = 0
                obs, *_ = env.step(nes_idle_action())
                continue

            cur = (snap.link_x, snap.link_y)
            stuck = stuck + 1 if cur == last_xy else 0
            last_xy = cur
            if stuck > 80:
                events.append(f"f{frame} {phase} wp{wp} stuck ({snap.link_x},{snap.link_y}) skip")
                if phase == "armos":
                    wp += 1
                stuck = 0
                continue
            btn = ("RIGHT" if dx > 0 else "LEFT") if abs(dx) >= abs(dy) else ("DOWN" if dy > 0 else "UP")
            obs, *_ = env.step(nes_action(btn))
            if assist is not None:
                assist.apply_env(env, frame=frame)

        s = read_snapshot(env.get_ram())
        leftover = leftover_from_snapshot(s)
        shot(obs, "final")
        payload = {"phase": phase, "wp": wp, "col": col, "leftover": leftover, "events": events}
        out = write_report("l7_24probe", payload, tag=args.tag)
        print(out)
        for e in events:
            print(e)
        print(f"final s{s.screen:02x} m{s.mode} ({s.link_x},{s.link_y})")
    finally:
        env.close()


if __name__ == "__main__":
    main()
