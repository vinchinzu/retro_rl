"""H2 recon: continue past the confirmed-live 0x12 -> 0x02 north gap
(``probe_12_north_gap.py``, 2/2 identical: UP at x=48 (tol 3) -> 0x02
``(48,61)`` mode 7, f=2424 total). This probe repeats that exact crossing,
lets the screen settle into play mode, dumps 0x02's ``$6530`` tile map, then
tries a free push in ``--dir`` (default LEFT, toward 0x01) to see where the
row-0 west column leads -- looking for a way south to the pond band that
does not pass back through the mountain-locked 0x22/0x32 pocket.

No RAM pokes (writes=0).

    QT_QPA_PLATFORM=offscreen uv run python \
        nes/zelda_i/scratch/pond/probe_02_explore.py --no-video --dir LEFT --tag l7_02L
"""

from __future__ import annotations

import argparse

from retro_harness.env import make_env, reset_obs
from retro_harness.nes import nes_action, nes_idle_action
from retro_harness.segment_runner import configure_headless, save_rgb_png
from zelda_i.dungeon.tilemap import read_room_tiles
from zelda_i.level7.entry import POST_L6_EXIT_STATE
from zelda_i.level7.pond import POST_L6_TO_POND_HOPS, PostLevel6OverworldController
from zelda_i.overworld.stitch import handoff_from_ram
from zelda_i.paths import GAME, GAME_DIR, RECORDINGS_DIR
from zelda_i.ram import PLAY_MODE, read_snapshot
from zelda_i.runner import add_common_args, make_assist, write_report

WALK_MAX = 20_000
CROSS_MAX = 4000
SETTLE_MAX = 1000
PUSH_MAX = 4000
GAP_X = 48


def main() -> None:
    parser = argparse.ArgumentParser()
    add_common_args(parser, default_state=POST_L6_EXIT_STATE, default_tag="l7_02explore")
    parser.add_argument("--no-video", action="store_true")
    parser.add_argument("--dir", default="LEFT", choices=["LEFT", "RIGHT", "UP", "DOWN"])
    args = parser.parse_args()
    configure_headless()
    assist = make_assist(args.infinite_life)
    env = make_env(GAME, args.from_state, GAME_DIR, render_mode="rgb_array")
    obs_box: list = [None]
    total = [0]

    def step(action):
        obs, *_ = env.step(action)
        obs_box[0] = obs
        total[0] += 1
        if assist is not None:
            assist.apply_env(env, frame=total[0])
        return read_snapshot(env.get_ram())

    def shot(snap, suffix=""):
        RECORDINGS_DIR.mkdir(parents=True, exist_ok=True)
        png = RECORDINGS_DIR / f"{args.tag}_f{total[0]}_s{snap.screen:02x}{suffix}.png"
        save_rgb_png(obs_box[0], png)
        return str(png)

    def dump(snap):
        grid = read_room_tiles(env.get_ram())
        print(f"--- screen 0x{snap.screen:02x} tile map ---")
        for row_i, row in enumerate(grid):
            print(f"row{row_i:02d} " + " ".join(f"{v:02x}" for v in row))

    try:
        obs, _ = reset_obs(env)
        obs_box[0] = obs
        snap = step(nes_idle_action())

        hops = POST_L6_TO_POND_HOPS  # 7-hop prefix ends on 0x12
        handoff = handoff_from_ram(env.get_ram(), evidence="probe-02-explore", verified=True)
        ctl = PostLevel6OverworldController(handoff=handoff, hops=hops)
        ctl.bind_env(env)
        walked_ok = False
        for _ in range(WALK_MAX):
            snap = read_snapshot(env.get_ram())
            action = ctl.step(snap)
            snap = step(action.action)
            if int(snap.screen) == 0x12 and snap.mode == PLAY_MODE and not snap.transitioning:
                walked_ok = True
                break
            if ctl.failed:
                break
        print(f"walked_ok={walked_ok} landed=0x{snap.screen:02x} f={total[0]}")
        if not walked_ok:
            print("controller_report", ctl.report())
            return

        # Proven-live tolerant crossing: align x~48 (tol 3), push UP.
        last_xy = None
        stuck = 0
        crossed = False
        for _ in range(CROSS_MAX):
            snap = read_snapshot(env.get_ram())
            if snap.screen != 0x12:
                crossed = True
                break
            if abs(snap.link_x - GAP_X) > 3:
                btn = "RIGHT" if snap.link_x < GAP_X else "LEFT"
                snap = step(nes_action(btn))
                continue
            xy = (int(snap.link_x), int(snap.link_y))
            if xy == last_xy:
                stuck += 1
                if stuck > 90:
                    break
            else:
                stuck = 0
            last_xy = xy
            snap = step(nes_action("UP"))
        print(f"crossed={crossed} screen=0x{snap.screen:02x} xy=({snap.link_x},{snap.link_y}) f={total[0]}")
        if not crossed:
            print("did not cross into 0x02; aborting")
            return

        # Settle into stable play mode on 0x02.
        settled = False
        for _ in range(SETTLE_MAX):
            snap = read_snapshot(env.get_ram())
            if snap.mode == PLAY_MODE and not snap.transitioning:
                settled = True
                break
            snap = step(nes_idle_action())
        print(f"settled={settled} screen=0x{snap.screen:02x} xy=({snap.link_x},{snap.link_y}) f={total[0]}")
        shot(snap, "_settled")
        dump(snap)

        # Free push in the requested direction.
        start_screen = snap.screen
        last_xy = None
        stuck = 0
        left_screen = False
        for _ in range(PUSH_MAX):
            snap = read_snapshot(env.get_ram())
            if snap.screen != start_screen:
                left_screen = True
                break
            xy = (int(snap.link_x), int(snap.link_y))
            if xy == last_xy:
                stuck += 1
                if stuck > 90:
                    break
            else:
                stuck = 0
            last_xy = xy
            snap = step(nes_action(args.dir))
        final = read_snapshot(env.get_ram())
        print(
            f"push {args.dir}: left_screen={left_screen} final=0x{final.screen:02x} "
            f"xy=({final.link_x},{final.link_y}) mode={final.mode} "
            f"tile={final.colliding_tile} f={total[0]}"
        )
        shot(final, f"_after_{args.dir}")
        if left_screen:
            for _ in range(SETTLE_MAX):
                snap = read_snapshot(env.get_ram())
                if snap.mode == PLAY_MODE and not snap.transitioning:
                    break
                snap = step(nes_idle_action())
            dump(snap)
            shot(snap, "_new_screen_settled")

        out = write_report(
            "l7_02_explore",
            {
                "direction": args.dir,
                "left_screen": left_screen,
                "final_screen": hex(int(final.screen)),
                "final_xy": [int(final.link_x), int(final.link_y)],
                "writes": 0,
            },
            tag=args.tag,
        )
        print(out)
    finally:
        env.close()


if __name__ == "__main__":
    main()
