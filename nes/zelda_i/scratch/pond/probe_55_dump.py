"""Diagnostic: after the live Recorder-warp + join hop (0x24 -> ... -> 0x45
-> 0x55 via ``ScreenHop(0x55, "DOWN", align_x=128)``, confirmed live 2/2 by
``probe_recorder_warp_join.py``), dump the ``$6530`` tile map for 0x55 to see
why the generic ``ScreenHop(0x65, "DOWN", align_x=112)`` (the next hop in
the already-green ``LEVEL7_POND_HOPS`` tail, designed for an *east* 0x56->
0x55 arrival, not this *north* raft-dock arrival) got stuck around
``(128,103)`` instead of reaching the south exit (``probe_recorder_warp_
full_route.py`` rw_full_route_t1, f=32910, tail_failed=True).

Uses the shared Survival infinite-life assist (default on, matching every
other probe in this set) -- its absence in an earlier throwaway version of
this script let an overworld enemy knock Link off-course, which is why that
run diverged from the otherwise-deterministic join.

No RAM pokes beyond the disclosed Survival health assist (writes=0 on the
route-relevant progression/capacity/position counters; the assist only
touches hearts).

    QT_QPA_PLATFORM=offscreen uv run python \
        nes/zelda_i/scratch/pond/probe_55_dump.py --tag l7_55_dump
"""

from __future__ import annotations

import argparse
from typing import Any

from retro_harness.env import make_env, reset_obs
from retro_harness.nes import nes_action, nes_idle_action
from retro_harness.segment_runner import configure_headless, save_rgb_png
from zelda_i.dungeon.pause_select import B_SLOT_RECORDER, PauseSelectController
from zelda_i.dungeon.tilemap import read_room_tiles
from zelda_i.level7.entry import POST_L6_EXIT_STATE
from zelda_i.level7.pond import POST_L6_TO_POND_HOPS, PostLevel6OverworldController
from zelda_i.overworld.graph import ScreenHop
from zelda_i.overworld.stitch import handoff_from_ram
from zelda_i.paths import GAME, GAME_DIR, RECORDINGS_DIR
from zelda_i.ram import PLAY_MODE, read_snapshot
from zelda_i.runner import add_common_args, make_assist

WALK_MAX = 20_000
FACE_SETTLE = 20
SETTLE_STABLE_FRAMES = 90
MAX_WAIT_PER_BLOW = 3000
MAX_BLOWS = 10
ISLAND_SCREEN = 0x45
JOIN_TARGET = 0x55
JOIN_ALIGN_X = 128
HOP_MAX_FRAMES = 2500


def main() -> None:
    parser = argparse.ArgumentParser()
    add_common_args(parser, default_state=POST_L6_EXIT_STATE, default_tag="l7_55_dump")
    parser.add_argument("--hops", type=int, default=4)
    args = parser.parse_args()
    configure_headless()
    assist = make_assist(args.infinite_life)
    env = make_env(GAME, args.from_state, GAME_DIR, render_mode="rgb_array")
    obs_box: list = [None]
    total = [0]

    def step(action) -> Any:
        obs, *_ = env.step(action)
        obs_box[0] = obs
        total[0] += 1
        if assist is not None:
            assist.apply_env(env, frame=total[0])
        return read_snapshot(env.get_ram())

    def wait_settled(max_wait: int):
        stable = 0
        last_xy = None
        for _ in range(max_wait):
            snap = read_snapshot(env.get_ram())
            xy = (int(snap.link_x), int(snap.link_y))
            if snap.mode == PLAY_MODE and not snap.transitioning and xy == last_xy:
                stable += 1
                if stable >= SETTLE_STABLE_FRAMES:
                    return snap, True
            else:
                stable = 0
            last_xy = xy
            snap = step(nes_idle_action())
        return read_snapshot(env.get_ram()), False

    def dump(screen_hex: str) -> None:
        grid = read_room_tiles(env.get_ram())
        print(f"--- tile map for screen {screen_hex} ({grid.shape[0]}x{grid.shape[1]}) ---")
        for row_i, row in enumerate(grid):
            print(f"row{row_i:02d} " + " ".join(f"{v:02x}" for v in row))

    try:
        obs, _ = reset_obs(env)
        obs_box[0] = obs
        snap = step(nes_idle_action())

        truncated_hops = POST_L6_TO_POND_HOPS[: args.hops]
        prefix_target = truncated_hops[-1].target if truncated_hops else 0x22
        handoff = handoff_from_ram(env.get_ram(), evidence="probe-55-dump", verified=True)
        ctl = PostLevel6OverworldController(handoff=handoff, hops=truncated_hops)
        ctl.bind_env(env)
        for _ in range(WALK_MAX):
            snap = read_snapshot(env.get_ram())
            action = ctl.step(snap)
            snap = step(action.action)
            if int(snap.screen) == prefix_target and snap.mode == PLAY_MODE and not snap.transitioning:
                break
            if ctl.failed:
                break
        print(f"prefix landed=0x{snap.screen:02x} f={total[0]}")

        selector = PauseSelectController(want=B_SLOT_RECORDER, name="recorder")
        selector.bind_env(env)
        for _ in range(selector.max_frames + 20):
            snap = read_snapshot(env.get_ram())
            action = selector.drive(snap)
            if action is None:
                break
            if selector.failed:
                break
            snap = step(action.action)
        print(f"select done f={total[0]}")

        for blow_i in range(MAX_BLOWS):
            for _ in range(FACE_SETTLE):
                snap = step(nes_action("DOWN"))
            for _ in range(12):
                snap = step(nes_action("B"))
            post, settled = wait_settled(MAX_WAIT_PER_BLOW)
            print(f"blow {blow_i} -> 0x{int(post.screen):02x} xy=({post.link_x},{post.link_y}) f={total[0]}")
            if int(post.screen) == ISLAND_SCREEN and settled:
                break

        # Join hop 0x45 DOWN align_x=128 -> 0x55.
        handoff2 = handoff_from_ram(env.get_ram(), evidence="probe-55-dump-join", verified=True)
        hop = ScreenHop(JOIN_TARGET, "DOWN", align_x=JOIN_ALIGN_X)
        hop_ctl = PostLevel6OverworldController(handoff=handoff2, hops=(hop,), max_frames=HOP_MAX_FRAMES)
        hop_ctl.bind_env(env)
        join_ok = False
        for _ in range(HOP_MAX_FRAMES):
            snap = read_snapshot(env.get_ram())
            if int(snap.screen) == JOIN_TARGET and snap.mode == PLAY_MODE and not snap.transitioning:
                join_ok = True
                break
            if int(snap.level) != 0 or snap.mode == 16:
                print("  entered a dungeon/door during join hop; aborting")
                break
            action = hop_ctl.step(snap)
            snap = step(action.action)
            if int(snap.screen) == JOIN_TARGET and snap.mode == PLAY_MODE and not snap.transitioning:
                join_ok = True
                break
            if int(snap.level) != 0 or snap.mode == 16:
                print("  entered a dungeon/door during join hop; aborting")
                break
        print(f"join_ok={join_ok} screen=0x{snap.screen:02x} xy=({snap.link_x},{snap.link_y}) f={total[0]}")
        if not join_ok:
            dump(f"0x{snap.screen:02x} (join FAILED, still here)")
            RECORDINGS_DIR.mkdir(parents=True, exist_ok=True)
            save_rgb_png(obs_box[0], RECORDINGS_DIR / f"{args.tag}_join_fail.png")
            return

        landed, settled = wait_settled(1000)
        print(f"landed on 0x{landed.screen:02x}: xy=({landed.link_x},{landed.link_y}) settled={settled} f={total[0]}")
        dump(f"0x{landed.screen:02x}")
        RECORDINGS_DIR.mkdir(parents=True, exist_ok=True)
        png = RECORDINGS_DIR / f"{args.tag}_s{landed.screen:02x}.png"
        save_rgb_png(obs_box[0], png)
        print(f"screenshot={png}")

        # Raw exploration: push DOWN from the dock, logging every xy change,
        # to trace the actual walkable corridor (no align_x assumption).
        print("--- raw DOWN from dock (no x-align), logging xy changes ---")
        last = None
        for i in range(600):
            snap = step(nes_action("DOWN"))
            xy = (int(snap.link_x), int(snap.link_y))
            if xy != last:
                print(f"  down f={i} xy={xy} screen=0x{snap.screen:02x} tile={snap.colliding_tile}")
                last = xy
            if int(snap.screen) != landed.screen:
                print(f"  scrolled to 0x{snap.screen:02x} at f={i}")
                break
    finally:
        env.close()


if __name__ == "__main__":
    main()
