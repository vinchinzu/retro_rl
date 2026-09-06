"""H1 diagnostic: dense per-frame RAM log around one recorder blow, away from
any dungeon-entrance screen, to see whether ANYTHING happens (submode,
facing, x/y drift, tile) even if the final screen never changes.

round 1/2 (rw1, rw_h1/rw_h2 sweeps) found zero change across 12 blows on
0x22 and 0x32. This probe walks to 0x24 (SCREEN_BRACELET_ARMOS, deep in the
already-greened prefix, nowhere near a dungeon mouth), centers away from any
screen edge, tries facing UP and facing DOWN before blowing (facing is
reported to determine warp-cycle direction), and logs every single frame
for a long window post-blow.

No RAM pokes (writes=0).

    QT_QPA_PLATFORM=offscreen uv run python \
        nes/zelda_i/scratch/pond/probe_recorder_warp_diag.py --no-video --tag rw_diag
"""

from __future__ import annotations

import argparse
from typing import Any

from retro_harness.env import make_env, reset_obs
from retro_harness.nes import nes_action, nes_idle_action
from retro_harness.segment_runner import configure_headless, save_rgb_png
from zelda_i.dungeon.pause_select import B_SLOT_RECORDER, PauseSelectController
from zelda_i.level7.entry import POST_L6_EXIT_STATE
from zelda_i.level7.pond import POST_L6_TO_POND_HOPS, PostLevel6OverworldController
from zelda_i.overworld.stitch import handoff_from_ram
from zelda_i.paths import GAME, GAME_DIR, RECORDINGS_DIR
from zelda_i.ram import (
    ADDR_SELECTED_ITEM,
    ADDR_TRIFORCE,
    ADDR_WHISTLE,
    PLAY_MODE,
    read_snapshot,
    read_u8,
)
from zelda_i.runner import add_common_args, make_assist, write_report

WALK_MAX = 20_000
DENSE_LOG_FRAMES = 1200
FACE_SETTLE = 20


def main() -> None:
    parser = argparse.ArgumentParser()
    add_common_args(parser, default_state=POST_L6_EXIT_STATE, default_tag="rw_diag")
    parser.add_argument("--no-video", action="store_true", help="ignored; always rgb_array")
    parser.add_argument("--hops", type=int, default=4, help="POST_L6_TO_POND_HOPS to walk (4=0x24)")
    parser.add_argument("--face", choices=["UP", "DOWN"], default="UP")
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

    try:
        obs, _ = reset_obs(env)
        obs_box[0] = obs
        snap = step(nes_idle_action())
        ram0 = env.get_ram()
        print(f"triforce=0x{int(read_u8(ram0, ADDR_TRIFORCE)):02x}")

        truncated_hops = POST_L6_TO_POND_HOPS[: args.hops]
        target_screen = truncated_hops[-1].target if truncated_hops else 0x22
        handoff = handoff_from_ram(env.get_ram(), evidence="probe-recorder-warp-diag", verified=True)
        ctl = PostLevel6OverworldController(handoff=handoff, hops=truncated_hops)
        ctl.bind_env(env)
        walked_ok = False
        for _ in range(WALK_MAX):
            snap = read_snapshot(env.get_ram())
            action = ctl.step(snap)
            snap = step(action.action)
            if int(snap.screen) == target_screen and snap.mode == PLAY_MODE and not snap.transitioning:
                walked_ok = True
                break
            if ctl.failed:
                break
        print(
            f"walked_ok={walked_ok} ctl_failed={ctl.failed} target=0x{target_screen:02x} "
            f"landed=0x{snap.screen:02x} xy=({snap.link_x},{snap.link_y}) f={total[0]}"
        )
        if not walked_ok:
            out = write_report(
                "l7_recorder_warp_diag",
                {"walked_ok": False, "controller_report": ctl.report(), "writes": 0},
                tag=args.tag,
            )
            print(out)
            return

        # Face the requested direction and hold it a moment (facing affects
        # warp-cycle direction per source: up/right = next, down/left = prev).
        for _ in range(FACE_SETTLE):
            snap = step(nes_action(args.face))
        print(f"facing_after={snap.facing} xy=({snap.link_x},{snap.link_y})")

        selector = PauseSelectController(want=B_SLOT_RECORDER, name="recorder")
        selector.bind_env(env)
        select_ok = False
        for _ in range(selector.max_frames + 20):
            snap = read_snapshot(env.get_ram())
            action = selector.drive(snap)
            if action is None:
                select_ok = True
                break
            if selector.failed:
                break
            snap = step(action.action)
        print(f"select_ok={select_ok} failed={selector.failed} reason={selector.fail_reason}")
        if not select_ok:
            out = write_report(
                "l7_recorder_warp_diag",
                {"walked_ok": True, "select_ok": False, "reason": selector.fail_reason, "writes": 0},
                tag=args.tag,
            )
            print(out)
            return

        pre = read_snapshot(env.get_ram())
        print(
            f"pre-blow screen=0x{pre.screen:02x} xy=({pre.link_x},{pre.link_y}) "
            f"facing={pre.facing} selected={int(read_u8(env.get_ram(), ADDR_SELECTED_ITEM))}"
        )

        # Hold B for 12 frames (the proven pond recipe), then dense-log.
        for _ in range(12):
            snap = step(nes_action("B"))

        log: list[dict[str, Any]] = []
        last = None
        for f in range(DENSE_LOG_FRAMES):
            snap = read_snapshot(env.get_ram())
            row = (
                int(snap.level),
                int(snap.screen),
                int(snap.mode),
                int(snap.submode),
                int(snap.link_x),
                int(snap.link_y),
                int(snap.facing),
                int(snap.colliding_tile),
            )
            if row != last:
                entry = {
                    "f": f,
                    "level": row[0],
                    "screen": hex(row[1]),
                    "mode": row[2],
                    "submode": row[3],
                    "x": row[4],
                    "y": row[5],
                    "facing": row[6],
                    "tile": row[7],
                }
                log.append(entry)
                print(f"  f={f} {entry}")
                last = row
            snap = step(nes_idle_action())
        post = read_snapshot(env.get_ram())
        print(
            f"post-blow(+{DENSE_LOG_FRAMES}f) screen=0x{post.screen:02x} "
            f"xy=({post.link_x},{post.link_y}) mode={post.mode} whistle="
            f"{int(read_u8(env.get_ram(), ADDR_WHISTLE))}"
        )
        RECORDINGS_DIR.mkdir(parents=True, exist_ok=True)
        png = RECORDINGS_DIR / f"{args.tag}_final_s{post.screen:02x}.png"
        save_rgb_png(obs_box[0], png)
        out = write_report(
            "l7_recorder_warp_diag",
            {
                "walked_ok": True,
                "select_ok": True,
                "face": args.face,
                "pre": {
                    "screen": hex(pre.screen),
                    "x": pre.link_x,
                    "y": pre.link_y,
                    "facing": pre.facing,
                },
                "post": {
                    "screen": hex(post.screen),
                    "x": post.link_x,
                    "y": post.link_y,
                    "mode": post.mode,
                },
                "change_log": log,
                "screenshot": str(png),
                "writes": 0,
            },
            tag=args.tag,
        )
        print(out)
    finally:
        env.close()


if __name__ == "__main__":
    main()
