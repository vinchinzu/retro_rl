"""H1 round 3: confirmed live! Blowing the Recorder on 0x24 (deep in the
already-greened post-L6 prefix, NOT a dungeon-entrance screen, NOT freshly
loaded) triggers a real whirlwind-carry cutscene: Link auto-walks east,
scrolls off 0x24 into 0x3b -> settles 0x3c (``SCREEN_LEVEL2_ENTRANCE``),
then keeps auto-walking east on the new screen (round 2 diagnostic,
``rw_diag_up2``, mode 5->6->7->4->5, screen 0x24->0x3b->0x3c).

This probe blows the recorder repeatedly (facing UP each time, i.e. "next
completed dungeon" per source), waiting for EACH cutscene to fully settle
(mode==5, not transitioning, x/y stable for SETTLE_STABLE_FRAMES) before the
next blow, and records the full destination CYCLE: does it visit 0x3c (L2),
0x74 (L3), 0x45/0x55 (L4 -- on the already-live LEVEL7_POND_APPROACH_HOPS
chain!), 0x0b (L5), 0x22 (L6, back to start)?

No RAM pokes (writes=0).

    QT_QPA_PLATFORM=offscreen uv run python \
        nes/zelda_i/scratch/pond/probe_recorder_warp_cycle.py \
        --no-video --n-blows 6 --tag rw_cycle
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
    ADDR_WHISTLE,
    PLAY_MODE,
    read_snapshot,
    read_u8,
)
from zelda_i.runner import add_common_args, make_assist, write_report

WALK_MAX = 20_000
FACE_SETTLE = 20
SETTLE_STABLE_FRAMES = 90
MAX_WAIT_PER_BLOW = 3000
N_BLOWS_DEFAULT = 6


def _shot(obs, tag: str, frame: int, snap, suffix: str = "") -> str:
    RECORDINGS_DIR.mkdir(parents=True, exist_ok=True)
    png = RECORDINGS_DIR / (
        f"{tag}_f{frame}_L{snap.level}_s{snap.screen:02x}_m{snap.mode}{suffix}.png"
    )
    save_rgb_png(obs, png)
    return str(png)


def main() -> None:
    parser = argparse.ArgumentParser()
    add_common_args(parser, default_state=POST_L6_EXIT_STATE, default_tag="rw_cycle")
    parser.add_argument("--no-video", action="store_true", help="ignored; always rgb_array")
    parser.add_argument("--hops", type=int, default=4, help="POST_L6_TO_POND_HOPS to walk (4=0x24)")
    parser.add_argument("--face", choices=["UP", "DOWN"], default="UP")
    parser.add_argument("--n-blows", type=int, default=N_BLOWS_DEFAULT)
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

    def wait_settled(max_wait: int) -> tuple[Any, bool]:
        """Idle until mode==PLAY_MODE, not transitioning, x/y unchanged for
        SETTLE_STABLE_FRAMES in a row. Returns (snap, settled)."""
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

    try:
        obs, _ = reset_obs(env)
        obs_box[0] = obs
        snap = step(nes_idle_action())

        truncated_hops = POST_L6_TO_POND_HOPS[: args.hops]
        target_screen = truncated_hops[-1].target if truncated_hops else 0x22
        handoff = handoff_from_ram(env.get_ram(), evidence="probe-recorder-warp-cycle", verified=True)
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
                "l7_recorder_warp_cycle",
                {"walked_ok": False, "controller_report": ctl.report(), "writes": 0},
                tag=args.tag,
            )
            print(out)
            return

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
                "l7_recorder_warp_cycle",
                {"walked_ok": True, "select_ok": False, "writes": 0},
                tag=args.tag,
            )
            print(out)
            return

        landings: list[dict[str, Any]] = []
        for blow_i in range(args.n_blows):
            # Face the requested direction before each blow.
            for _ in range(FACE_SETTLE):
                snap = step(nes_action(args.face))
            pre = read_snapshot(env.get_ram())
            for _ in range(12):
                snap = step(nes_action("B"))
            post, settled = wait_settled(MAX_WAIT_PER_BLOW)
            entry = {
                "blow_index": blow_i,
                "pre_screen": hex(int(pre.screen)),
                "pre_xy": [int(pre.link_x), int(pre.link_y)],
                "pre_facing": int(pre.facing),
                "post_screen": hex(int(post.screen)),
                "post_xy": [int(post.link_x), int(post.link_y)],
                "post_mode": int(post.mode),
                "settled": settled,
                "frame": total[0],
                "whistle": int(read_u8(env.get_ram(), ADDR_WHISTLE)),
            }
            landings.append(entry)
            print(
                f"blow {blow_i}: 0x{entry['pre_screen'] if False else int(pre.screen):02x} "
                f"-> 0x{int(post.screen):02x} xy={entry['post_xy']} mode={entry['post_mode']} "
                f"settled={settled} f={total[0]}"
            )
            _shot(obs_box[0], args.tag, total[0], post, suffix=f"_blow{blow_i}")
            if not settled:
                print("  did not settle within MAX_WAIT_PER_BLOW; stopping cycle")
                break
            if int(post.level) != 0:
                print("  landed inside a dungeon/cave; stopping cycle (no re-blow there)")
                break

        out = write_report(
            "l7_recorder_warp_cycle",
            {
                "walked_ok": True,
                "select_ok": True,
                "face": args.face,
                "hops_walked": args.hops,
                "landings": landings,
                "writes": 0,
            },
            tag=args.tag,
        )
        print(out)
    finally:
        env.close()


if __name__ == "__main__":
    main()
