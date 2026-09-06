"""H1 continuous confirmation: post-L6 -> Recorder warp (8 blows, facing
DOWN) -> island door 0x45 -> live join hop ``ScreenHop(0x55, "DOWN",
align_x=128)`` (confirmed by ``probe_recorder_warp_join.py``, candidate 128,
1/1) -> reuse the already-green ``LEVEL7_POND_HOPS`` tail from 0x55 onward
(``LEVEL7_POND_HOPS[6:]``: 0x65 DOWN align_x=112, 0x64 LEFT align_y=141,
0x54 UP, 0x53 LEFT align_y=141, 0x52 LEFT align_y=189, pond 0x42 UP
align_x=112) -> pond ``0x42`` mode 5.

No RAM pokes (writes=0).

    QT_QPA_PLATFORM=offscreen uv run python \
        nes/zelda_i/scratch/pond/probe_recorder_warp_full_route.py \
        --no-video --tag rw_full_route
"""

from __future__ import annotations

import argparse
from typing import Any

from retro_harness.env import make_env, reset_obs
from retro_harness.nes import nes_action, nes_idle_action
from retro_harness.segment_runner import configure_headless, save_rgb_png
from zelda_i.dungeon.pause_select import B_SLOT_RECORDER, PauseSelectController
from zelda_i.level7.entry import POST_L6_EXIT_STATE
from zelda_i.level7.overworld import LEVEL7_POND_HOPS, on_level7_pond_hyp
from zelda_i.level7.pond import POST_L6_TO_POND_HOPS, PostLevel6OverworldController
from zelda_i.overworld.graph import ScreenHop
from zelda_i.overworld.stitch import handoff_from_ram
from zelda_i.paths import GAME, GAME_DIR, RECORDINGS_DIR
from zelda_i.ram import PLAY_MODE, read_snapshot
from zelda_i.runner import add_common_args, make_assist, write_report

WALK_MAX = 20_000
FACE_SETTLE = 20
SETTLE_STABLE_FRAMES = 90
MAX_WAIT_PER_BLOW = 3000
MAX_BLOWS = 10
TAIL_MAX_FRAMES = 30_000
ISLAND_SCREEN = 0x45
JOIN_TARGET = 0x55
JOIN_ALIGN_X = 128
# probe_55_dump.py: the raft-dock column x=128 is a clear open corridor the
# full height of 0x55 (74f raw DOWN, no realign, straight to 0x65) -- the
# stock LEVEL7_POND_HOPS 0x65 hop (align_x=112) assumes the *east* 0x56->
# 0x55 arrival band and instead drags Link LEFT into the mid-screen house/
# tree obstacle (cols 14-17, rows 8-11) when starting from the dock at
# x=128, which is why probe_recorder_warp_full_route.py rw_full_route_t1
# stuck at (128,103) for the full 30000f budget. Keep x=128 through the
# 0x55->0x65 hop instead of re-aligning to 112.
HOP_65_ALIGN_X = 128


def _shot(obs, tag: str, frame: int, snap, suffix: str = "") -> str:
    RECORDINGS_DIR.mkdir(parents=True, exist_ok=True)
    png = RECORDINGS_DIR / (
        f"{tag}_f{frame}_L{snap.level}_s{snap.screen:02x}_m{snap.mode}{suffix}.png"
    )
    save_rgb_png(obs, png)
    return str(png)


def main() -> None:
    parser = argparse.ArgumentParser()
    add_common_args(parser, default_state=POST_L6_EXIT_STATE, default_tag="rw_full_route")
    parser.add_argument("--no-video", action="store_true", help="ignored; always rgb_array")
    parser.add_argument("--hops", type=int, default=4, help="POST_L6_TO_POND_HOPS to walk (4=0x24)")
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

        # --- Phase 1: walk the greened non-entrance prefix to 0x24 ---
        truncated_hops = POST_L6_TO_POND_HOPS[: args.hops]
        prefix_target = truncated_hops[-1].target if truncated_hops else 0x22
        handoff = handoff_from_ram(env.get_ram(), evidence="probe-recorder-warp-full-route", verified=True)
        ctl = PostLevel6OverworldController(handoff=handoff, hops=truncated_hops)
        ctl.bind_env(env)
        walked_ok = False
        for _ in range(WALK_MAX):
            snap = read_snapshot(env.get_ram())
            action = ctl.step(snap)
            snap = step(action.action)
            if int(snap.screen) == prefix_target and snap.mode == PLAY_MODE and not snap.transitioning:
                walked_ok = True
                break
            if ctl.failed:
                break
        print(f"prefix walked_ok={walked_ok} landed=0x{snap.screen:02x} f={total[0]}")
        if not walked_ok:
            out = write_report(
                "l7_recorder_warp_full_route",
                {"walked_ok": False, "writes": 0},
                tag=args.tag,
            )
            print(out)
            return

        # --- Phase 2: select the Recorder ---
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
        print(f"select_ok={select_ok} f={total[0]}")
        if not select_ok:
            out = write_report(
                "l7_recorder_warp_full_route",
                {"walked_ok": True, "select_ok": False, "writes": 0},
                tag=args.tag,
            )
            print(out)
            return

        # --- Phase 3: blow facing DOWN until landing on island 0x45 ---
        reached_island = False
        blow_landings: list[dict[str, Any]] = []
        for blow_i in range(MAX_BLOWS):
            for _ in range(FACE_SETTLE):
                snap = step(nes_action("DOWN"))
            for _ in range(12):
                snap = step(nes_action("B"))
            post, settled = wait_settled(MAX_WAIT_PER_BLOW)
            blow_landings.append(
                {
                    "blow_index": blow_i,
                    "post_screen": hex(int(post.screen)),
                    "post_xy": [int(post.link_x), int(post.link_y)],
                    "settled": settled,
                    "frame": total[0],
                }
            )
            print(
                f"blow {blow_i}: -> 0x{int(post.screen):02x} xy=({int(post.link_x)},{int(post.link_y)}) "
                f"settled={settled} f={total[0]}"
            )
            if not settled or int(post.level) != 0:
                break
            if int(post.screen) == ISLAND_SCREEN and post.mode == PLAY_MODE and not post.transitioning:
                reached_island = True
                break
        print(f"reached_island={reached_island} blows_used={len(blow_landings)}")
        if not reached_island:
            out = write_report(
                "l7_recorder_warp_full_route",
                {"walked_ok": True, "select_ok": True, "reached_island": False, "blow_landings": blow_landings, "writes": 0},
                tag=args.tag,
            )
            print(out)
            return
        _shot(obs_box[0], args.tag, total[0], read_snapshot(env.get_ram()), suffix="_island")

        # --- Phase 4: join hop 0x45 DOWN align_x=128 -> 0x55, then the
        # already-green LEVEL7_POND_HOPS tail from 0x55 onward. ---
        full_hops: tuple[ScreenHop, ...] = (
            ScreenHop(JOIN_TARGET, "DOWN", align_x=JOIN_ALIGN_X),
            ScreenHop(0x65, "DOWN", align_x=HOP_65_ALIGN_X),
        ) + LEVEL7_POND_HOPS[7:]
        print("tail hops:", [(hex(h.target), h.direction) for h in full_hops])
        handoff2 = handoff_from_ram(env.get_ram(), evidence="probe-recorder-warp-full-route-tail", verified=True)
        tail_ctl = PostLevel6OverworldController(handoff=handoff2, hops=full_hops, max_frames=TAIL_MAX_FRAMES)
        tail_ctl.bind_env(env)
        pond_ok = False
        snap = read_snapshot(env.get_ram())
        for _ in range(TAIL_MAX_FRAMES):
            snap = read_snapshot(env.get_ram())
            if on_level7_pond_hyp(snap):
                pond_ok = True
                break
            if tail_ctl.failed:
                break
            action = tail_ctl.step(snap)
            snap = step(action.action)
            if on_level7_pond_hyp(snap):
                pond_ok = True
                break
            if tail_ctl.failed:
                break
        print(
            f"pond_ok={pond_ok} tail_failed={tail_ctl.failed} screen=0x{snap.screen:02x} "
            f"xy=({snap.link_x},{snap.link_y}) mode={snap.mode} f={total[0]}"
        )
        print("tail notes(tail 15):", tail_ctl.notes[-15:])
        _shot(obs_box[0], args.tag, total[0], snap, suffix="_final")

        payload = {
            "walked_ok": True,
            "select_ok": True,
            "reached_island": True,
            "blow_landings": blow_landings,
            "join_hop": {"target": hex(JOIN_TARGET), "dir": "DOWN", "align_x": JOIN_ALIGN_X},
            "tail_hops": [
                {"target": hex(h.target), "dir": h.direction, "align_x": h.align_x, "align_y": h.align_y,
                 "y_band_lo": h.y_band_lo, "y_band_hi": h.y_band_hi}
                for h in full_hops
            ],
            "pond_ok": pond_ok,
            "tail_failed": tail_ctl.failed,
            "tail_report": tail_ctl.report(),
            "final_screen": hex(int(snap.screen)),
            "final_xy": [int(snap.link_x), int(snap.link_y)],
            "final_mode": int(snap.mode),
            "final_frame": total[0],
            "writes": 0,
        }
        out = write_report("l7_recorder_warp_full_route", payload, tag=args.tag)
        print(out)
        print(f"POND_OK={pond_ok} frames={total[0]}")
    finally:
        env.close()


if __name__ == "__main__":
    main()
