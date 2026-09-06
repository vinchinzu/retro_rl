"""H1 follow-up: does the Recorder-warp destination on a completed-dungeon
door screen (``0x45`` L4 island, ``0x74`` L3 door -- both one screen off the
already-green ``LEVEL7_POND_APPROACH_HOPS`` chain per
``level7/overworld.py:96-152``) actually let Link walk off toward the pond
band, or is it a dead end (raft water / boulder wall / re-enter the door)?

Reuses the exact ``probe_recorder_warp_cycle.py`` walk + pause-select +
blow-with-settle-detection recipe, but stops blowing as soon as the
destination screen equals ``--target`` (rather than blowing a fixed count),
then dumps the live ``$6530`` tile map for that screen (never ``$049E``) and
sweeps a list of ``align_x`` candidates for one join ``ScreenHop`` off the
requested edge, using the project's own ``OverworldPathController`` engine
(``PostLevel6OverworldController`` as a generic single-hop walker -- its
``_after_hops`` pond-only success check does not matter here; success is
read directly off RAM: ``screen == join_target and mode == PLAY_MODE and not
transitioning``).

No RAM pokes (writes=0). Never poke Whistle/Food/door/keys/position.

    QT_QPA_PLATFORM=offscreen uv run python \
        nes/zelda_i/scratch/pond/probe_recorder_warp_join.py \
        --no-video --target 0x45 --join-target 0x55 --join-dir DOWN \
        --candidates 128,120,136,112,144 --tag rw_join_45
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
DEFAULT_HOP_MAX_FRAMES = 2000


def _int_hex(s: str) -> int:
    return int(s, 0)


def _shot(obs, tag: str, frame: int, snap, suffix: str = "") -> str:
    RECORDINGS_DIR.mkdir(parents=True, exist_ok=True)
    png = RECORDINGS_DIR / (
        f"{tag}_f{frame}_L{snap.level}_s{snap.screen:02x}_m{snap.mode}{suffix}.png"
    )
    save_rgb_png(obs, png)
    return str(png)


def main() -> None:
    parser = argparse.ArgumentParser()
    add_common_args(parser, default_state=POST_L6_EXIT_STATE, default_tag="rw_join")
    parser.add_argument("--no-video", action="store_true", help="ignored; always rgb_array")
    parser.add_argument("--hops", type=int, default=4, help="POST_L6_TO_POND_HOPS to walk (4=0x24)")
    parser.add_argument("--face", choices=["UP", "DOWN"], default="DOWN")
    parser.add_argument("--max-blows", type=int, default=14)
    parser.add_argument("--target", type=_int_hex, required=True, help="dungeon door screen, e.g. 0x45")
    parser.add_argument("--join-target", type=_int_hex, required=True, help="screen to walk off into, e.g. 0x55")
    parser.add_argument("--join-dir", choices=["UP", "DOWN", "LEFT", "RIGHT"], required=True)
    parser.add_argument(
        "--candidates",
        default="128,120,136,112,144,96,160,80,176",
        help="comma-separated align_x (or align_y for LEFT/RIGHT joins) candidates, tried in order",
    )
    parser.add_argument("--hop-max-frames", type=int, default=DEFAULT_HOP_MAX_FRAMES)
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

        # --- Phase 1: walk the greened non-entrance prefix ---
        truncated_hops = POST_L6_TO_POND_HOPS[: args.hops]
        prefix_target = truncated_hops[-1].target if truncated_hops else 0x22
        handoff = handoff_from_ram(env.get_ram(), evidence="probe-recorder-warp-join", verified=True)
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
        print(
            f"prefix walked_ok={walked_ok} ctl_failed={ctl.failed} target=0x{prefix_target:02x} "
            f"landed=0x{snap.screen:02x} xy=({snap.link_x},{snap.link_y}) f={total[0]}"
        )
        if not walked_ok:
            out = write_report(
                "l7_recorder_warp_join",
                {"walked_ok": False, "controller_report": ctl.report(), "writes": 0},
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
        print(f"select_ok={select_ok} failed={selector.failed} reason={selector.fail_reason}")
        if not select_ok:
            out = write_report(
                "l7_recorder_warp_join",
                {"walked_ok": True, "select_ok": False, "writes": 0},
                tag=args.tag,
            )
            print(out)
            return

        # --- Phase 3: blow repeatedly, settling fully, until target screen ---
        landings: list[dict[str, Any]] = []
        reached_target = False
        for blow_i in range(args.max_blows):
            for _ in range(FACE_SETTLE):
                snap = step(nes_action(args.face))
            for _ in range(12):
                snap = step(nes_action("B"))
            post, settled = wait_settled(MAX_WAIT_PER_BLOW)
            entry = {
                "blow_index": blow_i,
                "post_screen": hex(int(post.screen)),
                "post_xy": [int(post.link_x), int(post.link_y)],
                "post_mode": int(post.mode),
                "post_level": int(post.level),
                "settled": settled,
                "frame": total[0],
            }
            landings.append(entry)
            print(
                f"blow {blow_i}: -> 0x{int(post.screen):02x} L{int(post.level)} "
                f"xy={entry['post_xy']} mode={entry['post_mode']} settled={settled} f={total[0]}"
            )
            if not settled:
                print("  did not settle within MAX_WAIT_PER_BLOW; stopping")
                break
            if int(post.level) != 0:
                print("  landed inside a dungeon/cave; stopping (no re-blow there)")
                break
            if int(post.screen) == args.target and post.mode == PLAY_MODE and not post.transitioning:
                reached_target = True
                break

        print(f"reached_target={reached_target} blows_used={len(landings)}")
        if not reached_target:
            out = write_report(
                "l7_recorder_warp_join",
                {
                    "walked_ok": True,
                    "select_ok": True,
                    "reached_target": False,
                    "landings": landings,
                    "writes": 0,
                },
                tag=args.tag,
            )
            print(out)
            return

        # --- Phase 4: dump the $6530 tile map for the landed screen ---
        landed = read_snapshot(env.get_ram())
        grid = read_room_tiles(env.get_ram())
        tile_rows: list[str] = []
        print(f"--- tile map for screen 0x{landed.screen:02x} ({grid.shape[0]}x{grid.shape[1]}) ---")
        for row_i, row in enumerate(grid):
            line = " ".join(f"{v:02x}" for v in row)
            tile_rows.append(line)
            print(f"row{row_i:02d} {line}")
        _shot(obs_box[0], args.tag, total[0], landed, suffix="_target")

        # --- Phase 5: sweep join-hop candidates from the live landing spot ---
        candidates = [int(c.strip(), 0) for c in args.candidates.split(",") if c.strip()]
        attempts: list[dict[str, Any]] = []
        join_ok = False
        aborted_dungeon = False
        for cand in candidates:
            pre = read_snapshot(env.get_ram())
            kwargs: dict[str, Any] = {}
            if args.join_dir in ("UP", "DOWN"):
                kwargs["align_x"] = cand
            else:
                kwargs["align_y"] = cand
            hop = ScreenHop(args.join_target, args.join_dir, **kwargs)
            handoff2 = handoff_from_ram(env.get_ram(), evidence="probe-recorder-warp-join-hop", verified=True)
            hop_ctl = PostLevel6OverworldController(handoff=handoff2, hops=(hop,), max_frames=args.hop_max_frames)
            hop_ctl.bind_env(env)
            hop_success = False
            entered_dungeon = False
            snap2 = pre
            for _ in range(args.hop_max_frames):
                snap2 = read_snapshot(env.get_ram())
                if int(snap2.screen) == args.join_target and snap2.mode == PLAY_MODE and not snap2.transitioning:
                    hop_success = True
                    break
                if int(snap2.level) != 0 or snap2.mode == 16:
                    entered_dungeon = True
                    break
                act = hop_ctl.step(snap2)
                snap2 = step(act.action)
                if int(snap2.screen) == args.join_target and snap2.mode == PLAY_MODE and not snap2.transitioning:
                    hop_success = True
                    break
                if int(snap2.level) != 0 or snap2.mode == 16:
                    entered_dungeon = True
                    break
            attempt = {
                "candidate": cand,
                "pre_xy": [int(pre.link_x), int(pre.link_y)],
                "hop_success": hop_success,
                "entered_dungeon": entered_dungeon,
                "leftover_screen": hex(int(snap2.screen)),
                "leftover_xy": [int(snap2.link_x), int(snap2.link_y)],
                "leftover_mode": int(snap2.mode),
                "frame": total[0],
                "ctl_notes_tail": hop_ctl.notes[-6:],
            }
            attempts.append(attempt)
            print(
                f"  candidate={cand}: success={hop_success} entered_dungeon={entered_dungeon} "
                f"leftover=0x{int(snap2.screen):02x} xy=({int(snap2.link_x)},{int(snap2.link_y)}) f={total[0]}"
            )
            _shot(
                obs_box[0],
                args.tag,
                total[0],
                snap2,
                suffix=f"_cand{cand}_{'ok' if hop_success else 'miss'}",
            )
            if hop_success:
                join_ok = True
                break
            if entered_dungeon:
                aborted_dungeon = True
                print("  entered a dungeon/door during sweep; stopping sweep (cannot safely continue)")
                break

        payload = {
            "walked_ok": True,
            "select_ok": True,
            "reached_target": True,
            "target": hex(args.target),
            "blows_used": len(landings),
            "face": args.face,
            "landings": landings,
            "tile_map_screen": hex(int(landed.screen)),
            "tile_map": tile_rows,
            "join_target": hex(args.join_target),
            "join_dir": args.join_dir,
            "join_ok": join_ok,
            "aborted_dungeon": aborted_dungeon,
            "attempts": attempts,
            "writes": 0,
            "final_frame": total[0],
        }
        out = write_report("l7_recorder_warp_join", payload, tag=args.tag)
        print(out)
        print(f"JOIN_OK={join_ok} frames={total[0]}")
    finally:
        env.close()


if __name__ == "__main__":
    main()
