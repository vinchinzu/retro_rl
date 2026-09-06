"""H1 recon: does blowing the Recorder on the post-L6 overworld (0x22) warp
Link toward the Demon pond 0x42, or anywhere on the green west band?

Not a route claim. No RAM pokes (writes=0). Whistle is already owned coming
out of L6 (ADDR_WHISTLE=1); B-slot select goes through the shared
``dungeon.pause_select.PauseSelectController`` (never poke $0656). This
probe presses B (the actual Recorder blow) and idles, then logs the raw
level/screen/x/y/mode RAM every frame during the transition window plus a
screenshot on every screen change, repeated for several consecutive blows
so the destination CYCLE can be read off.

    QT_QPA_PLATFORM=offscreen uv run python \
        nes/zelda_i/scratch/pond/probe_recorder_warp.py --no-video --tag rw1
"""

from __future__ import annotations

import argparse
from typing import Any

from retro_harness.env import make_env, reset_obs
from retro_harness.nes import nes_action, nes_idle_action
from retro_harness.segment_runner import configure_headless, save_rgb_png
from zelda_i.dungeon.pause_select import B_SLOT_RECORDER, PauseSelectController
from zelda_i.level7.entry import POST_L6_EXIT_STATE
from zelda_i.level7.overworld import at_l6_cave_mouth
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
from zelda_i.screen_glance import leftover_from_snapshot

LEAVE_MOUTH_MAX = 400
BLOW_PRESSES = 12
BLOW_WAIT_FRAMES = 400
POST_BLOW_SETTLE = 120
BACKOFF_MAX = 400
N_BLOWS_DEFAULT = 8


def _glance(env) -> dict[str, Any]:
    ram = env.get_ram()
    snap = read_snapshot(ram)
    leftover = leftover_from_snapshot(snap)
    leftover.update(
        {
            "level": int(snap.level),
            "whistle": int(read_u8(ram, ADDR_WHISTLE)),
            "selected_item": int(read_u8(ram, ADDR_SELECTED_ITEM)),
            "triforce": hex(int(read_u8(ram, ADDR_TRIFORCE))),
            "mode": int(snap.mode),
        }
    )
    return leftover


def _shot(obs, tag: str, frame: int, snap, suffix: str = "") -> str:
    RECORDINGS_DIR.mkdir(parents=True, exist_ok=True)
    png = RECORDINGS_DIR / (
        f"{tag}_f{frame}_L{snap.level}_s{snap.screen:02x}_m{snap.mode}{suffix}.png"
    )
    save_rgb_png(obs, png)
    return str(png)


def main() -> None:
    parser = argparse.ArgumentParser()
    add_common_args(parser, default_state=POST_L6_EXIT_STATE, default_tag="rw1")
    parser.add_argument("--no-video", action="store_true", help="ignored; always rgb_array")
    parser.add_argument("--n-blows", type=int, default=N_BLOWS_DEFAULT)
    args = parser.parse_args()
    configure_headless()
    assist = make_assist(args.infinite_life)
    env = make_env(GAME, args.from_state, GAME_DIR, render_mode="rgb_array")
    obs_box: list = [None]
    total = [0]
    landings: list[dict[str, Any]] = []
    frame_log: list[dict[str, Any]] = []

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
        print(
            f"start screen=0x{snap.screen:02x} xy=({snap.link_x},{snap.link_y}) "
            f"mode={snap.mode} whistle={int(read_u8(env.get_ram(), ADDR_WHISTLE))}"
        )

        # Step 1: leave the L6 cave-mouth reentry box (DOWN), per brief.
        left_mouth = False
        for _ in range(LEAVE_MOUTH_MAX):
            if not at_l6_cave_mouth(snap):
                left_mouth = True
                break
            snap = step(nes_action("DOWN"))
        print(f"left_mouth={left_mouth} xy=({snap.link_x},{snap.link_y}) f={total[0]}")
        _shot(obs_box[0], args.tag, total[0], snap, suffix="_off_mouth")

        # Step 2: pause-select the Recorder onto B (shared PauseSelectController).
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
        print(
            f"select_ok={select_ok} failed={selector.failed} "
            f"reason={selector.fail_reason} selected_item="
            f"{int(read_u8(env.get_ram(), ADDR_SELECTED_ITEM))} f={total[0]}"
        )
        if not select_ok:
            payload = {
                "hypothesis": "H1_recorder_warp",
                "select_ok": False,
                "select_fail_reason": selector.fail_reason,
                "writes": 0,
            }
            out = write_report("l7_recorder_warp", payload, tag=args.tag)
            print(out)
            return

        # Step 3: blow the recorder N times, logging the destination each time.
        last_screen = snap.screen
        last_level = snap.level
        for blow_i in range(args.n_blows):
            pre = _glance(env)
            for _ in range(BLOW_PRESSES):
                snap = step(nes_action("B"))
            # Idle through the whirlwind / scroll, logging every mode/screen
            # change so the transition sequence is visible.
            for f in range(BLOW_WAIT_FRAMES):
                snap = read_snapshot(env.get_ram())
                if snap.screen != last_screen or snap.level != last_level:
                    entry = {
                        "blow": blow_i,
                        "frame": total[0],
                        "level": int(snap.level),
                        "screen": hex(int(snap.screen)),
                        "x": int(snap.link_x),
                        "y": int(snap.link_y),
                        "mode": int(snap.mode),
                    }
                    frame_log.append(entry)
                    _shot(obs_box[0], args.tag, total[0], snap, suffix=f"_blow{blow_i}")
                    last_screen = snap.screen
                    last_level = snap.level
                snap = step(nes_idle_action())
            for _ in range(POST_BLOW_SETTLE):
                snap = step(nes_idle_action())
            post = _glance(env)
            landing = {
                "blow_index": blow_i,
                "pre": pre,
                "post": post,
                "frame": total[0],
            }
            landings.append(landing)
            print(
                f"blow {blow_i}: pre_screen=0x{pre['screen']:02x} -> "
                f"post level={post['level']} screen=0x{post['screen']:02x} "
                f"xy=({post['x']},{post['y']}) mode={post['mode']} "
                f"whistle={post['whistle']} f={total[0]}"
            )
            _shot(obs_box[0], args.tag, total[0], snap, suffix=f"_landing{blow_i}")

            # If the whirlwind carried Link into a dungeon mouth/cave, back
            # off before the next blow so a fresh B-press does not just
            # replay inside the dungeon. Never poke position: only walk.
            if int(post["level"]) != 0 or int(post["mode"]) not in (PLAY_MODE,):
                backed_off = False
                for _ in range(BACKOFF_MAX):
                    snap = read_snapshot(env.get_ram())
                    if int(snap.level) == 0 and int(snap.mode) == PLAY_MODE:
                        backed_off = True
                        break
                    snap = step(nes_action("DOWN"))
                print(f"  post-warp backoff: backed_off={backed_off} f={total[0]}")
                if not backed_off:
                    break

        payload = {
            "hypothesis": "H1_recorder_warp",
            "select_ok": True,
            "n_blows_attempted": args.n_blows,
            "landings": landings,
            "transition_frame_log": frame_log,
            "final": _glance(env),
            "from_state": args.from_state,
            "writes": 0,
        }
        out = write_report("l7_recorder_warp", payload, tag=args.tag)
        print(out)
    finally:
        env.close()


if __name__ == "__main__":
    main()
