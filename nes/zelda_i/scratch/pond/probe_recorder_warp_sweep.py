"""H1 recon (round 2): does the Recorder warp fire away from the L6 entrance
screen 0x22 itself?

Round 1 (``probe_recorder_warp.py``, tag rw1) blew the recorder 8x while
standing on 0x22 (a dungeon-entrance overworld screen) and observed NO
change at all: screen/x/y/mode stayed byte-identical across every blow.
Hypothesis: dungeon-entrance overworld screens suppress the recorder-warp
check entirely (consistent with the pond 0x42 special-casing a drain
instead of a warp -- entrance-flagged screens may just no-op).

This probe walks the already-greened ``POST_L6_TO_POND_HOPS`` prefix N hops
(0..7, i.e. up to and including 0x12) using the real
``PostLevel6OverworldController`` hop table (same controller the spine
uses), stops as soon as that screen is reached, then repeats the exact same
pause-select + 12xB + idle recorder-blow recipe from ``probe_recorder_warp.py``
and logs whether the screen/level/x/y changed.

No RAM pokes (writes=0). Handoff is built from live RAM so the controller
can run outside the spine, same technique as probe_post_l6_to_pond.py.

    QT_QPA_PLATFORM=offscreen uv run python \
        nes/zelda_i/scratch/pond/probe_recorder_warp_sweep.py \
        --no-video --hops 1 --tag rw_h1
"""

from __future__ import annotations

import argparse
from dataclasses import replace
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
from zelda_i.screen_glance import leftover_from_snapshot

WALK_MAX = 20_000
BLOW_PRESSES = 12
BLOW_WAIT_FRAMES = 400
POST_BLOW_SETTLE = 120
BACKOFF_MAX = 400
N_BLOWS_DEFAULT = 4


def _glance(env) -> dict[str, Any]:
    ram = env.get_ram()
    snap = read_snapshot(ram)
    leftover = leftover_from_snapshot(snap)
    leftover.update(
        {
            "level": int(snap.level),
            "whistle": int(read_u8(ram, ADDR_WHISTLE)),
            "selected_item": int(read_u8(ram, ADDR_SELECTED_ITEM)),
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
    add_common_args(parser, default_state=POST_L6_EXIT_STATE, default_tag="rw_sweep")
    parser.add_argument("--no-video", action="store_true", help="ignored; always rgb_array")
    parser.add_argument(
        "--hops", type=int, default=1, help="how many POST_L6_TO_POND_HOPS to walk first"
    )
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

    try:
        obs, _ = reset_obs(env)
        obs_box[0] = obs
        snap = step(nes_idle_action())

        truncated_hops = POST_L6_TO_POND_HOPS[: args.hops]
        target_screen = truncated_hops[-1].target if truncated_hops else 0x22
        handoff = handoff_from_ram(env.get_ram(), evidence="probe-recorder-warp", verified=True)
        ctl = PostLevel6OverworldController(handoff=handoff, hops=truncated_hops)
        ctl.bind_env(env)
        walked_ok = False
        last_screen = None
        for _ in range(WALK_MAX):
            snap = read_snapshot(env.get_ram())
            if last_screen is None or snap.screen != last_screen:
                _shot(obs_box[0], args.tag, total[0], snap, suffix="_walk")
                last_screen = snap.screen
            action = ctl.step(snap)
            snap = step(action.action)
            if int(snap.screen) == target_screen and not ctl.failed:
                # PostLevel6OverworldController only reports success on the
                # pond; for a truncated table treat "reached target screen,
                # standing, not transitioning" as walked_ok.
                if snap.mode == PLAY_MODE and not snap.transitioning:
                    walked_ok = True
                    break
            if ctl.failed:
                break
        print(
            f"walked_ok={walked_ok} ctl_failed={ctl.failed} target=0x{target_screen:02x} "
            f"landed=0x{snap.screen:02x} xy=({snap.link_x},{snap.link_y}) f={total[0]}"
        )
        if not walked_ok:
            payload = {
                "hypothesis": "H1_recorder_warp_sweep",
                "hops_walked": args.hops,
                "target_screen": hex(target_screen),
                "walked_ok": False,
                "controller_report": ctl.report(),
                "writes": 0,
            }
            out = write_report("l7_recorder_warp_sweep", payload, tag=args.tag)
            print(out)
            return

        pre_blow = _glance(env)
        print(f"pre-blow on 0x{target_screen:02x}: {pre_blow}")

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
            payload = {
                "hypothesis": "H1_recorder_warp_sweep",
                "hops_walked": args.hops,
                "target_screen": hex(target_screen),
                "walked_ok": True,
                "select_ok": False,
                "select_fail_reason": selector.fail_reason,
                "writes": 0,
            }
            out = write_report("l7_recorder_warp_sweep", payload, tag=args.tag)
            print(out)
            return

        last_screen = snap.screen
        last_level = snap.level
        landings: list[dict[str, Any]] = []
        for blow_i in range(args.n_blows):
            pre = _glance(env)
            for _ in range(BLOW_PRESSES):
                snap = step(nes_action("B"))
            changed = False
            for _f in range(BLOW_WAIT_FRAMES):
                snap = read_snapshot(env.get_ram())
                if snap.screen != last_screen or snap.level != last_level:
                    changed = True
                    _shot(obs_box[0], args.tag, total[0], snap, suffix=f"_blow{blow_i}_change")
                    last_screen = snap.screen
                    last_level = snap.level
                snap = step(nes_idle_action())
            for _ in range(POST_BLOW_SETTLE):
                snap = step(nes_idle_action())
            post = _glance(env)
            landings.append({"blow_index": blow_i, "pre": pre, "post": post, "changed": changed})
            print(
                f"blow {blow_i}: changed={changed} post level={post['level']} "
                f"screen=0x{post['screen']:02x} xy=({post['x']},{post['y']}) "
                f"mode={post['mode']} f={total[0]}"
            )
            _shot(obs_box[0], args.tag, total[0], snap, suffix=f"_landing{blow_i}")
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
            "hypothesis": "H1_recorder_warp_sweep",
            "hops_walked": args.hops,
            "target_screen": hex(target_screen),
            "walked_ok": True,
            "select_ok": True,
            "pre_blow": pre_blow,
            "landings": landings,
            "final": _glance(env),
            "writes": 0,
        }
        out = write_report("l7_recorder_warp_sweep", payload, tag=args.tag)
        print(out)
    finally:
        env.close()


if __name__ == "__main__":
    main()
