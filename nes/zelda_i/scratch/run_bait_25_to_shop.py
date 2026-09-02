"""Recon: 0x25 Armos grid → bait shop 0x34. Not a route claim.

Starts from the Level6ExitOverworld recon fixture, walks the fixture-live
0x22->0x25 bait prefix, then continues with hypothesis explore hops past 0x25
toward the source bait shop screen 0x34.  Screenshots every screen change and
every 200 stuck frames.

    uv run python nes/zelda_i/scratch/run_bait_25_to_shop.py --tag l7_bait_25x \
        --explore 0x35:DOWN:ax128 0x34:LEFT:ay141
"""

from __future__ import annotations

import argparse

from retro_harness.env import make_env, reset_obs
from retro_harness.nes import nes_idle_action
from retro_harness.segment_runner import configure_headless, save_rgb_png
from zelda_i.level7.entry import POST_L6_EXIT_STATE
from zelda_i.level7.overworld import POST_L6_TO_BAIT_HOPS, OverworldToBaitShopController
from zelda_i.overworld.graph import ScreenHop
from zelda_i.paths import GAME, GAME_DIR, RECORDINGS_DIR
from zelda_i.ram import ADDR_FOOD, ADDR_WHISTLE, read_snapshot, read_u8
from zelda_i.runner import add_common_args, make_assist, write_report
from zelda_i.screen_glance import leftover_from_snapshot


def _parse_hop(spec: str) -> ScreenHop:
    """"0x35:DOWN:ax128" / "0x34:LEFT:ay141" / "0x35:DOWN:yb150-170"."""
    parts = spec.split(":")
    target = int(parts[0], 16)
    direction = parts[1].upper()
    align_x = align_y = yb_lo = yb_hi = None
    for extra in parts[2:]:
        if extra.startswith("ax"):
            align_x = int(extra[2:])
        elif extra.startswith("ay"):
            align_y = int(extra[2:])
        elif extra.startswith("yb"):
            lo, hi = extra[2:].split("-")
            yb_lo, yb_hi = int(lo), int(hi)
    return ScreenHop(target, direction, align_x, align_y, yb_lo, yb_hi)


def main() -> None:
    parser = argparse.ArgumentParser()
    add_common_args(parser, default_state=POST_L6_EXIT_STATE, default_tag="l7_bait_25x")
    parser.add_argument(
        "--explore",
        nargs="*",
        default=["0x35:DOWN:ax128", "0x34:LEFT:ay141"],
        help="explore hops appended after the 0x22->0x25 prefix",
    )
    parser.add_argument("--max-frames", type=int, default=20000)
    args = parser.parse_args()
    configure_headless()

    explore = tuple(_parse_hop(s) for s in args.explore)
    hops = POST_L6_TO_BAIT_HOPS + explore
    controller = OverworldToBaitShopController(hops=hops, max_frames=args.max_frames)
    assist = make_assist(args.infinite_life)
    env = make_env(GAME, args.from_state, GAME_DIR, render_mode="rgb_array")
    last_screen = None
    shots = 0
    try:
        obs, _ = reset_obs(env)
        obs, *_ = env.step(nes_idle_action())
        for frame in range(controller.max_frames):
            snap = read_snapshot(env.get_ram())
            if last_screen is None or snap.screen != last_screen:
                RECORDINGS_DIR.mkdir(parents=True, exist_ok=True)
                png = RECORDINGS_DIR / f"{args.tag}_f{frame}_s{snap.screen:02x}.png"
                save_rgb_png(obs, png)
                last_screen = snap.screen
                shots += 1
            elif controller.stuck > 0 and controller.stuck % 200 == 0:
                png = (
                    RECORDINGS_DIR
                    / f"{args.tag}_f{frame}_s{snap.screen:02x}_stuck{controller.stuck}.png"
                )
                save_rgb_png(obs, png)
            action = controller.step(snap)
            obs, *_ = env.step(action.action)
            if assist is not None:
                assist.apply_env(env, frame=frame)
            if controller.success or controller.failed:
                break
        snap = read_snapshot(env.get_ram())
        ram = env.get_ram()
        leftover = leftover_from_snapshot(snap)
        leftover["food"] = int(read_u8(ram, ADDR_FOOD))
        leftover["whistle"] = int(read_u8(ram, ADDR_WHISTLE))
        leftover["rupees"] = int(snap.rupees)
        png = RECORDINGS_DIR / f"{args.tag}_final_s{snap.screen:02x}.png"
        save_rgb_png(obs, png)
        payload = {
            **controller.report(),
            "from_state": args.from_state,
            "explore_hops": list(args.explore),
            "leftover": leftover,
            "screenshot": str(png),
            "shots": shots,
        }
        out = write_report("l7_bait_25x", payload, tag=args.tag)
        print(out)
        print(
            f"success={payload['success']} failed={payload['failed']} "
            f"frames={payload['frames']} hop_index={payload['hop_index']} "
            f"leftover={leftover} notes={payload.get('notes')[-10:]}"
        )
    finally:
        env.close()


if __name__ == "__main__":
    main()
