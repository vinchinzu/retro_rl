"""Post-L6 0x22 west hop recon. One trial. Not a route claim.

    QT_QPA_PLATFORM=offscreen uv run python nes/zelda_i/scratch/probe_post_l6_to_pond.py --no-video --tag l7_p22w

Starts from Level6ExitOverworld (same 0x22 screen as the measured leave).
Handoff is built from live RAM so geometry can run; production still gates
on MEASURED_POST_L6_EXIT. Screenshot every screen change + final.
"""

from __future__ import annotations

import argparse

from retro_harness.env import make_env, reset_obs
from retro_harness.nes import nes_idle_action
from retro_harness.segment_runner import configure_headless, save_rgb_png
from zelda_i.level7.entry import (
    POST_L6_EXIT_STATE,
    PostLevel6OverworldController,
)
from zelda_i.overworld.stitch import handoff_from_ram
from zelda_i.paths import GAME, GAME_DIR, RECORDINGS_DIR
from zelda_i.ram import ADDR_FOOD, ADDR_WHISTLE, read_snapshot, read_u8
from zelda_i.runner import add_common_args, make_assist, write_report
from zelda_i.screen_glance import leftover_from_snapshot


def main() -> None:
    parser = argparse.ArgumentParser()
    add_common_args(parser, default_state=POST_L6_EXIT_STATE, default_tag="l7_p22w")
    parser.add_argument("--no-video", action="store_true", help="ignored; always rgb_array")
    parser.add_argument("--max-frames", type=int, default=0)
    args = parser.parse_args()
    configure_headless()
    assist = make_assist(args.infinite_life)
    env = make_env(GAME, args.from_state, GAME_DIR, render_mode="rgb_array")
    last_screen = None
    try:
        obs, _ = reset_obs(env)
        obs, *_ = env.step(nes_idle_action())
        ram0 = env.get_ram()
        snap0 = read_snapshot(ram0)
        handoff = handoff_from_ram(ram0, evidence="probe-l6exit-ow", verified=True)
        controller = PostLevel6OverworldController(handoff=handoff)
        if args.max_frames:
            controller.max_frames = args.max_frames
        controller.bind_env(env)
        print(
            f"start screen=0x{snap0.screen:02x} xy=({snap0.link_x},{snap0.link_y}) "
            f"mode={snap0.mode} whistle={int(read_u8(ram0, ADDR_WHISTLE))} "
            f"food={int(read_u8(ram0, ADDR_FOOD))} hops={[hex(h.target) for h in controller.hops]}"
        )
        for frame in range(controller.max_frames):
            snap = read_snapshot(env.get_ram())
            if last_screen is None or snap.screen != last_screen:
                RECORDINGS_DIR.mkdir(parents=True, exist_ok=True)
                png = RECORDINGS_DIR / f"{args.tag}_f{frame}_s{snap.screen:02x}.png"
                save_rgb_png(obs, png)
                print(
                    f"  screen 0x{snap.screen:02x} f={frame} "
                    f"xy=({snap.link_x},{snap.link_y}) mode={snap.mode} "
                    f"tile={snap.colliding_tile} png={png}"
                )
                last_screen = snap.screen
            action = controller.step(snap)
            if frame % 200 == 0 or (controller.stuck > 0 and controller.stuck % 250 == 0):
                png = RECORDINGS_DIR / f"{args.tag}_f{frame}_s{snap.screen:02x}.png"
                save_rgb_png(obs, png)
                print(
                    f"  f={frame} xy=({snap.link_x},{snap.link_y}) "
                    f"stuck={controller.stuck} tile={snap.colliding_tile} "
                    f"mode={snap.mode} reason={action.reason}"
                )
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
        png = RECORDINGS_DIR / f"{args.tag}_final.png"
        save_rgb_png(obs, png)
        payload = {
            **controller.report(),
            "from_state": args.from_state,
            "infinite_life": args.infinite_life,
            "leftover": leftover,
            "screenshot": str(png),
            "assist": assist.report() if assist is not None else None,
        }
        out = write_report("l7_post_l6_to_pond", payload, tag=args.tag)
        print(out)
        print(
            f"success={payload.get('success')} failed={payload.get('failed')} "
            f"phase={payload.get('phase')} frames={payload.get('frames')} "
            f"leftover={leftover} hop={payload.get('hop')} "
            f"notes={(payload.get('notes') or [])[-8:]}"
        )
    finally:
        env.close()


if __name__ == "__main__":
    main()
