"""Geometry-only walk PostSwordStart -> Demon pond 0x42. Not a route claim.

    uv run python nes/zelda_i/scratch/run_l7_pond_from_start.py --tag l7_pond

Whistle is NOT required for this pass (drain/entry is a later chapter); it maps
the 0x53 -> 0x52 -> 0x42 overworld geometry that rr-dnp / rr-8t4.2 need.
"""

from __future__ import annotations

import argparse

from retro_harness.env import make_env, reset_obs
from retro_harness.nes import nes_idle_action
from retro_harness.segment_runner import configure_headless, save_rgb_png
from zelda_i.level7.overworld import OverworldToLevel7PondController
from zelda_i.paths import GAME, GAME_DIR, RECORDINGS_DIR
from zelda_i.ram import ADDR_FOOD, ADDR_WHISTLE, read_snapshot, read_u8
from zelda_i.runner import add_common_args, make_assist, write_report
from zelda_i.screen_glance import leftover_from_snapshot


def main() -> None:
    parser = argparse.ArgumentParser()
    add_common_args(parser, default_state="PostSwordStart", default_tag="l7_pond")
    args = parser.parse_args()
    configure_headless()
    controller = OverworldToLevel7PondController()
    assist = make_assist(args.infinite_life)
    env = make_env(GAME, args.from_state, GAME_DIR, render_mode="rgb_array")
    last_screen = None
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
        out = write_report("l7_pond_from_start", payload, tag=args.tag)
        print(out)
        print(
            f"success={payload.get('success')} phase={payload.get('phase')} "
            f"frames={payload.get('frames')} leftover={leftover} "
            f"hop={payload.get('hop')} notes={(payload.get('notes') or [])[-8:]}"
        )
    finally:
        env.close()


if __name__ == "__main__":
    main()
