"""rr-6o7.1 recon: north-strip 0x42 west ring + reverse hops to 0x6D.

Loads Level7Entrance (same north-strip geometry as the Survival leftover),
skips the inventory handoff (fixture TF/candle do not match 0x7F/2), and
drives PostLevel7ToBushController hops. Success is settled OW 0x6D.

Not a route claim. Do not STATUS.

    QT_QPA_PLATFORM=offscreen uv run python \
        nes/zelda_i/scratch/pond/probe_42_to_6d.py --tag l8_42to6d
"""

from __future__ import annotations

import argparse
import json

from retro_harness.env import make_env, reset_obs
from retro_harness.nes import nes_action, nes_idle_action
from retro_harness.segment_runner import configure_headless, save_rgb_png
from zelda_i.level8.entry import PostLevel7ToBushController
from zelda_i.level8.overworld import L7_POND_TO_LEVEL8_BUSH_HOPS
from zelda_i.paths import GAME, GAME_DIR, RECORDINGS_DIR
from zelda_i.ram import PLAY_MODE, read_snapshot
from zelda_i.runner import add_common_args, make_assist

POND_SCREEN = 0x42
BUSH_SCREEN = 0x6D
EXIT_MAX = 800


def main() -> None:
    parser = argparse.ArgumentParser()
    add_common_args(parser, default_state="Level7Entrance", default_tag="l8_42to6d")
    args = parser.parse_args()
    configure_headless()
    assist = make_assist(args.infinite_life)
    env = make_env(GAME, args.from_state, GAME_DIR, render_mode="rgb_array")
    obs_box: list = [None]
    total = [0]
    transitions: list[dict] = []

    def step_raw(btn: str | None):
        obs, *_ = env.step(nes_idle_action() if not btn else nes_action(btn))
        obs_box[0] = obs
        total[0] += 1
        if assist is not None:
            assist.apply_env(env, frame=total[0])
        return read_snapshot(env.get_ram())

    def shot(label: str, snap) -> str:
        RECORDINGS_DIR.mkdir(parents=True, exist_ok=True)
        path = RECORDINGS_DIR / (
            f"{args.tag}_{label}_f{total[0]}_L{snap.level}_s{snap.screen:02x}.png"
        )
        save_rgb_png(obs_box[0], path)
        return str(path)

    try:
        obs, _ = reset_obs(env)
        obs_box[0] = obs
        snap = step_raw(None)
        for _ in range(EXIT_MAX):
            snap = read_snapshot(env.get_ram())
            if (
                snap.level == 0
                and snap.mode == PLAY_MODE
                and snap.screen == POND_SCREEN
                and not snap.transitioning
            ):
                break
            snap = step_raw("DOWN")
        print(f"settled 0x42 ({snap.link_x},{snap.link_y}) f={total[0]}")
        shot("settled", snap)

        ctl = PostLevel7ToBushController(hops=L7_POND_TO_LEVEL8_BUSH_HOPS)
        ctl.bind_env(env)
        ctl._handoff_checked = True
        last = (snap.level, snap.screen)
        reached = False
        for _ in range(ctl.max_frames):
            snap = read_snapshot(env.get_ram())
            action = ctl.step(snap)
            obs, *_ = env.step(action.action)
            obs_box[0] = obs
            total[0] += 1
            if assist is not None:
                assist.apply_env(env, frame=total[0])
            snap = read_snapshot(env.get_ram())
            key = (snap.level, snap.screen)
            if key != last:
                shot(f"t{len(transitions):02d}", snap)
                transitions.append(
                    {
                        "f": total[0],
                        "screen": int(snap.screen),
                        "xy": [int(snap.link_x), int(snap.link_y)],
                        "mode": int(snap.mode),
                        "reason": action.reason,
                    }
                )
                print(
                    f"t{len(transitions):02d} f={total[0]} "
                    f"s=0x{snap.screen:02x} ({snap.link_x},{snap.link_y}) "
                    f"{action.reason}"
                )
                last = key
            if (
                snap.level == 0
                and snap.mode == PLAY_MODE
                and snap.screen == BUSH_SCREEN
                and not snap.transitioning
            ):
                reached = True
                break
            if ctl.phase.name == "FAILED":
                break

        shot("final", snap)
        payload = {
            "bead": "rr-6o7.1",
            "from_state": args.from_state,
            "reached_0x6d": reached,
            "end": {
                "screen": int(snap.screen),
                "xy": [int(snap.link_x), int(snap.link_y)],
                "mode": int(snap.mode),
            },
            "ctl": ctl.report(),
            "transitions": transitions,
            "frames": total[0],
            "writes": 0,
            "route_eligible": False,
        }
        out = RECORDINGS_DIR / f"{args.tag}.json"
        out.write_text(json.dumps(payload, indent=2) + "\n")
        print("reached", reached, "end", payload["end"], "notes", ctl.notes[-8:])
        print("wrote", out)
    finally:
        env.close()


if __name__ == "__main__":
    main()
