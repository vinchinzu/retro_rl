"""Find the 0x64 -> 0x54 north gap. Geometry recon, not a route claim.

    uv run python nes/zelda_i/scratch/probe_64_north_to_54.py --tag p64

Drives PostSwordStart -> 0x64 with the stock pond controller, then sweeps
UP-pushes across x to locate the north screen transition to 0x54.
"""

from __future__ import annotations

import argparse

from retro_harness.env import make_env, reset_obs
from retro_harness.nes import nes_action, nes_idle_action
from retro_harness.segment_runner import configure_headless, save_rgb_png
from zelda_i.level7.overworld import OverworldToLevel7PondController
from zelda_i.paths import GAME, GAME_DIR, RECORDINGS_DIR
from zelda_i.ram import read_snapshot
from zelda_i.runner import add_common_args, make_assist, write_report


def _walk_to(env, assist, tx, ty, *, tol=6, max_f=600):
    for _ in range(max_f):
        snap = read_snapshot(env.get_ram())
        if snap.screen != 0x64:
            return snap
        dx, dy = tx - snap.link_x, ty - snap.link_y
        if abs(dx) <= tol and abs(dy) <= tol:
            return snap
        if abs(dx) > tol:
            act = nes_action("RIGHT" if dx > 0 else "LEFT")
        else:
            act = nes_action("DOWN" if dy > 0 else "UP")
        env.step(act)
        if assist is not None:
            assist.apply_env(env, frame=0)
    return read_snapshot(env.get_ram())


def main() -> None:
    parser = argparse.ArgumentParser()
    add_common_args(parser, default_state="PostSwordStart", default_tag="p64")
    args = parser.parse_args()
    configure_headless()
    assist = make_assist(args.infinite_life)
    env = make_env(GAME, args.from_state, GAME_DIR, render_mode="rgb_array")
    RECORDINGS_DIR.mkdir(parents=True, exist_ok=True)
    results = []
    try:
        obs, _ = reset_obs(env)
        env.step(nes_idle_action())
        ctl = OverworldToLevel7PondController()
        # Drive until we settle on 0x64.
        for _ in range(20000):
            snap = read_snapshot(env.get_ram())
            if snap.screen == 0x64 and snap.mode == 5 and not snap.transitioning:
                break
            obs, *_ = env.step(ctl.step(snap).action)
            if assist is not None:
                assist.apply_env(env, frame=0)
            if ctl.failed and snap.screen != 0x64:
                break
        arrived = read_snapshot(env.get_ram())
        save_rgb_png(obs, RECORDINGS_DIR / f"{args.tag}_arrived_s{arrived.screen:02x}.png")
        print(f"arrived screen=0x{arrived.screen:02x} xy=({arrived.link_x},{arrived.link_y})")

        if arrived.screen == 0x64:
            for tx in range(24, 232, 12):
                # Re-stage: drop to mid-band then walk to (tx, 100).
                _walk_to(env, assist, tx, 120)
                s = _walk_to(env, assist, tx, 96)
                if s.screen != 0x64:
                    results.append({"tx": tx, "left_at": "restage", "screen": f"0x{s.screen:02x}"})
                    break
                hit = None
                for f in range(50):
                    env.step(nes_action("UP"))
                    if assist is not None:
                        assist.apply_env(env, frame=0)
                    s = read_snapshot(env.get_ram())
                    if s.screen != 0x64:
                        hit = f"0x{s.screen:02x}@f{f}_y{s.link_y}"
                        break
                results.append({"tx": tx, "final_xy": [s.link_x, s.link_y], "hit": hit})
                print(f"  tx={tx} hit={hit} xy=({s.link_x},{s.link_y})")
                if s.screen == 0x54:
                    save_rgb_png(
                        read_snapshot(env.get_ram()) and obs,
                        RECORDINGS_DIR / f"{args.tag}_hit54_tx{tx}.png",
                    )
                    break
                if s.screen != 0x64:
                    # wandered elsewhere; walk back
                    _walk_to(env, assist, 120, 130)
        save_rgb_png(obs, RECORDINGS_DIR / f"{args.tag}_final.png")
        payload = {"from_state": args.from_state, "arrived": [arrived.screen, arrived.link_x, arrived.link_y], "sweep": results}
        out = write_report("probe_64_north", payload, tag=args.tag)
        print(out)
        print("sweep:", results)
    finally:
        env.close()


if __name__ == "__main__":
    main()
