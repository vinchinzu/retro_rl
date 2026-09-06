"""H2 recon: dump the raw cart-WRAM $6530 tile map for one overworld screen,
reached by walking the already-greened ``POST_L6_TO_POND_HOPS`` prefix N
hops. Per HYGIENE / AGENTS traps, geometry must be measured from this
read-only tile map, never from direction-sensitive ``$049E`` sweeps.

No RAM pokes (writes=0).

    QT_QPA_PLATFORM=offscreen uv run python \
        nes/zelda_i/scratch/pond/probe_dump_ow_tilemap.py --hops 7 --tag l7_dump12
"""

from __future__ import annotations

import argparse

from retro_harness.env import make_env, reset_obs
from retro_harness.nes import nes_idle_action
from retro_harness.segment_runner import configure_headless, save_rgb_png
from zelda_i.dungeon.tilemap import read_room_tiles
from zelda_i.level7.entry import POST_L6_EXIT_STATE
from zelda_i.level7.pond import POST_L6_TO_POND_HOPS, PostLevel6OverworldController
from zelda_i.overworld.stitch import handoff_from_ram
from zelda_i.paths import GAME, GAME_DIR, RECORDINGS_DIR
from zelda_i.ram import PLAY_MODE, read_snapshot

WALK_MAX = 20_000


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--from-state", default=POST_L6_EXIT_STATE)
    parser.add_argument("--tag", default="ow_dump")
    parser.add_argument("--hops", type=int, default=7, help="POST_L6_TO_POND_HOPS length to walk")
    args = parser.parse_args()
    configure_headless()
    env = make_env(GAME, args.from_state, GAME_DIR, render_mode="rgb_array")
    obs_box: list = [None]
    total = [0]

    def step(action):
        obs, *_ = env.step(action)
        obs_box[0] = obs
        total[0] += 1
        return read_snapshot(env.get_ram())

    try:
        obs, _ = reset_obs(env)
        obs_box[0] = obs
        snap = step(nes_idle_action())

        truncated_hops = POST_L6_TO_POND_HOPS[: args.hops]
        target_screen = truncated_hops[-1].target if truncated_hops else 0x22
        handoff = handoff_from_ram(env.get_ram(), evidence="probe-dump-tilemap", verified=True)
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
            print("controller_report", ctl.report())
            return

        ram = env.get_ram()
        grid = read_room_tiles(ram)
        print(f"tile map for screen 0x{snap.screen:02x} ({grid.shape[0]}x{grid.shape[1]}):")
        for row_i, row in enumerate(grid):
            print(f"row{row_i:02d} " + " ".join(f"{v:02x}" for v in row))
        RECORDINGS_DIR.mkdir(parents=True, exist_ok=True)
        png = RECORDINGS_DIR / f"{args.tag}_s{snap.screen:02x}.png"
        save_rgb_png(obs_box[0], png)
        print(f"screenshot={png}")
    finally:
        env.close()


if __name__ == "__main__":
    main()
