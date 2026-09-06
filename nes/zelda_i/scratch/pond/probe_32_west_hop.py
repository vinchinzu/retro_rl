"""H2 recon: same test as probe_32_west.py but using the project's own
generic ``ScreenHop`` / ``OverworldPathController`` engine (``align_and_push``
+ swing) instead of a raw scripted push -- the raw push got stuck fighting
the screen-entry zone (``align_and_push`` deliberately skips x/y alignment
near scroll-entry edges; a naive script does not).

0x32's tile map (probe_dump_ow_tilemap.py) column 0 (x=0-7) reads open
(tile 0x26) for rows 8-13 (y=128-183). This tries
``ScreenHop(0x31, "LEFT", y_band_lo=128, y_band_hi=183)`` appended after the
live ``0x22 DOWN -> 0x32`` hop.

No RAM pokes (writes=0).

    QT_QPA_PLATFORM=offscreen uv run python \
        nes/zelda_i/scratch/pond/probe_32_west_hop.py --no-video --tag l7_32w_hop
"""

from __future__ import annotations

import argparse

from retro_harness.env import make_env, reset_obs
from retro_harness.nes import nes_idle_action
from retro_harness.segment_runner import configure_headless, save_rgb_png
from zelda_i.dungeon.tilemap import read_room_tiles
from zelda_i.level7.entry import POST_L6_EXIT_STATE
from zelda_i.level7.overworld import bait_32_north_action
from zelda_i.level7.pond import POST_L6_TO_POND_HOPS, PostLevel6OverworldController
from zelda_i.overworld.graph import ScreenHop
from zelda_i.overworld.stitch import handoff_from_ram
from zelda_i.paths import GAME, GAME_DIR, RECORDINGS_DIR
from zelda_i.ram import PLAY_MODE, ZeldaSnapshot, read_snapshot
from zelda_i.runner import add_common_args, make_assist, write_report

WALK_MAX = 20_000


class _ProbeController(PostLevel6OverworldController):
    """Same engine, but always tries the known 0x32-north-mouth x=112
    realign fix (``bait_32_north_action``) regardless of the next hop's
    target -- production only wires it for ``hop.target == 0x33``. Dense
    frame logging (probe_32_dense) showed DOWN alone freezes forever at the
    (120,61) arrival point; the fix is documented in ``overworld.py`` as
    ``l7_bait_from_l6`` and just needs to fire for any hop leaving 0x32.
    """

    def _extra_hop_action(self, snap: ZeldaSnapshot, hop: ScreenHop):
        act = bait_32_north_action(snap, swing=self._swing)
        if act is not None:
            return act
        return super()._extra_hop_action(snap, hop)


def main() -> None:
    parser = argparse.ArgumentParser()
    add_common_args(parser, default_state=POST_L6_EXIT_STATE, default_tag="l7_32w_hop")
    parser.add_argument("--no-video", action="store_true")
    parser.add_argument("--y-lo", type=int, default=128)
    parser.add_argument("--y-hi", type=int, default=183)
    parser.add_argument("--target", type=lambda s: int(s, 0), default=0x31)
    parser.add_argument("--extra-south", type=lambda s: int(s, 0), default=None)
    parser.add_argument("--extra-south-x", type=int, default=120)
    args = parser.parse_args()
    configure_headless()
    assist = make_assist(args.infinite_life)
    env = make_env(GAME, args.from_state, GAME_DIR, render_mode="rgb_array")
    obs_box: list = [None]
    total = [0]

    def step(action):
        obs, *_ = env.step(action)
        obs_box[0] = obs
        total[0] += 1
        if assist is not None:
            assist.apply_env(env, frame=total[0])
        return read_snapshot(env.get_ram())

    def shot(snap, suffix=""):
        RECORDINGS_DIR.mkdir(parents=True, exist_ok=True)
        png = RECORDINGS_DIR / f"{args.tag}_f{total[0]}_s{snap.screen:02x}{suffix}.png"
        save_rgb_png(obs_box[0], png)
        return str(png)

    try:
        obs, _ = reset_obs(env)
        obs_box[0] = obs
        snap = step(nes_idle_action())

        hops = POST_L6_TO_POND_HOPS[:1] + (
            ScreenHop(args.target, "LEFT", y_band_lo=args.y_lo, y_band_hi=args.y_hi),
        )
        if args.extra_south is not None:
            hops = hops + (
                ScreenHop(args.extra_south, "DOWN", align_x=args.extra_south_x),
            )
        handoff = handoff_from_ram(env.get_ram(), evidence="probe-32-west-hop", verified=True)
        ctl = _ProbeController(handoff=handoff, hops=hops)
        ctl.bind_env(env)
        last_screen = None
        for _ in range(WALK_MAX):
            snap = read_snapshot(env.get_ram())
            if last_screen is None or snap.screen != last_screen:
                shot(snap, f"_s{snap.screen:02x}")
                last_screen = snap.screen
            action = ctl.step(snap)
            snap = step(action.action)
            if ctl.success or ctl.failed:
                break
        print(
            f"success={ctl.success} failed={ctl.failed} phase={getattr(ctl.phase,'name',ctl.phase)} "
            f"screen=0x{snap.screen:02x} xy=({snap.link_x},{snap.link_y}) "
            f"mode={snap.mode} tile={snap.colliding_tile} f={total[0]}"
        )
        print("notes(tail)=", ctl.notes[-10:])
        if int(snap.screen) == args.target:
            for _ in range(1000):
                snap = read_snapshot(env.get_ram())
                if snap.mode == PLAY_MODE and not snap.transitioning:
                    break
                snap = step(nes_idle_action())
            grid = read_room_tiles(env.get_ram())
            print(f"--- screen 0x{snap.screen:02x} tile map ---")
            for row_i, row in enumerate(grid):
                print(f"row{row_i:02d} " + " ".join(f"{v:02x}" for v in row))
            shot(snap, "_settled")
        out = write_report(
            "l7_32_west_hop",
            {
                "success": ctl.success,
                "failed": ctl.failed,
                "final_screen": hex(int(snap.screen)),
                "final_xy": [int(snap.link_x), int(snap.link_y)],
                "writes": 0,
            },
            tag=args.tag,
        )
        print(out)
    finally:
        env.close()


if __name__ == "__main__":
    main()
