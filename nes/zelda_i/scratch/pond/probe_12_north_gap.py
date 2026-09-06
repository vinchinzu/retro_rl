"""H2 recon: 0x12's `$6530` tile map shows a persistent non-mountain column
at tile-cols 6-7 (x=48-63) running the full north edge (rows0-9, y=64-136),
flanked by solid mountain (tiles 0xd8/0xd9/0xda/0xdb) elsewhere in that band
(see probe_dump_ow_tilemap.py dump, tag l7_dump12). Link already owns the
Stepladder (ADDR_LADDER=1) coming out of L6. This probe walks the greened
7-hop POST_L6_TO_POND_HOPS prefix to 0x12, then aligns to x~52 and pushes
UP through that column to see whether it is a laddered water-crossing to
screen 0x02 (the code comment's "NW blue ladder") or just a decorative
non-floor tile that still blocks.

No RAM pokes (writes=0).

    QT_QPA_PLATFORM=offscreen uv run python \
        nes/zelda_i/scratch/pond/probe_12_north_gap.py --no-video --tag l7_12north
"""

from __future__ import annotations

import argparse

from retro_harness.env import make_env, reset_obs
from retro_harness.nes import nes_action, nes_idle_action
from retro_harness.segment_runner import configure_headless, save_rgb_png
from zelda_i.level7.entry import POST_L6_EXIT_STATE
from zelda_i.level7.pond import POST_L6_TO_POND_HOPS, PostLevel6OverworldController
from zelda_i.overworld.stitch import handoff_from_ram
from zelda_i.paths import GAME, GAME_DIR, RECORDINGS_DIR
from zelda_i.ram import ADDR_LADDER, PLAY_MODE, read_snapshot, read_u8
from zelda_i.runner import add_common_args, make_assist, write_report

WALK_MAX = 20_000
GAP_X = 52
NAV_MAX = 4000


def main() -> None:
    parser = argparse.ArgumentParser()
    add_common_args(parser, default_state=POST_L6_EXIT_STATE, default_tag="l7_12north")
    parser.add_argument("--no-video", action="store_true", help="ignored; always rgb_array")
    parser.add_argument("--gap-x", type=int, default=GAP_X)
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
        print(f"ladder={int(read_u8(env.get_ram(), ADDR_LADDER))}")

        hops = POST_L6_TO_POND_HOPS  # full 7-hop prefix ends on 0x12
        handoff = handoff_from_ram(env.get_ram(), evidence="probe-12-north-gap", verified=True)
        ctl = PostLevel6OverworldController(handoff=handoff, hops=hops)
        ctl.bind_env(env)
        walked_ok = False
        for _ in range(WALK_MAX):
            snap = read_snapshot(env.get_ram())
            action = ctl.step(snap)
            snap = step(action.action)
            if int(snap.screen) == 0x12 and snap.mode == PLAY_MODE and not snap.transitioning:
                walked_ok = True
                break
            if ctl.failed:
                break
        print(
            f"walked_ok={walked_ok} ctl_failed={ctl.failed} "
            f"landed=0x{snap.screen:02x} xy=({snap.link_x},{snap.link_y}) f={total[0]}"
        )
        if not walked_ok:
            print("controller_report", ctl.report())
            return
        shot(snap, "_arrive")

        # Align x to the gap, then push UP through the whole column.
        last_xy = None
        stuck = 0
        notes = []
        for _ in range(NAV_MAX):
            snap = read_snapshot(env.get_ram())
            if snap.screen != 0x12:
                notes.append(f"left_0x12_to_0x{snap.screen:02x}@f{total[0]}")
                break
            if abs(snap.link_x - args.gap_x) > 3:
                btn = "RIGHT" if snap.link_x < args.gap_x else "LEFT"
                snap = step(nes_action(btn))
                continue
            xy = (int(snap.link_x), int(snap.link_y))
            if xy == last_xy:
                stuck += 1
                if stuck > 90:
                    notes.append(f"stuck_at_{xy}_tile{int(snap.colliding_tile)}")
                    break
            else:
                stuck = 0
            last_xy = xy
            snap = step(nes_action("UP"))
        final = read_snapshot(env.get_ram())
        print(
            f"final screen=0x{final.screen:02x} xy=({final.link_x},{final.link_y}) "
            f"mode={final.mode} tile={final.colliding_tile} notes={notes} f={total[0]}"
        )
        shot(final, "_final")
        out = write_report(
            "l7_12_north_gap",
            {
                "walked_ok": True,
                "gap_x": args.gap_x,
                "final_screen": hex(int(final.screen)),
                "final_xy": [int(final.link_x), int(final.link_y)],
                "final_mode": int(final.mode),
                "notes": notes,
                "writes": 0,
            },
            tag=args.tag,
        )
        print(out)
    finally:
        env.close()


if __name__ == "__main__":
    main()
