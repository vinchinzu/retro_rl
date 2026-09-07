"""Trace STAIRS_03 after the CLEAR_03 bail: Link parks at (96,149) holding UP
on the push column and the block never moves.

Dumps the pushable block and every live object while STAIRS_03 is active, to
tell "block object missing from the slots" apart from "Link is standing in the
y=144..172 dead zone of room03_stairs_step".

    QT_QPA_PLATFORM=offscreen uv run python \
        nes/zelda_i/scratch/probe_l9_stairs03_stall.py
"""
from __future__ import annotations

from retro_harness.env import make_env, reset_obs
from retro_harness.segment_runner import configure_headless
from zelda_i.assist import UnlimitedHealthAssist
from zelda_i.level9.natural_path import PatraJoinPhase, make_natural_patra_join_controller
from zelda_i.level9.stairs import (
    live_combat_objects, pushable_block, room03_stairs_step, room03_west_block_pushed,
)
from zelda_i.paths import GAME, GAME_DIR
from zelda_i.ram import read_snapshot

STATE_NAME = "L9PostArrowsReal"


def main() -> int:
    configure_headless()
    env = make_env(GAME, STATE_NAME, GAME_DIR, render_mode="rgb_array")
    obs, _ = reset_obs(env)
    assist = UnlimitedHealthAssist(enabled=True)
    ctl = make_natural_patra_join_controller()
    frame = 0
    entered = -1
    while not ctl.success and not ctl.failed and frame < 22000:
        snap = read_snapshot(env.get_ram())
        if ctl.phase is PatraJoinPhase.STAIRS_03:
            if entered < 0:
                entered = frame
                print(f"STAIRS_03 entered f{frame}", flush=True)
            if (frame - entered) % 300 == 0:
                blk = pushable_block(snap)
                act = room03_stairs_step(snap)
                print(
                    f"  +{frame - entered} xy=({snap.link_x},{snap.link_y}) "
                    f"block={None if blk is None else (blk.x, blk.y)} "
                    f"pushed={room03_west_block_pushed(snap)} "
                    f"doors=0x{snap.cur_opened_doors:02x} reason={act.reason} "
                    f"objs={[(hex(o.type_id), o.x, o.y, o.hp) for o in snap.objects if o.type_id]}",
                    flush=True,
                )
        act = ctl.step(snap)
        obs, *_ = env.step(act.action)
        frame += 1
        assist.apply_env(env, frame=frame)
    env.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
