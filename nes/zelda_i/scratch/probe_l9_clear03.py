"""Trace the join's CLEAR_03 phase, which burned 10,500 of the 24,000-frame
join budget from the L9PostArrowsReal pin (f12418 -> f22947) and parked Link
at (144,165) for thousands of frames.

Logs every live combat object while CLEAR_03 is active, so we can see whether
the chase list (0x13, 0x14, 0x17) is missing the type that keeps the phase
from clearing.

    QT_QPA_PLATFORM=offscreen uv run python \
        nes/zelda_i/scratch/probe_l9_clear03.py
"""
from __future__ import annotations

from retro_harness.env import make_env, reset_obs
from retro_harness.segment_runner import configure_headless
from zelda_i.assist import UnlimitedHealthAssist
from zelda_i.level9.stairs import live_combat_objects
from zelda_i.level9.natural_path import PatraJoinPhase, make_natural_patra_join_controller
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
    while not ctl.success and not ctl.failed and frame < 30000:
        snap = read_snapshot(env.get_ram())
        if ctl.phase is PatraJoinPhase.CLEAR_03:
            if entered < 0:
                entered = frame
                print(f"CLEAR_03 entered f{frame}", flush=True)
            if (frame - entered) % 400 == 0:
                combat = live_combat_objects(snap)
                kept = [o for o in combat if o.type_id != 0x2B]
                print(
                    f"  +{frame - entered} xy=({snap.link_x},{snap.link_y}) "
                    f"all={[(hex(o.type_id), o.x, o.y, o.hp) for o in combat]} "
                    f"blocking={[hex(o.type_id) for o in kept]}",
                    flush=True,
                )
        elif entered >= 0:
            print(f"CLEAR_03 left f{frame} after {frame - entered} frames -> {ctl.phase.name}", flush=True)
            break
        act = ctl.step(snap)
        obs, *_ = env.step(act.action)
        frame += 1
        assist.apply_env(env, frame=frame)
    env.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
