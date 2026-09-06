"""Fast-iteration probe for the 0x38 -> 0x28 oscillation (rr-sz8.5 follow-on).

Loads ``Level8OWLeaveLive`` (byte-identical to MEASURED_POST_L8_HANDOFF),
drives Level9PostL8OverworldController to reach 0x38, then holds raw UP
and dumps per-frame (x, y, tile) to find the real convergence pattern.

    QT_QPA_PLATFORM=offscreen uv run python \
        nes/zelda_i/scratch/probe_l9_38_bridge.py
"""

from __future__ import annotations

from retro_harness.env import make_env, reset_obs
from retro_harness.nes import nes_action
from retro_harness.segment_runner import configure_headless
from zelda_i.assist import UnlimitedHealthAssist
from zelda_i.level9.dungeon import MEASURED_POST_L8_HANDOFF
from zelda_i.level9.overworld import Level9PostL8OverworldController
from zelda_i.paths import GAME, GAME_DIR
from zelda_i.ram import read_snapshot


def main() -> int:
    configure_headless()
    assist = UnlimitedHealthAssist(enabled=True)
    env = make_env(GAME, "Level8OWLeaveLive", GAME_DIR, render_mode="rgb_array")
    reset_obs(env)
    ctl = Level9PostL8OverworldController(handoff=MEASURED_POST_L8_HANDOFF)
    ctl.bind_env(env)

    frame = 0
    for _ in range(ctl.max_frames):
        snap = read_snapshot(env.get_ram())
        if snap.screen == 0x38:
            break
        act = ctl.step(snap)
        env.step(act.action)
        frame += 1
        assist.apply_env(env, frame=frame)
        if ctl.failed or ctl.success:
            print(f"controller ended early: failed={ctl.failed} success={ctl.success} "
                  f"reason={ctl.blocked_reason} screen=0x{snap.screen:02x} xy=({snap.link_x},{snap.link_y})")
            return 1

    print(f"reached 0x38 at frame {frame}, xy=({snap.link_x},{snap.link_y})")

    # Now hand control to the real controller for N frames and dump every frame.
    last_xy = None
    for i in range(2000):
        snap = read_snapshot(env.get_ram())
        if snap.screen != 0x38:
            print(f"f{i}: LEFT SCREEN -> 0x{snap.screen:02x} xy=({snap.link_x},{snap.link_y})")
            return 0
        act = ctl.step(snap)
        xy = (snap.link_x, snap.link_y)
        if xy != last_xy:
            print(f"f{i}: xy={xy} tile=0x{snap.colliding_tile:02x} act={act.reason}")
            last_xy = xy
        env.step(act.action)
        frame += 1
        assist.apply_env(env, frame=frame)
        if ctl.failed:
            print(f"FAILED at f{i}: {ctl.blocked_reason}")
            break

    env.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
