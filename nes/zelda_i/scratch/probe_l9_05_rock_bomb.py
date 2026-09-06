"""Fast-iteration probe for Level9SpectacleRockBombController (rr-sz8.5).

Loads ``Level8OWLeaveLive``, drives the full post-L8 overworld walk to 0x05,
then drives the rock-bomb controller and dumps per-frame (x, y, tile, phase)
to diagnose the ROCK_BOTTOM_Y/ROCK_LEFT_X stall at (72,165) vs target y=173.

    QT_QPA_PLATFORM=offscreen uv run python \
        nes/zelda_i/scratch/probe_l9_05_rock_bomb.py
"""

from __future__ import annotations

from retro_harness.env import make_env, reset_obs
from retro_harness.segment_runner import configure_headless
from zelda_i.assist import UnlimitedHealthAssist
from zelda_i.level9.dungeon import MEASURED_POST_L8_HANDOFF
from zelda_i.level9.overworld import (
    Level9PostL8OverworldController,
    Level9SpectacleRockBombController,
)
from zelda_i.paths import GAME, GAME_DIR
from zelda_i.ram import read_snapshot


def main() -> int:
    configure_headless()
    assist = UnlimitedHealthAssist(enabled=True)
    env = make_env(GAME, "Level8OWLeaveLive", GAME_DIR, render_mode="rgb_array")
    reset_obs(env)

    ow = Level9PostL8OverworldController(handoff=MEASURED_POST_L8_HANDOFF)
    ow.bind_env(env)
    frame = 0
    for _ in range(ow.max_frames):
        snap = read_snapshot(env.get_ram())
        act = ow.step(snap)
        env.step(act.action)
        frame += 1
        assist.apply_env(env, frame=frame)
        if ow.failed or ow.success:
            break
    print(f"overworld walk: failed={ow.failed} success={ow.success} frame={frame} "
          f"screen=0x{snap.screen:02x}")
    if ow.failed:
        return 1

    bomb = Level9SpectacleRockBombController(handoff=MEASURED_POST_L8_HANDOFF)
    bomb.bind_env(env)
    last_key = None
    for i in range(bomb.max_frames):
        snap = read_snapshot(env.get_ram())
        act = bomb.step(snap)
        key = (bomb.phase, snap.link_x, snap.link_y)
        if key != last_key:
            print(f"f{i}: phase={bomb.phase} xy=({snap.link_x},{snap.link_y}) "
                  f"tile=0x{snap.colliding_tile:02x} act={act.reason}")
            last_key = key
        env.step(act.action)
        frame += 1
        assist.apply_env(env, frame=frame)
        if bomb.failed or bomb.success:
            print(f"DONE f{i}: failed={bomb.failed} success={bomb.success} "
                  f"reason={bomb.failure}")
            break
    else:
        print("bomb controller ran out of max_frames")

    env.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
