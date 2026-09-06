"""Column sweep for the post-blast Spectacle Rock cave entrance (rr-sz8.5).

Level9SpectacleRockBombController reaches ROCK_ENTER (stand x=72, blast the
left rock, walk UP) but stalls forever at (72,165) -- ``left_rock_mouth_did_
not_enter``. This drives the real walk once from ``Level8OWLeaveLive`` up
through the blast, saves a clean savestate right at blast-complete
(``L9RockBlastClean``), then reloads that savestate once per candidate x
column and pushes UP for 120 frames to see how far north each column can
walk and whether any of them ever flips ``snap.level``.

Findings (2026-09-06): every tested column from x=16 to x=120 (8px steps)
fails to enter, always stopping on a solid tile (0xD4/0xD7/0xD9/0xDB/0xDD/
0xDE depending on column -- all distinct rock-pile tile ids, not open sand
like the 0x26 seen everywhere on the approach walk). More importantly: a
side-by-side pixel diff of ``save_rgb_png`` screenshots taken immediately
before the bomb and ~100 frames after (cropped to just the left rock-pile
region) shows **zero visual change** -- same silhouette, same tile pattern.
The bomb consumes exactly one bomb (14 -> 13) but does not destroy, shrink,
or open any hole in the pile. A second manual trial bombing directly under
the *right* pile (still fully intact in these screenshots too) also left it
unchanged. Health/damage telemetry never changes either, so this isn't
combat interference -- Link is simply standing at the base of ordinary,
non-bombable Death Mountain rock terrain.

This overturns the LEVEL9_ROUTE.md claim that "live recon reached 0x05 ...
and bombed the left rock to settle in room 0x76" -- that composed fixture
was evidently never actually produced by a genuine bomb-and-walk (or was
produced against a different screen/position). ``SCREEN_LEVEL9_ROCK_HYP``
(OW 0x05) and/or the coded stand position are likely just wrong: either
this is not the correct overworld screen for Spectacle Rock's secret
entrance at all, or the correct bombable formation is a visually distinct
small boulder pair (the "spectacles") elsewhere on this screen that hasn't
been located yet. Next step needs either ROM overworld-map data for the
real warp-tile location, or a systematic visual survey of screen 0x05 (and
its neighbors) for a rock formation distinct from these two large permanent
piles -- not another coordinate-tuning guess on the current hypothesis.
See rr-sz8.5 notes / LEVEL9_ROUTE.md for the writeup and evidence.

    QT_QPA_PLATFORM=offscreen uv run python \
        nes/zelda_i/scratch/probe_l9_05_entry_sweep.py
"""

from __future__ import annotations

from retro_harness.env import make_env, reset_obs, save_state
from retro_harness.nes import nes_action
from retro_harness.segment_runner import configure_headless
from zelda_i.assist import UnlimitedHealthAssist
from zelda_i.level9.dungeon import MEASURED_POST_L8_HANDOFF
from zelda_i.level9.overworld import (
    Level9PostL8OverworldController,
    Level9SpectacleRockBombController,
    SpectacleRockBombPhase,
)
from zelda_i.paths import GAME, GAME_DIR
from zelda_i.ram import read_snapshot

CLEAN_STATE = "L9RockBlastClean"


def _make_clean_state() -> None:
    """Drive the real walk once and pin a savestate right at blast-complete."""
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
    if ow.failed:
        raise RuntimeError(f"post-L8 overworld walk failed: {ow.blocked_reason}")

    bomb = Level9SpectacleRockBombController(handoff=MEASURED_POST_L8_HANDOFF)
    bomb.bind_env(env)
    for _ in range(bomb.max_frames):
        snap = read_snapshot(env.get_ram())
        act = bomb.step(snap)
        env.step(act.action)
        frame += 1
        assist.apply_env(env, frame=frame)
        if bomb.phase is SpectacleRockBombPhase.ROCK_ENTER and bomb.phase_frames == 1:
            save_state(env, GAME_DIR, GAME, CLEAN_STATE)
            env.close()
            return
        if bomb.failed:
            env.close()
            raise RuntimeError(f"bomb controller failed before blast: {bomb.failure}")
    env.close()
    raise RuntimeError("never reached ROCK_ENTER")


def _try_column(target_x: int) -> None:
    assist = UnlimitedHealthAssist(enabled=True)
    env = make_env(GAME, CLEAN_STATE, GAME_DIR, render_mode="rgb_array")
    reset_obs(env)
    frame = 0
    for _ in range(80):
        snap = read_snapshot(env.get_ram())
        if abs(snap.link_x - target_x) <= 2:
            break
        btn = "RIGHT" if target_x > snap.link_x else "LEFT"
        env.step(nes_action(btn))
        frame += 1
        assist.apply_env(env, frame=frame)
    start = read_snapshot(env.get_ram())
    min_y = start.link_y
    entered = False
    for _ in range(120):
        snap = read_snapshot(env.get_ram())
        if snap.level != 0:
            entered = True
            break
        min_y = min(min_y, snap.link_y)
        env.step(nes_action("UP"))
        frame += 1
        assist.apply_env(env, frame=frame)
    final = read_snapshot(env.get_ram())
    env.close()
    print(
        f"x_target={target_x}: start=({start.link_x},{start.link_y}) min_y={min_y} "
        f"final=({final.link_x},{final.link_y}) tile=0x{final.colliding_tile:02x} "
        f"level={final.level} entered={entered}"
    )


def main() -> int:
    _make_clean_state()
    for x in range(16, 121, 8):
        _try_column(x)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
