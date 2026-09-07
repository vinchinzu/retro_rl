"""Pin the STAIRS_03 stall, then test what actually pushes room 0x03's 0x68.

At the stall Link sits at (96,149) holding UP with the block at (96,144) and
nothing moves, yet the uncapped run pushed the same block incidentally during
combat. This drives Link explicitly to a set of candidate push stands and
reports whether the block's y ever drops to <= 0x80.

    QT_QPA_PLATFORM=offscreen uv run python \
        nes/zelda_i/scratch/probe_l9_03_push_test.py
"""
from __future__ import annotations

from retro_harness.env import make_env, reset_obs, save_state
from retro_harness.segment_runner import configure_headless
from zelda_i.assist import UnlimitedHealthAssist
from zelda_i.level9.natural_path import PatraJoinPhase, make_natural_patra_join_controller
from zelda_i.level9.stairs import pushable_block
from zelda_i.paths import GAME, GAME_DIR
from zelda_i.ram import read_snapshot
from retro_harness.nes import nes_action, nes_idle_action

STATE_NAME = "L9Stairs03StallReal"


def run_to_stall(env, assist):
    ctl = make_natural_patra_join_controller()
    frame = 0
    held = 0
    while frame < 22000:
        snap = read_snapshot(env.get_ram())
        if ctl.phase is PatraJoinPhase.STAIRS_03:
            held += 1
            if held > 300:
                return frame
        env.step(ctl.step(snap).action)
        frame += 1
        assist.apply_env(env, frame=frame)
    return -1


def drive(env, assist, direction, frames, *, tag):
    """Hold one direction and report the block."""
    for i in range(frames):
        env.step(nes_action(direction) if direction else nes_idle_action())
        assist.apply_env(env, frame=i)
    snap = read_snapshot(env.get_ram())
    blk = pushable_block(snap)
    print(f"  {tag}: link=({snap.link_x},{snap.link_y}) "
          f"block={None if blk is None else (blk.x, blk.y)}", flush=True)
    return snap


def walk_to(env, assist, tx, ty, limit=400):
    for i in range(limit):
        snap = read_snapshot(env.get_ram())
        dx, dy = tx - int(snap.link_x), ty - int(snap.link_y)
        if abs(dx) <= 1 and abs(dy) <= 1:
            return True
        d = ("RIGHT" if dx > 0 else "LEFT") if abs(dx) > 1 else ("DOWN" if dy > 0 else "UP")
        env.step(nes_action(d))
        assist.apply_env(env, frame=i)
    return False


def main() -> int:
    configure_headless()
    env = make_env(GAME, "L9PostArrowsReal", GAME_DIR, render_mode="rgb_array")
    reset_obs(env)
    assist = UnlimitedHealthAssist(enabled=True)
    f = run_to_stall(env, assist)
    snap = read_snapshot(env.get_ram())
    blk = pushable_block(snap)
    print(f"stall at f{f} link=({snap.link_x},{snap.link_y}) "
          f"block={None if blk is None else (blk.x, blk.y)}", flush=True)
    save_state(env, GAME_DIR, GAME, STATE_NAME)
    stall_bytes = env.em.get_state()
    print(f"pinned {STATE_NAME}", flush=True)

    for tx, ty in ((96, 165), (96, 170), (96, 173), (88, 170), (104, 170)):
        env.em.set_state(stall_bytes)
        ok = walk_to(env, assist, tx, ty)
        snap = read_snapshot(env.get_ram())
        print(f"stand({tx},{ty}) reached={ok} actual=({snap.link_x},{snap.link_y})", flush=True)
        drive(env, assist, "UP", 240, tag=f"hold UP from ({tx},{ty})")
    env.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
