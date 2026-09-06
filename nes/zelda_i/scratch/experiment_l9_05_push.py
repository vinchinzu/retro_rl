"""Experiment with fixed push-block policies for level9_stairs_05 against the
real power-on pin L9Room05EntryReal, before committing a source change.

    QT_QPA_PLATFORM=offscreen uv run python \
        nes/zelda_i/scratch/experiment_l9_05_push.py
"""
from __future__ import annotations

from retro_harness.input_script import FrameAction
from retro_harness.env import make_env, reset_obs
from retro_harness.nes import nes_action, nes_idle_action
from retro_harness.segment_runner import configure_headless
from zelda_i.assist import UnlimitedHealthAssist
from zelda_i.level9.prefix import (
    STAIRS_05_ORIGIN, STAIRS_05_PUSH_BLOCK_Y, STAIRS_05_PUSH_X,
    STAIRS_05_STAIR_X, STAIRS_05_STAIR_Y,
)
from zelda_i.paths import GAME, GAME_DIR
from zelda_i.ram import PLAY_MODE, read_snapshot

STATE_NAME = "L9Room05EntryReal"


def policy_v2(snap, push_attempts):
    block = next((o for o in snap.objects if o.type_id == 0x68 or o.slot == 11), None)
    live_wizz = [o for o in snap.objects if o.type_id in (0x23, 0x24) and o.hp > 0]

    if block is not None and block.y > STAIRS_05_PUSH_BLOCK_Y:
        # Align x first (unconditional), then get to a safe stand-off y
        # south of the block, then push UP. Do not treat "x aligned" alone
        # as "ready to push" -- also require y in the push band.
        if abs(snap.link_x - STAIRS_05_PUSH_X) > 2:
            d = "LEFT" if snap.link_x > STAIRS_05_PUSH_X else "RIGHT"
            return FrameAction(nes_action(d), "v2_align_x"), push_attempts
        if snap.link_y < 165:
            return FrameAction(nes_action("DOWN"), "v2_recenter_y"), push_attempts
        push_attempts += 1
        return FrameAction(nes_action("UP"), "v2_push_up"), push_attempts

    if snap.link_x < STAIRS_05_STAIR_X:
        if snap.link_y < 173:
            return FrameAction(nes_action("DOWN"), "v2_walk_stair_south_aisle"), push_attempts
        return FrameAction(nes_action("RIGHT"), "v2_walk_stair_x"), push_attempts
    if snap.link_y > STAIRS_05_STAIR_Y:
        return FrameAction(nes_action("UP"), "v2_walk_stair_y"), push_attempts
    return FrameAction(nes_action("UP"), "v2_stand_on_stairs"), push_attempts


def main() -> int:
    configure_headless()
    assist = UnlimitedHealthAssist(enabled=True)
    env = make_env(GAME, STATE_NAME, GAME_DIR, render_mode="rgb_array")
    reset_obs(env)

    push_attempts = 0
    last_key = None
    max_frames = 8000
    for i in range(max_frames):
        snap = read_snapshot(env.get_ram())
        if snap.mode != PLAY_MODE:
            act = FrameAction(nes_idle_action(), f"wait_mode_{snap.mode}")
        elif snap.screen != STAIRS_05_ORIGIN:
            print(f"LEFT ROOM f{i}: screen=0x{snap.screen:02x} mode={snap.mode} "
                  f"xy=({snap.link_x},{snap.link_y})")
            break
        else:
            act, push_attempts = policy_v2(snap, push_attempts)
        key = (act.reason, snap.link_x, snap.link_y)
        if key != last_key:
            block = next((o for o in snap.objects if o.type_id == 0x68 or o.slot == 11), None)
            print(f"f{i}: xy=({snap.link_x},{snap.link_y}) block_y={block.y if block else None} "
                  f"act={act.reason}")
            last_key = key
        env.step(act.action)
        assist.apply_env(env, frame=i)
    else:
        snap = read_snapshot(env.get_ram())
        print(f"ran out of {max_frames} frames: screen=0x{snap.screen:02x} xy=({snap.link_x},{snap.link_y})")
    env.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
