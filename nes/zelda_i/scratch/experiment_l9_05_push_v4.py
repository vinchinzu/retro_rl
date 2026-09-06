"""v4: ignore wizzrobes entirely (room 0x05 swarm cannot be reliably killed
by scripted chase-attack -- 12000f, 0 kills in v3), rely on
UnlimitedHealthAssist to survive contact/bolt damage and just keep
re-asserting the correct navigation direction every frame. Latch-based
push (no re-checked threshold), tolerance not exact-equality for x.

    QT_QPA_PLATFORM=offscreen uv run python \
        nes/zelda_i/scratch/experiment_l9_05_push_v4.py
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


class State:
    cleared_stand_off = False


def policy_v4(snap, st: State):
    block = next((o for o in snap.objects if o.type_id == 0x68 or o.slot == 11), None)
    if block is not None and block.y > STAIRS_05_PUSH_BLOCK_Y:
        if abs(snap.link_x - STAIRS_05_PUSH_X) > 2:
            d = "LEFT" if snap.link_x > STAIRS_05_PUSH_X else "RIGHT"
            return FrameAction(nes_action(d), "v4_align_x")
        if not st.cleared_stand_off:
            if snap.link_y < 165:
                return FrameAction(nes_action("DOWN"), "v4_recenter_y")
            st.cleared_stand_off = True
        return FrameAction(nes_action("UP"), "v4_push_up")

    if snap.link_x < STAIRS_05_STAIR_X:
        if snap.link_y < 173:
            return FrameAction(nes_action("DOWN"), "v4_walk_stair_south_aisle")
        return FrameAction(nes_action("RIGHT"), "v4_walk_stair_x")
    if snap.link_y > STAIRS_05_STAIR_Y:
        return FrameAction(nes_action("UP"), "v4_walk_stair_y")
    return FrameAction(nes_action("UP"), "v4_stand_on_stairs")


def main() -> int:
    configure_headless()
    assist = UnlimitedHealthAssist(enabled=True)
    env = make_env(GAME, STATE_NAME, GAME_DIR, render_mode="rgb_array")
    reset_obs(env)

    st = State()
    last_key = None
    max_frames = 12000
    for i in range(max_frames):
        snap = read_snapshot(env.get_ram())
        if snap.mode != PLAY_MODE:
            act = FrameAction(nes_idle_action(), f"wait_mode_{snap.mode}")
        elif snap.screen != STAIRS_05_ORIGIN:
            print(f"LEFT ROOM f{i}: screen=0x{snap.screen:02x} xy=({snap.link_x},{snap.link_y})")
            env.close()
            return 0
        else:
            act = policy_v4(snap, st)
        key = (act.reason, snap.link_x, snap.link_y)
        if key != last_key:
            block = next((o for o in snap.objects if o.type_id == 0x68 or o.slot == 11), None)
            print(f"f{i}: xy=({snap.link_x},{snap.link_y}) block_y={block.y if block else None} act={act.reason}")
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
