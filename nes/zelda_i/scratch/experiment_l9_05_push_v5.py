"""v5: proper hitbox-checked wizzrobe combat (should_swing_at + backstep-when-
stuck, ported from level6.wizzrobe.Level6EastKeyController) instead of the v3
blind chase-and-mash-A (0 kills/12000f) and v4 ignore-them (knocked back to
~y=93 in a stable loop, never converges -- 12000f, 0 progress).

Full-clear before pushing: only enter the align/push phase once all live
Wizzrobes are dead, so nothing can knock Link off the push stand-off mid-push.

    QT_QPA_PLATFORM=offscreen uv run python \
        nes/zelda_i/scratch/experiment_l9_05_push_v5.py
"""
from __future__ import annotations

from retro_harness.input_script import FrameAction
from retro_harness.env import make_env, reset_obs
from retro_harness.nes import nes_action, nes_idle_action
from retro_harness.segment_runner import configure_headless
from zelda_i.assist import UnlimitedHealthAssist
from zelda_i.combat import should_swing_at
from zelda_i.level9.prefix import (
    STAIRS_05_ORIGIN, STAIRS_05_PUSH_BLOCK_Y, STAIRS_05_PUSH_X,
    STAIRS_05_STAIR_X, STAIRS_05_STAIR_Y,
)
from zelda_i.paths import GAME, GAME_DIR
from zelda_i.ram import PLAY_MODE, read_snapshot

STATE_NAME = "L9Room05EntryReal"


class State:
    prev_live_count = -1
    last_progress_frame = 0
    backstep_frames = 0
    recentered_y = False


def policy_v5(snap, st: State, frame: int):
    block = next((o for o in snap.objects if o.type_id == 0x68 or o.slot == 11), None)
    live_wizz = [o for o in snap.objects if o.type_id in (0x23, 0x24) and o.hp > 0]

    if live_wizz:
        n_live = len(live_wizz)
        if st.prev_live_count < 0:
            st.prev_live_count = n_live
            st.last_progress_frame = frame
        elif n_live < st.prev_live_count:
            st.prev_live_count = n_live
            st.last_progress_frame = frame
            st.backstep_frames = 0

        nearest = min(live_wizz, key=lambda o: abs(o.x - snap.link_x) + abs(o.y - snap.link_y))
        dist = abs(nearest.x - snap.link_x) + abs(nearest.y - snap.link_y)
        stuck_close = dist < 16 and (frame - st.last_progress_frame) > 100
        if stuck_close or st.backstep_frames > 0:
            if st.backstep_frames <= 0:
                st.backstep_frames = 24
            st.backstep_frames -= 1
            if st.backstep_frames == 0:
                st.last_progress_frame = frame
            dx = nearest.x - snap.link_x
            dy = nearest.y - snap.link_y
            if abs(dx) >= abs(dy):
                d = "LEFT" if dx >= 0 else "RIGHT"
            else:
                d = "UP" if dy >= 0 else "DOWN"
            return FrameAction(nes_action(d), "v5_backstep")

        dx = nearest.x - snap.link_x
        dy = nearest.y - snap.link_y
        if abs(dx) > abs(dy):
            direction = "RIGHT" if dx > 0 else "LEFT"
        else:
            direction = "DOWN" if dy > 0 else "UP"
        if should_swing_at(snap.link_x, snap.link_y, direction, live_wizz):
            return FrameAction(nes_action(direction, "A"), "v5_engage_slash")
        return FrameAction(nes_action(direction), "v5_engage")

    if block is not None and block.y > STAIRS_05_PUSH_BLOCK_Y:
        if not st.recentered_y:
            if snap.link_y < 165:
                return FrameAction(nes_action("DOWN"), "v5_recenter_y")
            st.recentered_y = True
        if abs(snap.link_x - STAIRS_05_PUSH_X) > 4:
            d = "LEFT" if snap.link_x > STAIRS_05_PUSH_X else "RIGHT"
            return FrameAction(nes_action(d), "v5_align_x")
        return FrameAction(nes_action("UP"), "v5_push_up")

    if snap.link_x < STAIRS_05_STAIR_X:
        if snap.link_y < 173:
            return FrameAction(nes_action("DOWN"), "v5_walk_stair_south_aisle")
        return FrameAction(nes_action("RIGHT"), "v5_walk_stair_x")
    if snap.link_y > STAIRS_05_STAIR_Y:
        return FrameAction(nes_action("UP"), "v5_walk_stair_y")
    return FrameAction(nes_action("UP"), "v5_stand_on_stairs")


def main() -> int:
    configure_headless()
    assist = UnlimitedHealthAssist(enabled=True)
    env = make_env(GAME, STATE_NAME, GAME_DIR, render_mode="rgb_array")
    reset_obs(env)

    st = State()
    last_key = None
    max_frames = 16000
    for i in range(max_frames):
        snap = read_snapshot(env.get_ram())
        if snap.mode != PLAY_MODE:
            act = FrameAction(nes_idle_action(), f"wait_mode_{snap.mode}")
        elif snap.screen != STAIRS_05_ORIGIN:
            print(f"LEFT ROOM f{i}: screen=0x{snap.screen:02x} xy=({snap.link_x},{snap.link_y})")
            env.close()
            return 0
        else:
            act = policy_v5(snap, st, i)
        key = (act.reason, snap.link_x, snap.link_y)
        if key != last_key:
            live_wizz = [o for o in snap.objects if o.type_id in (0x23, 0x24) and o.hp > 0]
            block = next((o for o in snap.objects if o.type_id == 0x68 or o.slot == 11), None)
            print(
                f"f{i}: xy=({snap.link_x},{snap.link_y}) health={snap.health:#x} "
                f"n_wizz={len(live_wizz)} block_y={block.y if block else None} act={act.reason}"
            )
            last_key = key
        env.step(act.action)
        assist.apply_env(env, frame=i)
    else:
        snap = read_snapshot(env.get_ram())
        live_wizz = [o for o in snap.objects if o.type_id in (0x23, 0x24) and o.hp > 0]
        print(f"ran out of {max_frames} frames: screen=0x{snap.screen:02x} xy=({snap.link_x},{snap.link_y}) n_wizz={len(live_wizz)}")
    env.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
