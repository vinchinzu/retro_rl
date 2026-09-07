"""Controlled test of the room 0x10 `0x68` object: is it a push block or a
self-moving object? The previous 120-frame blind-hold test could not tell
those apart (rr-sz8.6, 2026-09-06).

Phase A: Link idles far away for 240 frames -- if the 0x68 moves with zero
input, it is not a static push block.
Phase B: walk adjacent to its live position and hold one direction in short
30-frame bursts, logging the block xy every frame it changes.

    QT_QPA_PLATFORM=offscreen uv run python \
        nes/zelda_i/scratch/probe_l9_10_block_push.py
"""
from __future__ import annotations

from retro_harness.env import make_env, reset_obs
from retro_harness.nes import nes_action, nes_idle_action
from retro_harness.segment_runner import configure_headless
from zelda_i.assist import UnlimitedHealthAssist
from zelda_i.paths import GAME, GAME_DIR
from zelda_i.ram import read_snapshot

STATE_NAME = "L9Room10EntryReal"
BLOCK_TYPE = 0x68


def block_of(snap):
    for o in snap.objects:
        if o.type_id == BLOCK_TYPE:
            return o
    return None


def main() -> int:
    configure_headless()
    env = make_env(GAME, STATE_NAME, GAME_DIR, render_mode="rgb_array")
    obs, _ = reset_obs(env)
    assist = UnlimitedHealthAssist(enabled=True)
    frame = 0

    def step(action):
        nonlocal frame, obs
        obs, *_ = env.step(action)
        frame += 1
        assist.apply_env(env, frame=frame)
        return read_snapshot(env.get_ram())

    snap = read_snapshot(env.get_ram())
    b = block_of(snap)
    print(f"entry link=({snap.link_x},{snap.link_y}) block={b and (b.x, b.y)} "
          f"item={snap.room_item_id} dead={snap.room_all_dead}")

    # Phase A: pure idle, no input at all.
    start = b and (b.x, b.y)
    moved = []
    for _ in range(240):
        snap = step(nes_idle_action())
        b = block_of(snap)
        if b and (b.x, b.y) != start:
            moved.append((frame, b.x, b.y))
            start = (b.x, b.y)
    print(f"PHASE A idle 240f: block moves with zero input = {len(moved)}")
    for m in moved[:10]:
        print("   ", m)

    b = block_of(snap)
    if b is None:
        print("no 0x68 object present")
        env.close()
        return 1
    print(f"after idle: link=({snap.link_x},{snap.link_y}) block=({b.x},{b.y})")

    # Phase B: for each direction, park Link on the opposite face and push.
    faces = {
        "LEFT": (16, 0),    # stand east of block, push west
        "RIGHT": (-16, 0),  # stand west of block, push east
        "UP": (0, 16),      # stand south of block, push north
        "DOWN": (0, -16),   # stand north of block, push south
    }
    for direction, (ox, oy) in faces.items():
        b = block_of(snap)
        if b is None:
            break
        tx, ty = b.x + ox, b.y + oy
        # crude walk: x first then y, 300 frame cap
        for _ in range(300):
            if abs(snap.link_x - tx) <= 3 and abs(snap.link_y - ty) <= 3:
                break
            if abs(snap.link_x - tx) > 3:
                d = "RIGHT" if snap.link_x < tx else "LEFT"
            else:
                d = "DOWN" if snap.link_y < ty else "UP"
            snap = step(nes_action(d))
        b0 = block_of(snap)
        pos0 = b0 and (b0.x, b0.y)
        for _ in range(40):
            snap = step(nes_action(direction))
        b1 = block_of(snap)
        pos1 = b1 and (b1.x, b1.y)
        print(f"PUSH {direction}: link=({snap.link_x},{snap.link_y}) target=({tx},{ty}) "
              f"block {pos0} -> {pos1} moved={pos0 != pos1} item={snap.room_item_id} "
              f"objs={[(hex(o.type_id), o.x, o.y) for o in snap.objects if o.type_id]}")

    env.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
