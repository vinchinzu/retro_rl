"""From the STAIRS_03 stall pin, find what actually unlocks room 0x03's 0x68.

Link presses the block from the south for 240 frames with nothing happening,
yet the uncapped join pushed it during combat. Two candidates:
  A. the block is gated on the room being cleared (room_all_dead),
  B. the push face is wrong (memory: push a 0x68 from its OPEN side).

Tests both from the same pin.

    QT_QPA_PLATFORM=offscreen uv run python \
        nes/zelda_i/scratch/probe_l9_03_block_gate.py
"""
from __future__ import annotations

from retro_harness.env import make_env, reset_obs
from retro_harness.nes import nes_action
from retro_harness.segment_runner import configure_headless
from zelda_i.assist import UnlimitedHealthAssist
from zelda_i.level9.stairs import chase_sword_step, live_combat_objects, pushable_block
from zelda_i.paths import GAME, GAME_DIR
from zelda_i.ram import read_snapshot

STATE_NAME = "L9Stairs03StallReal"


def report(env, tag):
    snap = read_snapshot(env.get_ram())
    blk = pushable_block(snap)
    print(f"  {tag}: link=({snap.link_x},{snap.link_y}) "
          f"block={None if blk is None else (blk.x, blk.y)} "
          f"all_dead={snap.room_all_dead} "
          f"live={[hex(o.type_id) for o in live_combat_objects(snap) if o.type_id != 0x2B]}",
          flush=True)
    return snap


def walk_to(env, assist, tx, ty, limit=600):
    for i in range(limit):
        snap = read_snapshot(env.get_ram())
        dx, dy = tx - int(snap.link_x), ty - int(snap.link_y)
        if abs(dx) <= 1 and abs(dy) <= 1:
            return True
        d = ("RIGHT" if dx > 0 else "LEFT") if abs(dx) > 1 else ("DOWN" if dy > 0 else "UP")
        env.step(nes_action(d))
        assist.apply_env(env, frame=i)
    return False


def hold(env, assist, d, n):
    for i in range(n):
        env.step(nes_action(d))
        assist.apply_env(env, frame=i)


def main() -> int:
    configure_headless()
    env = make_env(GAME, STATE_NAME, GAME_DIR, render_mode="rgb_array")
    reset_obs(env)
    assist = UnlimitedHealthAssist(enabled=True)
    pin = env.em.get_state()
    report(env, "stall")

    # B. wrong-face test: approach from the north and push DOWN.
    print("B. push from the north (hold DOWN):", flush=True)
    ok = walk_to(env, assist, 96, 120)
    report(env, f"north stand reached={ok}")
    hold(env, assist, "DOWN", 240)
    report(env, "after 240f DOWN")

    # A. clear-gate test: fight until room_all_dead, then push from the south.
    env.em.set_state(pin)
    print("A. clear the room, then push UP from the south:", flush=True)
    cooldown = 0
    for i in range(20000):
        snap = read_snapshot(env.get_ram())
        if snap.room_all_dead:
            break
        act, cooldown = chase_sword_step(snap, cooldown, types=(0x13, 0x14, 0x17))
        env.step(act.action)
        assist.apply_env(env, frame=i)
    snap = report(env, f"after combat ({i}f)")
    ok = walk_to(env, assist, 96, 170)
    report(env, f"south stand reached={ok}")
    hold(env, assist, "UP", 240)
    report(env, "after 240f UP")
    env.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
