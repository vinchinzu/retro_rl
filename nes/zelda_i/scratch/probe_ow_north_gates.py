"""Test for a north exit out of 0x29 / 0x2A / 0x1C toward the enclosed
north-west block (0x09, 0x0A, 0x19, 0x1A) that holds the White Sword cave.

Established so far: 0x0A carries a unique ROM cave id (18), 0x0B has no west
exit (its walkable area is one uniform block and every band stays put), and
Lost Hills 0x1B wraps to itself in all four directions. That leaves 0x1A
(north of 0x2A) and 0x09/0x19 (north of 0x29) as the only ways in.

For each screen this dumps the walkable tilemap, then tries UP from every
column at several starting rows -- driving Link there by real movement and
skipping trials where he cannot actually reach the start.

    QT_QPA_PLATFORM=offscreen uv run python \
        nes/zelda_i/scratch/probe_ow_north_gates.py
"""
from __future__ import annotations

from retro_harness.env import make_env, read_state_bytes, reset_obs, save_state, state_path
from retro_harness.nes import nes_action
from retro_harness.segment_runner import configure_headless
from zelda_i.assist import UnlimitedHealthAssist
from zelda_i.level9.stair_run import dump_room_tiles
from zelda_i.paths import GAME, GAME_DIR
from zelda_i.ram import read_snapshot

PLAY_MODE = 5
SEEDS = ("BFS_2A", "OW_28", "BFS_2C", "BFS_1C", "BFS_1D")
COLS = tuple(range(32, 225, 16))
ROWS = (85, 109, 141, 173)


def walk_to(env, assist, tx, ty, limit=500):
    origin = read_snapshot(env.get_ram()).screen
    for i in range(limit):
        snap = read_snapshot(env.get_ram())
        if snap.screen != origin or snap.mode != PLAY_MODE:
            return False
        dx, dy = tx - int(snap.link_x), ty - int(snap.link_y)
        if abs(dx) <= 3 and abs(dy) <= 3:
            return True
        d = ("RIGHT" if dx > 0 else "LEFT") if abs(dx) > 3 else ("DOWN" if dy > 0 else "UP")
        env.step(nes_action(d))
        assist.apply_env(env, frame=i)
    return False


def try_up(env, assist):
    start = read_snapshot(env.get_ram())
    for i in range(500):
        snap = read_snapshot(env.get_ram())
        if snap.screen != start.screen or snap.mode not in (PLAY_MODE, 6, 7):
            break
        env.step(nes_action("UP"))
        assist.apply_env(env, frame=i)
    for i in range(180):
        snap = read_snapshot(env.get_ram())
        if snap.mode == PLAY_MODE:
            break
        env.step(nes_action("UP"))
        assist.apply_env(env, frame=i)
    return start, read_snapshot(env.get_ram())


def main() -> int:
    configure_headless()
    env = make_env(GAME, SEEDS[0], GAME_DIR, render_mode="rgb_array")
    reset_obs(env)
    assist = UnlimitedHealthAssist(enabled=True)

    for name in SEEDS:
        path = state_path(GAME_DIR, GAME, name)
        if not path.exists():
            print(f"\n== {name}: no pin ==", flush=True)
            continue
        env.em.set_state(read_state_bytes(path))
        env.step(env.action_space.sample() * 0)
        snap = read_snapshot(env.get_ram())
        if snap.level != 0 or snap.mode != PLAY_MODE:
            print(f"\n== {name}: not overworld play ==", flush=True)
            continue
        print(f"\n== {name}: screen 0x{snap.screen:02x} at "
              f"({snap.link_x},{snap.link_y}) ==", flush=True)
        pin = env.em.get_state()

        dump = dump_room_tiles(env, total=[0])
        ox, oy = dump["grid_origin"]
        print("  tiles " + " ".join(f"{ox + 8 * i:3d}" for i in range(len(dump["grid"][0]))), flush=True)
        for ri, row in enumerate(dump["grid"]):
            print(f"  y={oy + 8 * ri:3d} " + " ".join(f" {t:02x}" for t in row), flush=True)

        found = False
        for row in ROWS:
            for col in COLS:
                env.em.set_state(pin)
                if not walk_to(env, assist, col, row):
                    continue
                s, e = try_up(env, assist)
                if e.screen != s.screen and e.mode == PLAY_MODE:
                    print(f"  UP from ({col},{row}) -> 0x{e.screen:02x} "
                          f"at ({e.link_x},{e.link_y})", flush=True)
                    save_state(env, GAME_DIR, GAME, f"OW_{e.screen:02X}_north_of_{s.screen:02X}")
                    found = True
                    break
            if found:
                break
        if not found:
            print("  no north exit found from any reachable column", flush=True)
    env.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
