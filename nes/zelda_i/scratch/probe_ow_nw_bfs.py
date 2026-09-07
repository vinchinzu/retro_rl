"""Live BFS of the north-west overworld to find a real walking route to the
White Sword cave screen 0x0A.

ROM decode (`dump_ow_rom_screens.py`) confirms 0x0A carries a unique cave id
(18), the sibling of the wooden sword cave 0x77 (id 16) -- so the screen is
right and only the approach is missing. `route/item_gate_hops.py` guessed
"west off 0x0B", which is sealed live (0x0B's north half is mountain), and
west off Lost Hills 0x1B is sealed too. This searches instead of guessing:
from each reachable screen, try all four directions at every alignment band
and record the edges that actually move Link.

    QT_QPA_PLATFORM=offscreen uv run python \
        nes/zelda_i/scratch/probe_ow_nw_bfs.py
"""
from __future__ import annotations

from collections import deque

from retro_harness.env import make_env, read_state_bytes, reset_obs, save_state, state_path
from retro_harness.nes import nes_action
from retro_harness.segment_runner import configure_headless
from zelda_i.assist import UnlimitedHealthAssist
from zelda_i.paths import GAME, GAME_DIR
from zelda_i.ram import read_snapshot

PLAY_MODE = 5
TARGET = 0x0A
BANDS = (61, 77, 93, 109, 125, 141, 157, 173, 189)
COLS = (32, 64, 96, 128, 160, 192, 216)
DIRS = ("LEFT", "UP", "RIGHT", "DOWN")
MAX_ROW = 2  # keep the search in the northern overworld
# OW_07_Row0Real is pinned from a real power-on run (pin_ow_row0.py) and is
# the only seed actually inside the northern block; the rest bound the search.
SEEDS = ("OW_07_Row0Real", "OW_0B_L5Door", "BFS_0C", "BFS_1C", "BFS_1D")


def try_edge(env, assist, direction, band):
    """Align on the cross axis then hold `direction`. Returns (start, end)."""
    horizontal = direction in ("LEFT", "RIGHT")
    origin = read_snapshot(env.get_ram())
    for i in range(300):
        snap = read_snapshot(env.get_ram())
        if snap.screen != origin.screen or snap.mode != PLAY_MODE:
            return None, snap  # alignment wandered off; trial is meaningless
        cur = int(snap.link_y) if horizontal else int(snap.link_x)
        if abs(cur - band) <= 2:
            break
        d = ("DOWN" if cur < band else "UP") if horizontal else ("RIGHT" if cur < band else "LEFT")
        env.step(nes_action(d))
        assist.apply_env(env, frame=i)
    start = read_snapshot(env.get_ram())
    for i in range(600):
        snap = read_snapshot(env.get_ram())
        if snap.screen != start.screen or snap.mode not in (PLAY_MODE, 6, 7):
            break
        env.step(nes_action(direction))
        assist.apply_env(env, frame=i)
    for i in range(180):
        snap = read_snapshot(env.get_ram())
        if snap.mode == PLAY_MODE:
            break
        env.step(nes_action(direction))
        assist.apply_env(env, frame=i)
    return start, read_snapshot(env.get_ram())


def main() -> int:
    configure_headless()
    assist = UnlimitedHealthAssist(enabled=True)
    seen: dict[int, bytes] = {}
    parent: dict[int, tuple[int, str, int]] = {}
    queue: deque[int] = deque()

    # Lost Hills 0x1B is deliberately NOT a seed: it wraps to itself in all
    # four directions at every band (the maze), so BFS from it sees no edges
    # at all. Seed from the screens around it instead.
    # stable-retro allows only one emulator per process, so seeds are loaded
    # as raw state bytes into a single env rather than one env per pin.
    env = make_env(GAME, SEEDS[0], GAME_DIR, render_mode="rgb_array")
    reset_obs(env)
    for name in SEEDS:
        path = state_path(GAME_DIR, GAME, name)
        if not path.exists():
            print(f"seed {name}: no such pin", flush=True)
            continue
        env.em.set_state(read_state_bytes(path))
        env.step(env.action_space.sample() * 0)
        snap = read_snapshot(env.get_ram())
        if snap.mode != PLAY_MODE or snap.level != 0:
            print(f"seed {name}: not overworld play (level={snap.level} "
                  f"mode={snap.mode}); skipped", flush=True)
            continue
        print(f"seed {name}: 0x{snap.screen:02x} at ({snap.link_x},{snap.link_y})", flush=True)
        if snap.screen not in seen:
            seen[snap.screen] = env.em.get_state()
            queue.append(snap.screen)

    while queue:
        screen = queue.popleft()
        state = seen[screen]
        for direction in DIRS:
            bands = BANDS if direction in ("LEFT", "RIGHT") else COLS
            for band in bands:
                env.em.set_state(state)
                s, e = try_edge(env, assist, direction, band)
                if s is None or e.mode != PLAY_MODE:
                    continue
                if e.screen == s.screen or e.screen in seen:
                    continue
                if (e.screen >> 4) > MAX_ROW:
                    continue
                seen[e.screen] = env.em.get_state()
                parent[e.screen] = (screen, direction, band)
                print(f"  0x{screen:02x} {direction:5s} band={band:3d} -> 0x{e.screen:02x} "
                      f"at ({e.link_x},{e.link_y})", flush=True)
                if e.screen == TARGET:
                    save_state(env, GAME_DIR, GAME, "OW_0A_WhiteSword")
                    print("\n*** reached 0x0A -- pinned OW_0A_WhiteSword ***", flush=True)
                    path = []
                    cur = TARGET
                    while cur in parent:
                        src, d, b = parent[cur]
                        path.append((src, d, b, cur))
                        cur = src
                    for src, d, b, dst in reversed(path):
                        print(f"    0x{src:02x} --{d} band={b}--> 0x{dst:02x}", flush=True)
                    env.close()
                    return 0
                queue.append(e.screen)
                break  # this direction is settled; keep trying the others

    print("\n0x0A not reached; screens seen:",
          " ".join(f"0x{s:02x}" for s in sorted(seen)), flush=True)
    env.close()
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
