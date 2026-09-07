"""Find a live overworld route to the White Sword cave screen 0x0A.

`route/item_gate_hops.py` marks 0x0A residual ("no OW west off 0x0B live"),
so the last leg of the white-sword route has never been walked. This sweeps
each candidate approach: from the L5 door 0x0B westward, and from the Level 9
approach screens 0x07/0x06 eastward, at several alignment bands, and reports
which screen Link actually lands on.

    QT_QPA_PLATFORM=offscreen uv run python \
        nes/zelda_i/scratch/probe_ow_0a_sweep.py
"""
from __future__ import annotations

import sys

from retro_harness.env import make_env, reset_obs
from retro_harness.nes import nes_action
from retro_harness.segment_runner import configure_headless
from zelda_i.assist import UnlimitedHealthAssist
from zelda_i.paths import GAME, GAME_DIR
from zelda_i.ram import read_snapshot

PLAY_MODE = 5
BANDS = (77, 93, 109, 125, 141, 157, 173)
COLS = (48, 80, 112, 144, 176, 208)


def align_then_walk(env, assist, direction, *, band, limit=900):
    """Align on the cross axis to `band`, then hold `direction`.

    Aborts if the alignment walk itself leaves the overworld -- on 0x0B the
    north column is the Level 5 door, so aligning x there enters the dungeon
    and every reading after that is meaningless.
    """
    horizontal = direction in ("LEFT", "RIGHT")
    origin = read_snapshot(env.get_ram()).screen
    for i in range(240):
        snap = read_snapshot(env.get_ram())
        if snap.screen != origin or snap.mode != PLAY_MODE:
            return None, snap
        cur = int(snap.link_y) if horizontal else int(snap.link_x)
        if abs(cur - band) <= 2:
            break
        if horizontal:
            d = "DOWN" if cur < band else "UP"
        else:
            d = "RIGHT" if cur < band else "LEFT"
        env.step(nes_action(d))
        assist.apply_env(env, frame=i)
    start = read_snapshot(env.get_ram())
    for i in range(limit):
        snap = read_snapshot(env.get_ram())
        if snap.screen != start.screen or snap.mode not in (PLAY_MODE, 6, 7):
            break
        env.step(nes_action(direction))
        assist.apply_env(env, frame=i)
    for i in range(120):  # settle
        snap = read_snapshot(env.get_ram())
        if snap.mode == PLAY_MODE:
            break
        env.step(nes_action(direction))
        assist.apply_env(env, frame=i)
    snap = read_snapshot(env.get_ram())
    return start, snap


def sweep(env, assist, pin, direction, bands, *, label):
    """Try `direction` at every band from `pin`; return {screen: state_bytes}."""
    env.em.set_state(pin)
    snap = read_snapshot(env.get_ram())
    print(f"\n== {label}: screen=0x{snap.screen:02x} at ({snap.link_x},{snap.link_y}) "
          f"walk {direction} ==", flush=True)
    found: dict[int, bytes] = {}
    for band in bands:
        env.em.set_state(pin)
        start, end = align_then_walk(env, assist, direction, band=band)
        if start is None:
            print(f"  band={band:3d} ABORT: alignment left the screen "
                  f"(0x{end.screen:02x} mode={end.mode})", flush=True)
            continue
        tag = "SAME" if end.screen == start.screen else f"-> 0x{end.screen:02x}"
        print(f"  band={band:3d} from 0x{start.screen:02x} {tag} "
              f"mode={end.mode} xy=({end.link_x},{end.link_y})", flush=True)
        if end.screen != start.screen and end.mode == PLAY_MODE:
            found.setdefault(end.screen, env.em.get_state())
    return found


def main() -> int:
    configure_headless()
    env = make_env(GAME, "OW_1B_LostHills", GAME_DIR, render_mode="rgb_array")
    reset_obs(env)
    assist = UnlimitedHealthAssist(enabled=True)

    lost_hills = env.em.get_state()
    west = sweep(env, assist, lost_hills, "LEFT", BANDS, label="Lost Hills 0x1B")
    for screen, state in sorted(west.items()):
        up = sweep(env, assist, state, "UP", COLS, label=f"0x{screen:02x} (west of 0x1B)")
        for s2, st2 in sorted(up.items()):
            if s2 == 0x0A:
                print(f"\n*** reached 0x0A from 0x{screen:02x} going UP ***", flush=True)
                sweep(env, assist, st2, "LEFT", BANDS, label="0x0A")
    env.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
