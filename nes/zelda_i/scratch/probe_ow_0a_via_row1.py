"""Reach the White Sword cave 0x0A via row 1, from the real power-on pin
OW_07_Row0Real.

Ruled out live so far: 0x0B has no west exit, Lost Hills 0x1B wraps to itself
in all four directions, 0x28/0x2A/0x1C/0x1D have no north exit, and row 0
runs 0x07 -> 0x08 -> 0x09 at band y=141 but 0x09's east is sealed. That
leaves 0x0A's south neighbour 0x1A, reachable from 0x09 by dropping to 0x19
and going east.

Tries the chain 0x09 DOWN -> 0x19 RIGHT -> 0x1A UP -> 0x0A, sweeping every
alignment band at each step.

    QT_QPA_PLATFORM=offscreen uv run python \
        nes/zelda_i/scratch/probe_ow_0a_via_row1.py
"""
from __future__ import annotations

from retro_harness.env import make_env, reset_obs, save_state
from retro_harness.nes import nes_action
from retro_harness.segment_runner import configure_headless
from zelda_i.assist import UnlimitedHealthAssist
from zelda_i.paths import GAME, GAME_DIR
from zelda_i.ram import read_snapshot

PLAY_MODE = 5
BANDS = (61, 77, 93, 109, 125, 141, 157, 173, 189)
COLS = tuple(range(32, 225, 16))
# (direction, expected target); None means "any new screen is interesting".
CHAIN = (("RIGHT", 0x08), ("RIGHT", 0x09), ("DOWN", 0x19), ("RIGHT", 0x1A), ("UP", 0x0A))


def walk_to(env, assist, *, tx=None, ty=None, limit=600):
    origin = read_snapshot(env.get_ram()).screen
    for i in range(limit):
        snap = read_snapshot(env.get_ram())
        if snap.screen != origin or snap.mode != PLAY_MODE:
            return False
        if tx is not None and abs(int(snap.link_x) - tx) > 3:
            d = "RIGHT" if int(snap.link_x) < tx else "LEFT"
        elif ty is not None and abs(int(snap.link_y) - ty) > 3:
            d = "DOWN" if int(snap.link_y) < ty else "UP"
        else:
            return True
        env.step(nes_action(d))
        assist.apply_env(env, frame=i)
    return False


def hold(env, assist, direction, limit=600):
    start = read_snapshot(env.get_ram())
    for i in range(limit):
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
    env = make_env(GAME, "OW_07_Row0Real", GAME_DIR, render_mode="rgb_array")
    reset_obs(env)
    assist = UnlimitedHealthAssist(enabled=True)
    snap = read_snapshot(env.get_ram())
    print(f"start 0x{snap.screen:02x} at ({snap.link_x},{snap.link_y}) "
          f"containers={snap.heart_containers} sword={snap.sword}", flush=True)

    state = env.em.get_state()
    table = []
    for direction, want in CHAIN:
        bands = BANDS if direction in ("LEFT", "RIGHT") else COLS
        hit = None
        for band in bands:
            env.em.set_state(state)
            ok = walk_to(env, assist, ty=band) if direction in ("LEFT", "RIGHT") \
                else walk_to(env, assist, tx=band)
            if not ok:
                continue
            s, e = hold(env, assist, direction)
            if e.screen == s.screen or e.mode != PLAY_MODE:
                continue
            print(f"0x{s.screen:02x} {direction:5s} band={band:3d} -> 0x{e.screen:02x} "
                  f"at ({e.link_x},{e.link_y})"
                  + ("" if e.screen == want else f"   (wanted 0x{want:02x})"), flush=True)
            if e.screen == want:
                hit = env.em.get_state()
                table.append((s.screen, direction, band, e.screen))
                break
        if hit is None:
            cur = read_snapshot(env.get_ram()).screen
            print(f"no band reaches 0x{want:02x} going {direction}; stopping", flush=True)
            break
        state = hit

    env.em.set_state(state)
    snap = read_snapshot(env.get_ram())
    print(f"\nended on 0x{snap.screen:02x} at ({snap.link_x},{snap.link_y})", flush=True)
    print("hop table:", [(hex(a), d, b, hex(c)) for a, d, b, c in table], flush=True)
    if snap.screen == 0x0A:
        save_state(env, GAME_DIR, GAME, "OW_0A_WhiteSwordReal")
        print("*** pinned OW_0A_WhiteSwordReal ***", flush=True)
    env.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
