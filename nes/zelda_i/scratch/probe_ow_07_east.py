"""Reach the White Sword cave 0x0A by walking east along row 0 from the real
power-on pin OW_07_Row0Real --
the screen the Level 9 approach already passes through.

Everything else is now ruled out live: 0x0B has no west exit, Lost Hills 0x1B
wraps to itself, and 0x28/0x2A/0x1C/0x1D have no north exit into the enclosed
north-west block. But LEVEL9_ROCK_HOPS itself walks 0x27 -LEFT-> 0x17 -UP->
0x07 -LEFT-> 0x06 -LEFT-> 0x05, so row 0 is reachable, and 0x0A is three
screens EAST of 0x07 along that same row.

Follows the route's own hops to 0x07, then sweeps RIGHT screen by screen.

    QT_QPA_PLATFORM=offscreen uv run python \
        nes/zelda_i/scratch/probe_ow_07_east.py
"""
from __future__ import annotations

from retro_harness.env import make_env, reset_obs, save_state
from retro_harness.nes import nes_action
from retro_harness.segment_runner import configure_headless
from zelda_i.assist import UnlimitedHealthAssist
from zelda_i.overworld.graph import ScreenHop
from zelda_i.overworld.path import OverworldPathController
from zelda_i.paths import GAME, GAME_DIR
from zelda_i.ram import read_snapshot

PLAY_MODE = 5
TARGET = 0x0A
BANDS = (61, 77, 93, 109, 125, 141, 157, 173, 189)


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
    print(f"start 0x{snap.screen:02x} at ({snap.link_x},{snap.link_y})", flush=True)

    state = env.em.get_state()
    for hop in range(6):
        cur = read_snapshot(env.get_ram()).screen
        if cur == TARGET:
            break
        advanced = False
        for band in BANDS:
            env.em.set_state(state)
            if not walk_to(env, assist, ty=band):
                continue
            s, e = hold(env, assist, "RIGHT")
            if e.screen != s.screen and e.mode == PLAY_MODE:
                print(f"0x{s.screen:02x} RIGHT band={band} -> 0x{e.screen:02x} "
                      f"at ({e.link_x},{e.link_y})", flush=True)
                state = env.em.get_state()
                advanced = True
                break
        if not advanced:
            print(f"0x{cur:02x} RIGHT sealed at every band", flush=True)
            break

    env.em.set_state(state)
    snap = read_snapshot(env.get_ram())
    print(f"\nended on 0x{snap.screen:02x} at ({snap.link_x},{snap.link_y})", flush=True)
    if snap.screen == TARGET:
        save_state(env, GAME_DIR, GAME, "OW_0A_WhiteSword")
        print("*** pinned OW_0A_WhiteSword ***", flush=True)
    env.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
