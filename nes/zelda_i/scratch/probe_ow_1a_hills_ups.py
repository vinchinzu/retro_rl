"""Test whether repeated UPs out of 0x1A break into the White Sword cave
screen 0x0A, the way repeated UPs out of Lost Hills 0x1B break into the
Level 5 door screen 0x0B.

The live BFS reported 0x1A's north as "sealed", but it only ever walked UP
once. 0x1A sits in the Lost Hills maze, where a single UP wraps you back to
the same screen and only the right *count* gets through -- the recorded
white-sword run reaches 0x0B with `hills_ups_3_to_door` from 0x1B for exactly
this reason. So "same screen after one UP" is not evidence of a seal here.

Walks 0x07 -> 0x17 -> 0x18 -> 0x19 -> 0x1A from the real power-on pin, then
holds UP up to 8 times, reporting the screen after each.

    QT_QPA_PLATFORM=offscreen uv run python \
        nes/zelda_i/scratch/probe_ow_1a_hills_ups.py
"""
from __future__ import annotations

from retro_harness.env import make_env, reset_obs, save_state
from retro_harness.nes import nes_action
from retro_harness.segment_runner import configure_headless
from zelda_i.assist import UnlimitedHealthAssist
from zelda_i.paths import GAME, GAME_DIR
from zelda_i.ram import read_snapshot

PLAY_MODE = 5
# From probe_ow_nw_bfs.py, all live: (direction, band, expected screen).
CHAIN = (("DOWN", 32, 0x17), ("RIGHT", 141, 0x18), ("RIGHT", 141, 0x19),
         ("RIGHT", 141, 0x1A))
MAX_UPS = 8
UP_COLUMNS = (48, 80, 112, 144, 176)


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

    for direction, band, want in CHAIN:
        if direction in ("LEFT", "RIGHT"):
            walk_to(env, assist, ty=band)
        else:
            walk_to(env, assist, tx=band)
        s, e = hold(env, assist, direction)
        print(f"0x{s.screen:02x} {direction} band={band} -> 0x{e.screen:02x} "
              f"at ({e.link_x},{e.link_y})", flush=True)
        if e.screen != want:
            print(f"chain broke (wanted 0x{want:02x}); stopping", flush=True)
            env.close()
            return 1
    save_state(env, GAME_DIR, GAME, "OW_1A_Real")
    print("pinned OW_1A_Real", flush=True)
    pin = env.em.get_state()

    for col in UP_COLUMNS:
        env.em.set_state(pin)
        print(f"\n-- repeated UP at x={col} --", flush=True)
        for n in range(1, MAX_UPS + 1):
            if not walk_to(env, assist, tx=col):
                print(f"  up#{n}: cannot reach x={col}", flush=True)
                break
            s, e = hold(env, assist, "UP")
            print(f"  up#{n}: 0x{s.screen:02x} -> 0x{e.screen:02x} "
                  f"at ({e.link_x},{e.link_y})", flush=True)
            if e.screen == 0x0A:
                save_state(env, GAME_DIR, GAME, "OW_0A_WhiteSwordReal")
                print(f"  *** 0x0A after {n} UPs at x={col} -- pinned "
                      f"OW_0A_WhiteSwordReal ***", flush=True)
                env.close()
                return 0
            if e.screen != s.screen:
                print(f"  left the maze to 0x{e.screen:02x}; abandoning this column",
                      flush=True)
                break
    print("\nno UP count at any column reaches 0x0A", flush=True)
    env.close()
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
