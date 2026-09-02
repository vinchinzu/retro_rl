"""Scratch: wall faces and Samus's radii, so "contact from width" is measured.

Seats the climb on the y=475 ledge, then walks into each wall and reports the
resting x with $0AFE / $0B00. Resting x plus the radius gives the wall's first
solid column, which is what the wall-jump contact probe is measured against.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

from retro_harness.actions import buttons, idle_action
from super_metroid.ram import parse_state

sys.path.insert(0, str(Path(__file__).resolve().parent))
from entry_state import ENTRY_STATE, open_entry  # noqa: E402


def read(env) -> dict:
    ram = env.get_ram()
    u16 = lambda a: int(ram[a]) | (int(ram[a + 1]) << 8)  # noqa: E731
    return {
        "x": u16(0x0AF6), "xs": u16(0x0AF8), "y": u16(0x0AFA),
        "xr": u16(0x0AFE), "yr": u16(0x0B00),
        "pose": u16(0x0A1C), "mt": int(ram[0x0A1F]),
    }


def seat_475(env, session, blob: bytes) -> None:
    env.em.set_state(blob)
    session.state = parse_state(env.get_ram(), frame=session.frame)
    for names, n in ((("RIGHT", "A"), 16), (("LEFT",), 2),
                     (("LEFT", "A"), 8), (("A",), 34)):
        for _ in range(n):
            session.step(buttons(*names), "seat")
    for _ in range(45):
        session.step(idle_action(), "seat")


def walk(env, session, dirn: str, n: int = 120) -> dict:
    last = None
    for _ in range(n):
        session.step(buttons(dirn), "walk")
        cur = read(env)
        if last and cur["x"] == last["x"] and cur["xs"] == last["xs"]:
            return cur
        last = cur
    return last


def main() -> None:
    env, session = open_entry()
    blob = ENTRY_STATE.read_bytes()
    try:
        print("air pin against the right wall (spin, during the rise):")
        env.em.set_state(blob)
        session.state = parse_state(env.get_ram(), frame=session.frame)
        for _ in range(16):
            session.step(buttons("RIGHT", "A"), "rise")
        r = read(env)
        print(f"  x={r['x']}.{r['xs'] >> 8:03d} y={r['y']} pose={r['pose']} "
              f"$0AFE={r['xr']} $0B00={r['yr']}  "
              f"x+$0AFE+1={r['x'] + r['xr'] + 1}  x+$0B00+1={r['x'] + r['yr'] + 1}")

        seat_475(env, session, blob)
        print(f"seated: {read(env)}")
        for dirn in ("RIGHT", "LEFT"):
            seat_475(env, session, blob)
            r = walk(env, session, dirn)
            print(f"walk {dirn:5s}: x={r['x']}.{r['xs'] >> 8:03d} y={r['y']} "
                  f"pose={r['pose']} mt={r['mt']} $0AFE={r['xr']} $0B00={r['yr']}"
                  f"  edge={r['x'] + r['xr'] + 1 if dirn == 'RIGHT' else r['x'] - r['xr'] - 1}")
    finally:
        env.close()


if __name__ == "__main__":
    main()
