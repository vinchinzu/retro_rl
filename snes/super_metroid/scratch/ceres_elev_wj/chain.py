"""Scratch: chain a second wall jump out of the y=475 seat, using the contract.

The gate measured in ``gate3`` is ``$0A96 >= 0x0B`` on the frame A is newly
pressed, with Samus's leading edge within 8px of the wall. That says the kick
shape is: ride into the wall with A held, release A for two frames pressing
*away*, then away+A. This sweeps that shape against the x=155 chimney the TAS
latches (pose 131 at (155, 404)) on its way to the y=363 plant.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

from retro_harness.actions import buttons, idle_action
from super_metroid.ram import parse_state

sys.path.insert(0, str(Path(__file__).resolve().parent))
from entry_state import ENTRY_STATE, open_entry, snap  # noqa: E402

HERE = Path(__file__).resolve().parent
OUT = HERE / "chain.json"


def seat_475(env, session, blob: bytes) -> dict:
    env.em.set_state(blob)
    session.state = parse_state(env.get_ram(), frame=session.frame)
    for names, n in ((("RIGHT", "A"), 16), (("LEFT",), 2),
                     (("LEFT", "A"), 8), (("A",), 34)):
        for _ in range(n):
            session.step(buttons(*names), "seat")
    for _ in range(45):
        session.step(idle_action(), "seat")
        st = session.state
        if int(st.movement_type) in (0, 1) and int(st.velocity_y) == 0:
            break
    return snap(session.state)


def walk_to(session, target: int, limit: int = 60) -> None:
    for _ in range(limit):
        x = int(session.state.samus_x)
        if abs(x - target) <= 1:
            return
        session.step(buttons("RIGHT" if x < target else "LEFT"), "walk")


def attempt(env, session, blob, launch_x: int, into: str, ride: int) -> dict:
    seat_475(env, session, blob)
    walk_to(session, launch_x)
    away = "RIGHT" if into == "LEFT" else "LEFT"
    rows = []
    for _ in range(2):
        session.step(buttons(into), "turn")
    for _ in range(ride):
        session.step(buttons(into, "A"), "ride")
        rows.append(snap(session.state))
    for _ in range(2):
        session.step(buttons(away), "release")
        rows.append(snap(session.state))
    latch = None
    for _ in range(10):
        session.step(buttons(away, "A"), "kick")
        rows.append(snap(session.state))
        if int(session.state.pose) in (131, 132):
            latch = rows[-1]
            break
    for _ in range(50):
        session.step(buttons("A"), "carry")
        rows.append(snap(session.state))
    for _ in range(40):
        session.step(idle_action(), "land")
        rows.append(snap(session.state))
        st = session.state
        if int(st.movement_type) in (0, 1) and int(st.velocity_y) == 0:
            break
    return {
        "launch_x": launch_x, "into": into, "ride": ride,
        "latch": latch, "apex": min(r["y"] for r in rows),
        "seat": (rows[-1]["x"], rows[-1]["y"]),
    }


def main() -> None:
    env, session = open_entry()
    blob = ENTRY_STATE.read_bytes()
    out = []
    try:
        print("seat:", seat_475(env, session, blob), flush=True)
        for into in ("LEFT", "RIGHT"):
            for launch_x in (150, 156, 162, 168):
                for ride in range(4, 26, 2):
                    got = attempt(env, session, blob, launch_x, into, ride)
                    out.append(got)
                    if got["latch"] or got["seat"][1] <= 400:
                        print(
                            f"into={into:5s} x={launch_x:3d} ride={ride:2d} "
                            f"latch={got['latch']} apex={got['apex']} "
                            f"seat={got['seat']}",
                            flush=True,
                        )
        best = min(out, key=lambda r: r["seat"][1])
        print(f"\nbest seat: {best}")
        latched = [r for r in out if r["latch"]]
        print(f"latched variants: {len(latched)}/{len(out)}")
    finally:
        env.close()
    OUT.write_text(json.dumps(out, indent=1) + "\n")
    print(f"report: {OUT}")


if __name__ == "__main__":
    main()
