"""Scratch: what x still latches the wall jump? The contact-from-width contract.

Rides the right wall, pokes ``$0AF6`` to a candidate x on the release frame,
then runs the standard release+kick and reports whether pose 132 fired. The
widest latching x, against Samus's X radius ($0AFE), is the contact rule.

    PYTHONPATH=snes uv run python -m super_metroid.scratch.ceres_elev_wj.contact
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

from retro_harness.actions import buttons, idle_action
from super_metroid.ram import (
    ADDR_SAMUS_X,
    ADDR_SAMUS_X_SUB,
    parse_state,
    write_wram_u16,
)

sys.path.insert(0, str(Path(__file__).resolve().parent))
from entry_state import ENTRY_STATE, open_entry, snap  # noqa: E402

HERE = Path(__file__).resolve().parent
OUT = HERE / "contact.json"

RISE, RELEASE, KICK = 16, 2, 10


def trial(env, session, blob: bytes, x: int, xsub: int) -> dict:
    env.em.set_state(blob)
    session.state = parse_state(env.get_ram(), frame=session.frame)
    for _ in range(RISE):
        session.step(buttons("RIGHT", "A"), "rise")
    write_wram_u16(env, ADDR_SAMUS_X, x)
    write_wram_u16(env, ADDR_SAMUS_X_SUB, xsub)
    session.state = parse_state(env.get_ram(), frame=session.frame)
    rows = []
    for _ in range(RELEASE):
        session.step(buttons("LEFT"), "release")
        rows.append(snap(session.state))
    for _ in range(KICK):
        session.step(buttons("LEFT", "A"), "kick")
        rows.append(snap(session.state))
    latched = any(r["pose"] == 132 for r in rows)
    return {
        "x": x,
        "xsub": xsub,
        "held_x": rows[0]["x"],
        "latched": latched,
        "apex": min(r["y"] for r in rows),
    }


def main() -> None:
    env, session = open_entry()
    blob = ENTRY_STATE.read_bytes()
    out = []
    try:
        for x in range(196, 216):
            got = trial(env, session, blob, x, 0)
            out.append(got)
            print(
                f"x={x:3d} held={got['held_x']:3d} "
                f"{'LATCH' if got['latched'] else '  .  '} apex={got['apex']}",
                flush=True,
            )
        print("--- subpixel sweep at the last non-latching x ---", flush=True)
        edge = next((r["x"] for r in out if r["latched"]), None)
        if edge is not None and edge > 196:
            for sub in (0, 64, 128, 192, 255):
                got = trial(env, session, blob, edge - 1, sub)
                out.append(got)
                print(
                    f"x={edge - 1:3d}.{sub:3d} "
                    f"{'LATCH' if got['latched'] else '  .  '}",
                    flush=True,
                )
    finally:
        env.close()
    OUT.write_text(json.dumps(out, indent=1) + "\n")
    print(f"report: {OUT}")


if __name__ == "__main__":
    main()
