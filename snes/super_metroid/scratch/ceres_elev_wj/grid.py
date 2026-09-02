"""Scratch: rise x release grid — is the 2-frame A release a rule or an alias?

If a 1-frame release ever latches at some rise height, the requirement is
frame-phase aliasing. If it never does, it is a hard two-frame contract.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

from retro_harness.actions import buttons
from super_metroid.ram import parse_state

sys.path.insert(0, str(Path(__file__).resolve().parent))
from entry_state import ENTRY_STATE, open_entry, snap  # noqa: E402


def trial(env, session, blob: bytes, rise: int, release: int) -> bool:
    env.em.set_state(blob)
    session.state = parse_state(env.get_ram(), frame=session.frame)
    for _ in range(rise):
        session.step(buttons("RIGHT", "A"), "rise")
    for _ in range(release):
        session.step(buttons("LEFT"), "release")
    for _ in range(10):
        session.step(buttons("LEFT", "A"), "kick")
        if int(session.state.pose) == 132:
            return True
    return False


def main() -> None:
    env, session = open_entry()
    blob = ENTRY_STATE.read_bytes()
    try:
        rels = list(range(0, 5))
        print("rise\\rel " + " ".join(f"{r:>3d}" for r in rels))
        for rise in range(10, 25):
            marks = []
            for rel in rels:
                marks.append(" ## " if trial(env, session, blob, rise, rel) else "  . ")
            print(f"{rise:7d} " + "".join(marks), flush=True)
    finally:
        env.close()


if __name__ == "__main__":
    main()
