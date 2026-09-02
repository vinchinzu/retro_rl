"""Scratch: what does the second release frame change? Diff, then poke it back.

``$008F`` proves the game sees a fresh A press after a single release frame and
still refuses the wall jump, so the gate is state, not input. This diffs the
WRAM entering the kick frame for release=1 vs release=2, then tests whether
restoring the differing words alone buys the latch.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

import numpy as np

from retro_harness.actions import buttons
from super_metroid.ram import ADDR_VELOCITY_Y, ADDR_VELOCITY_Y_SUB, parse_state, write_wram_u16

sys.path.insert(0, str(Path(__file__).resolve().parent))
from entry_state import ENTRY_STATE, open_entry  # noqa: E402

RISE = 16
LO, HI = 0x0A00, 0x0B60


def ride(env, session, blob: bytes, release: int) -> np.ndarray:
    env.em.set_state(blob)
    session.state = parse_state(env.get_ram(), frame=session.frame)
    for _ in range(RISE):
        session.step(buttons("RIGHT", "A"), "rise")
    for _ in range(release):
        session.step(buttons("LEFT"), "release")
    return np.array(env.get_ram()[LO:HI])


def kick(env, session) -> bool:
    for _ in range(10):
        session.step(buttons("LEFT", "A"), "kick")
        if int(session.state.pose) == 132:
            return True
    return False


def main() -> None:
    env, session = open_entry()
    blob = ENTRY_STATE.read_bytes()
    try:
        one = ride(env, session, blob, 1)
        two = ride(env, session, blob, 2)
        diff = [(LO + i, int(one[i]), int(two[i]))
                for i in range(len(one)) if one[i] != two[i]]
        print(f"words differing entering the kick frame ({len(diff)} bytes):")
        for addr, a, b in diff:
            print(f"  ${addr:04X}: rel1={a:#04x} rel2={b:#04x}")

        print("\nrelease=1 + poked fall speed:")
        for vys in (0x1C00, 0x3800, 0x5400, 0x7000, 0xA000, 0xE000):
            ride(env, session, blob, 1)
            write_wram_u16(env, ADDR_VELOCITY_Y_SUB, vys)
            session.state = parse_state(env.get_ram(), frame=session.frame)
            print(f"  vy_sub={vys:#06x} latch={kick(env, session)}", flush=True)
        for vy in (1, 2, 3):
            ride(env, session, blob, 1)
            write_wram_u16(env, ADDR_VELOCITY_Y, vy)
            session.state = parse_state(env.get_ram(), frame=session.frame)
            print(f"  vy={vy} latch={kick(env, session)}", flush=True)
    finally:
        env.close()


if __name__ == "__main__":
    main()
