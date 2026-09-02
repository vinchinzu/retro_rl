"""Scratch: which of the 14 differing bytes is the wall-jump gate?

Rides to the release=1 state, pokes one candidate up to its release=2 value,
and asks for the latch. Whatever flips the answer alone is the contract.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

import numpy as np

from retro_harness.actions import buttons
from super_metroid.ram import parse_state, write_wram_u8

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
        diff = [LO + i for i in range(len(one)) if one[i] != two[i]]
        hits = []
        for addr in diff:
            ride(env, session, blob, 1)
            write_wram_u8(env, addr, int(two[addr - LO]))
            session.state = parse_state(env.get_ram(), frame=session.frame)
            got = kick(env, session)
            hits.append((addr, got))
            print(f"  poke ${addr:04X} -> {int(two[addr - LO]):#04x}  "
                  f"latch={got}", flush=True)
        if not any(g for _, g in hits):
            print("\nno single byte flips it; poking the whole diff:")
            ride(env, session, blob, 1)
            for addr in diff:
                write_wram_u8(env, addr, int(two[addr - LO]))
            session.state = parse_state(env.get_ram(), frame=session.frame)
            print(f"  all 14 -> latch={kick(env, session)}", flush=True)
            for group in ([0x0A28, 0x0A29], [0x0A94], [0x0A96], [0x0A10],
                          [0x0A94, 0x0A96], [0x0A28, 0x0A29, 0x0A94, 0x0A96]):
                ride(env, session, blob, 1)
                for addr in group:
                    write_wram_u8(env, addr, int(two[addr - LO]))
                session.state = parse_state(env.get_ram(), frame=session.frame)
                names = ",".join(f"${a:04X}" for a in group)
                print(f"  {names} -> latch={kick(env, session)}", flush=True)
    finally:
        env.close()


if __name__ == "__main__":
    main()
