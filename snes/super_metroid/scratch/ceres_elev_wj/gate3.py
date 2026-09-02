"""Scratch: the $0A96 threshold — the actual wall-jump contract, in one number."""

from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

from retro_harness.actions import buttons
from super_metroid.ram import parse_state, write_wram_u16

sys.path.insert(0, str(Path(__file__).resolve().parent))
from entry_state import ENTRY_STATE, open_entry  # noqa: E402

RISE = 16
A96 = 0x0A96


def ride(env, session, blob: bytes, release: int) -> None:
    env.em.set_state(blob)
    session.state = parse_state(env.get_ram(), frame=session.frame)
    for _ in range(RISE):
        session.step(buttons("RIGHT", "A"), "rise")
    for _ in range(release):
        session.step(buttons("LEFT"), "release")


def kick(env, session) -> tuple[bool, int]:
    for i in range(10):
        session.step(buttons("LEFT", "A"), "kick")
        if int(session.state.pose) == 132:
            return True, i
    return False, -1


def main() -> None:
    env, session = open_entry()
    blob = ENTRY_STATE.read_bytes()
    try:
        for release in (0, 1, 2, 3):
            ride(env, session, blob, release)
            ram = env.get_ram()
            print(f"release={release} $0A96="
                  f"{int(ram[A96]) | (int(ram[A96 + 1]) << 8):#06x} "
                  f"$0A94={int(ram[0x0A94]):#04x} "
                  f"pose={int(ram[0x0A1C]):d}", flush=True)
        print("\npoke $0A96 on the release=1 state:")
        for v in range(0, 0x20):
            ride(env, session, blob, 1)
            write_wram_u16(env, A96, v)
            session.state = parse_state(env.get_ram(), frame=session.frame)
            got, at = kick(env, session)
            print(f"  $0A96={v:#04x} latch={'YES' if got else ' no'} "
                  f"{'at kick f' + str(at) if got else ''}", flush=True)
    finally:
        env.close()


if __name__ == "__main__":
    main()
