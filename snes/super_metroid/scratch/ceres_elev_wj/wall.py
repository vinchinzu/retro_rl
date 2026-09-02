"""Scratch: is x=211 a wall pin or just spent momentum? Poke left, hold RIGHT."""

from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

from retro_harness.actions import buttons
from super_metroid.ram import ADDR_SAMUS_X, ADDR_SAMUS_X_SUB, parse_state, write_wram_u16

sys.path.insert(0, str(Path(__file__).resolve().parent))
from entry_state import ENTRY_STATE, open_entry  # noqa: E402


def main() -> None:
    env, session = open_entry()
    blob = ENTRY_STATE.read_bytes()
    try:
        env.em.set_state(blob)
        session.state = parse_state(env.get_ram(), frame=session.frame)
        for _ in range(8):
            session.step(buttons("RIGHT", "A"), "rise")
        write_wram_u16(env, ADDR_SAMUS_X, int(sys.argv[1]) if len(sys.argv) > 1 else 190)
        write_wram_u16(env, ADDR_SAMUS_X_SUB, 0)
        session.state = parse_state(env.get_ram(), frame=session.frame)
        for i in range(6):
            session.step(buttons("RIGHT", "A"), "push")
            ram = env.get_ram()
            x = int(ram[0x0AF6]) | (int(ram[0x0AF7]) << 8)
            xs = int(ram[0x0AF8]) | (int(ram[0x0AF9]) << 8)
            print(f"  f{i:2d} x={x}.{xs >> 8:03d} y="
                  f"{int(ram[0x0AFA]) | (int(ram[0x0AFB]) << 8)} "
                  f"pose={int(ram[0x0A1C])} $0AFE={int(ram[0x0AFE])} "
                  f"$0B00={int(ram[0x0B00])}", flush=True)
    finally:
        env.close()


if __name__ == "__main__":
    main()
