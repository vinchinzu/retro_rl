"""Scratch: dense WRAM dump of our own elevator wall jump, same schema as tas_wram.

Runs the shipped entry->475 recipe off the cached entry snapshot and records
every frame's low-WRAM slices plus the buttons we pressed, so the pose ladder,
the speed ladder and the hitbox width at contact can be read off directly.

    PYTHONPATH=snes uv run python -m super_metroid.scratch.ceres_elev_wj.our_wram
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

from retro_harness.actions import buttons, idle_action

sys.path.insert(0, str(Path(__file__).resolve().parent))
from entry_state import open_entry  # noqa: E402
from tas_wram import SLICES  # noqa: E402

HERE = Path(__file__).resolve().parent
OUT = HERE / "our_wram.json"

RISE = int(os.environ.get("RISE", 16))
RELEASE = int(os.environ.get("RELEASE", 2))
KICK = int(os.environ.get("KICK", 8))
RIDE = int(os.environ.get("RIDE", 34))
LAND = int(os.environ.get("LAND", 45))


def tape() -> list[tuple[str, ...]]:
    out: list[tuple[str, ...]] = []
    out += [("RIGHT", "A")] * RISE
    out += [("LEFT",)] * RELEASE
    out += [("LEFT", "A")] * KICK
    out += [("A",)] * RIDE
    out += [()] * LAND
    return out


def main() -> None:
    env, session = open_entry()
    rows: list[dict] = []
    try:
        ram = env.get_ram()
        rows.append(
            {"f": -1, "btn": [], "ram": {n: ram[lo:hi].tobytes().hex()
                                         for n, lo, hi in SLICES}}
        )
        for i, names in enumerate(tape()):
            session.step(buttons(*names) if names else idle_action(), "wj")
            ram = env.get_ram()
            rows.append(
                {
                    "f": i,
                    "btn": list(names),
                    "ram": {n: ram[lo:hi].tobytes().hex() for n, lo, hi in SLICES},
                }
            )
    finally:
        env.close()
    OUT.write_text(json.dumps({"tape": {"rise": RISE, "release": RELEASE,
                                        "kick": KICK, "ride": RIDE},
                               "rows": rows}) + "\n")
    print(f"rows={len(rows)} report: {OUT}")


if __name__ == "__main__":
    main()
