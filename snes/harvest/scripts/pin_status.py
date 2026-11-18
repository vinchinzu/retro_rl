"""Mid-run status of Year 3 sweep runs, read from their newest morning pins (no emulator).

    cd snes/harvest && .venv/bin/python scripts/pin_status.py y3c4 y3k5          # tags of scripts/y23_sweep.sh runs
    .venv/bin/python scripts/pin_status.py --pin PR_y3c4_s1_Y3_Summer_D12        # one named pin

The ROM scores only at the ending, so this prints the RAM-model score (``evaluate_ranch_mastery_score``) with the
development term recomputed from the farm array: the ``ranch_development`` byte is stale until the final tally.
``tot~`` is the score with that corrected term. Shipped counters are ``tomato/corn/potato/turnip``.
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(HERE), str(HERE.parent), str(HERE.parents[1])]

from harvest.core.ram_catalog import read_ram_value  # noqa: E402
from harvest.core.ranch_master import count_development_tiles, evaluate_ranch_mastery_score  # noqa: E402
from harvest.core.world_snapshot import load_state_ram  # noqa: E402

PINS = HERE / "custom_integrations" / "HarvestMoon-Snes"


def newest_pin(tag: str) -> str | None:
    best = None
    for path in PINS.glob(f"PR_{tag}_s[0-9]_Y3_*_D*.state"):
        match = re.search(r"Y3_(Spring|Summer)_D(\d+)\.state$", path.name)
        if match:
            key = (match.group(1) == "Summer", int(match.group(2)))
            if best is None or key > best[0]:
                best = (key, path.stem)
    return best[1] if best else None


def describe(pin: str) -> str:
    ram = load_state_ram(pin)
    score = evaluate_ranch_mastery_score(ram).to_dict()
    value = lambda key: int(read_ram_value(ram, key))  # noqa: E731
    tiles = count_development_tiles(ram)
    dev = min(100, tiles * 10 // 295) >> 1
    corrected = score["total_score"] - score["ranch_development_score"] + dev
    return (
        f"{pin.split('_Y3_')[-1]:12s} score={score['total_score']:3d} tot~{corrected:3d} money={value('money'):6d} "
        f"cows={value('num_cows')} hens={value('num_chickens')} berries={value('power_berry_count')} hay={value('hay')} "
        f"dev={tiles}t/{dev}p ship={value('shipped_tomatoes')}/{value('shipped_corn')}/{value('shipped_potatoes')}/"
        f"{value('shipped_turnips')} eve={value('eve_hearts')} happy={value('happiness')} "
        f"cow_pts={score['cow_happiness_details'][:8]}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("tags", nargs="*", help="y23_sweep tags (the part between PR_ and _s1_)")
    parser.add_argument("--pin", action="append", default=[], help="a pin name to read directly")
    args = parser.parse_args()
    for tag in args.tags:
        pin = newest_pin(tag)
        print(f"{tag:8s} " + (describe(pin) if pin else "no pin yet"))
    for pin in args.pin:
        print(describe(pin))


if __name__ == "__main__":
    main()
