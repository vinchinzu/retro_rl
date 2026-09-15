"""Print the Clean campaign ladder: tip, next open hop, blockers by class.

    uv run python nes/zelda_i/scripts/clean_tip.py
    uv run python nes/zelda_i/scripts/clean_tip.py --json
    uv run python nes/zelda_i/scripts/clean_tip.py --adoption

Reads ``spine/clean_tip.py`` only. No emulator, no ROM, no STATUS claim.
"""

from __future__ import annotations

import argparse
import json

from zelda_i.spine.clean_tip import (
    CLEAN_LADDER,
    adoption,
    by_blocker,
    next_open,
    render,
    render_adoption,
    tip,
    tool_for,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", action="store_true", help="machine-readable")
    parser.add_argument(
        "--adoption",
        action="store_true",
        help="which levels run the shared engine mechanisms (live specs)",
    )
    args = parser.parse_args()

    if args.adoption and not args.json:
        print(render_adoption())
        return 0
    if not args.json:
        print(render())
        return 0

    nxt = next_open()
    print(
        json.dumps(
            {
                "tip": None if tip() is None else tip().id,
                "next_open": None if nxt is None else nxt.id,
                "ladder": [
                    {
                        "id": step.id,
                        "bead": step.bead,
                        "segment": step.segment,
                        "rung": step.rung.name.lower(),
                        "blocker": step.blocker.value,
                        "room": step.room,
                        "pose": step.pose,
                        "residual": step.residual,
                        "pin": step.pin,
                    }
                    for step in CLEAN_LADDER
                ],
                "adoption": {
                    f"L{level}": {
                        name: {"rooms_with": have, "rooms_total": total}
                        for name, (have, total) in counts.items()
                    }
                    for level, counts in adoption().items()
                },
                "blockers": {
                    blocker.value: {
                        "steps": [step.id for step in steps],
                        "tool": tool_for(blocker),
                    }
                    for blocker, steps in by_blocker().items()
                },
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
