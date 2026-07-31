#!/usr/bin/env python3
"""Bounded Kraid door/PLM reconnaissance with no guessed RAM offsets.

The local RAM map has no validated live PLM or blue-door-open field.  This
probe therefore reuses the existing read-only approach and four-shot sequence
and records the already exposed navigation/door fields while explicitly
marking the PLM/open-state portion blocked.  It never writes door, PLM,
progression, capacity, event, boss, room, or position RAM.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from super_metroid.scripts.probe.kraid_door_blue_recon import (  # noqa: E402
    DEFAULT_FRAMES,
    run_probe,
)

DEFAULT_OUTPUT = ROOT / "super_metroid" / "debug" / "kraid_door_plm_recon.json"

SEARCHED_SOURCES = [
    "super_metroid/ram.py and tests/test_ram.py",
    "super_metroid/docs/ram_map.md",
    "super_metroid/docs/tasks/SM-DOOR-BLUE-report.md",
    "super_metroid/scripts/probe/kraid_door_blue_recon.py",
    "super_metroid/scripts/probe/kraid_door_phase_recon.py",
    "super_metroid/scripts/probe/kraid_left_door_recon.py",
    "super_metroid/custom_integrations/",
    "monorepo-wide source/comments search for door, PLM, BTS, and projectile offsets",
]


def run_recon(source: Path, frames: int, output: Path) -> dict[str, object]:
    """Run the existing shot diagnostic and annotate the unresolved field gap."""
    report = run_probe(source, frames, output)
    report.update(
        {
            "kind": "kraid_door_plm_recon",
            "fieldStatus": "blocked",
            "newFields": [],
            "sampleTable": [],
            "searchedSources": SEARCHED_SOURCES,
            "blockedReason": (
                "No source-confirmed or differential live WRAM offset for a PLM "
                "record, PLM activation, blue-door open state, or door BTS was "
                "found. ROM-side PLM IDs are not safe live WRAM fields."
            ),
            "nonClaims": [
                "No blue-door open/closed determination",
                "No PLM activation determination",
                "Not pure-green or continuous evidence",
                "No STATUS promotion",
            ],
        }
    )
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print("field_status=blocked new_fields=[]")
    print("reason=no validated live PLM or blue-door-open WRAM offset")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--frames", type=int, default=DEFAULT_FRAMES)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    run_recon(args.source, args.frames, args.output)


if __name__ == "__main__":
    main()
