#!/usr/bin/env python3
"""Rank high-dwell continuous splits / action_reasons without re-running the emu.

```bash
uv run python snes/super_metroid/scripts/export/split_dwell.py \\
  super_metroid/recordings/varia.json --top 15
uv run python snes/super_metroid/scripts/export/split_dwell.py \\
  super_metroid/recordings/varia.json --reasons --top 20
```
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
_SNES_IMPORT_ROOT = Path(__file__).resolve().parents[3]
for _p in (ROOT, globals().get('_SNES_IMPORT_ROOT', ROOT)):
    if _p is not None and str(_p) not in sys.path:
        sys.path.insert(0, str(_p))
from super_metroid.room_timer import (  # noqa: E402
    action_reason_hotspots,
    rank_split_dwells,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "report",
        type=Path,
        help="Continuous report JSON (e.g. recordings/varia.json)",
    )
    parser.add_argument("--top", type=int, default=15, help="Rows to print")
    parser.add_argument(
        "--min-dwell",
        type=int,
        default=200,
        help="Ignore split dwells below this (default 200)",
    )
    parser.add_argument(
        "--reasons",
        action="store_true",
        help="Rank action_reasons instead of split dwells",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Emit JSON instead of a table",
    )
    args = parser.parse_args(argv)

    payload = json.loads(args.report.read_text(encoding="utf-8"))
    if args.reasons:
        rows = action_reason_hotspots(
            payload, limit=args.top, min_frames=args.min_dwell
        )
    else:
        rows = rank_split_dwells(
            payload, limit=args.top, min_dwell=args.min_dwell
        )

    if args.json:
        print(json.dumps(rows, indent=2))
        return 0

    total = payload.get("total_frames")
    outcome = payload.get("outcome")
    print(f"# {args.report.name}  outcome={outcome}  total_frames={total}")
    if args.reasons:
        print(f"{'frames':>8}  reason")
        for row in rows:
            print(f"{row['frames']:8d}  {row['reason']}")
    else:
        print(f"{'dwell':>8}  split_id  room  frame")
        for row in rows:
            print(
                f"{row['dwell_frames']:8d}  {row['split_id']}  "
                f"{row['room_id_hex']}  @{row['frame']}"
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
