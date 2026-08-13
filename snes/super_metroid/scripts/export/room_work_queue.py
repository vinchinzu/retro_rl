#!/usr/bin/env python3
"""Export the easiest-first room practice work queue.

```bash
uv run python snes/super_metroid/scripts/export/room_work_queue.py
uv run python snes/super_metroid/scripts/export/room_work_queue.py --json PATH --csv PATH
```
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
from super_metroid.rooms.work_queue import (  # noqa: E402
    DEFAULT_QUEUE_CSV,
    DEFAULT_QUEUE_JSON,
    DEFAULT_QUEUE_MD,
    export_work_queue,
)
from super_metroid.paths import ROOM_PROBLEMS_PATH  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--catalog", type=Path, default=ROOM_PROBLEMS_PATH)
    parser.add_argument("--json", type=Path, default=DEFAULT_QUEUE_JSON)
    parser.add_argument("--csv", type=Path, default=DEFAULT_QUEUE_CSV)
    parser.add_argument("--md", type=Path, default=DEFAULT_QUEUE_MD)
    args = parser.parse_args()
    if not args.catalog.is_file():
        raise SystemExit(
            f"missing room problems catalog: {args.catalog}\n"
            "Regenerate with: uv run python snes/super_metroid/scripts/export/room_problems.py"
        )
    payload = export_work_queue(
        catalog_path=args.catalog,
        json_output=args.json,
        csv_output=args.csv,
        md_output=args.md,
    )
    summary = payload["summary"]
    pct = summary["percentComplete"]
    print(
        f"wrote {args.json}: {summary['problemCount']} problems ranked easiest→hardest"
    )
    print(f"wrote {args.csv}")
    print(f"wrote {args.md}")
    print(
        "progress: "
        f"easy+standard ready {pct['easyAndStandardReady']}% | "
        f"all ready {pct['allProblemsReady']}% | "
        f"teleport {pct['allTeleportReady']}% | "
        f"directed edges {summary.get('directedEdgeCount')}"
    )
    focus = summary["workFocus"]
    print(
        f"next open easy ({len(focus['nextOpenEasyProblemIds'])} shown): "
        + ", ".join(focus["nextOpenEasyProblemIds"][:5])
    )
    print(json.dumps({"percentComplete": pct, "workFocus": focus}, indent=2))


if __name__ == "__main__":
    main()
