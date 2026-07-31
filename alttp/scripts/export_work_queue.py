#!/usr/bin/env python3
"""Export ALTTP Sanctuary-path save-state work queue (JSON + markdown)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from alttp.opening_route.work_queue import (  # noqa: E402
    DEFAULT_QUEUE_JSON,
    DEFAULT_QUEUE_MD,
    build_work_queue,
    export_work_queue,
    top_items,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Rank Zelda3-Snes .state files for Sanctuary progress "
            "(post-sword: 0x55 exit / key / shutter first)."
        )
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print full JSON payload to stdout (still writes artifacts unless --no-write).",
    )
    parser.add_argument(
        "--top",
        type=int,
        default=15,
        help="How many ranked items to print (default 15). Ignored with --json.",
    )
    parser.add_argument(
        "--json-out",
        type=Path,
        default=DEFAULT_QUEUE_JSON,
        help=f"JSON output path (default: {DEFAULT_QUEUE_JSON})",
    )
    parser.add_argument(
        "--md-out",
        type=Path,
        default=DEFAULT_QUEUE_MD,
        help=f"Markdown output path (default: {DEFAULT_QUEUE_MD})",
    )
    parser.add_argument(
        "--no-write",
        action="store_true",
        help="Do not write artifact files; only print.",
    )
    args = parser.parse_args(argv)

    if args.no_write:
        payload = build_work_queue()
    else:
        payload = export_work_queue(
            json_output=args.json_out,
            md_output=args.md_out,
        )

    if args.json:
        print(json.dumps(payload, indent=2))
        return 0

    summary = payload.get("summary") or {}
    print(f"ALTTP work queue — {summary.get('stateCount', '?')} states")
    print(f"byStatus: {summary.get('byStatus')}")
    print(f"byTier:   {summary.get('byTier')}")
    print(f"Sanctuary claimed: {summary.get('sanctuaryClaimed')}")
    if not args.no_write:
        print(f"wrote: {args.json_out}")
        print(f"wrote: {args.md_out}")
    print()
    print(f"Top {args.top} (Sanctuary progress order):")
    print(f"{'rank':>4}  {'status':<16}  {'tier':<8}  {'group':<12}  {'goal':<18}  state")
    for row in top_items(payload, n=args.top):
        print(
            f"{row.get('rank'):>4}  "
            f"{str(row.get('status')):<16}  "
            f"{str(row.get('tier')):<8}  "
            f"{str(row.get('group')):<12}  "
            f"{str(row.get('goal')):<18}  "
            f"{row.get('state_name')}"
        )
    focus = payload.get("workFocus") or []
    if focus:
        print()
        print("Work focus:")
        for row in focus[:8]:
            note = (row.get("notes") or "")[:70]
            print(f"  - {row.get('state_name')}: {row.get('goal')} [{row.get('status')}] {note}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
