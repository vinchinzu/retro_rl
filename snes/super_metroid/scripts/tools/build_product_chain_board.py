#!/usr/bin/env python3
"""Build the full_start_v1 product-chain hop board (rr-4nli work list).

Offline. No emulator. Autopilot join contract is printed with the summary:
RoomAutopilot + room_adapter absorb subpixel / door speed / enemy phase.

```bash
uv run python snes/super_metroid/scripts/tools/build_product_chain_board.py --summary
uv run python snes/super_metroid/scripts/tools/build_product_chain_board.py --write
```
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
_SNES_IMPORT_ROOT = Path(__file__).resolve().parents[3]
for _p in (ROOT, _SNES_IMPORT_ROOT):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from super_metroid.human_tape.product_chain import (  # noqa: E402
    DEFAULT_BOARD,
    DEFAULT_TASK,
    build_product_chain_board,
    format_board_summary,
    write_product_chain_board,
)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--task", type=Path, default=DEFAULT_TASK)
    p.add_argument("--out", type=Path, default=DEFAULT_BOARD)
    p.add_argument("--write", action="store_true", help="Write the board JSON")
    p.add_argument("--summary", action="store_true", help="Print counts + next hop")
    p.add_argument("--no-live", action="store_true")
    args = p.parse_args(argv)

    if args.write:
        board = write_product_chain_board(
            args.task, out=args.out, include_live=not args.no_live
        )
        print(f"wrote {board.get('written')}")
    else:
        board = build_product_chain_board(
            args.task, include_live=not args.no_live
        )
    if args.summary or not args.write:
        print(format_board_summary(board))
    if args.write and args.out.is_file() and not args.summary:
        # still print one-line counts
        c = board.get("counts") or {}
        print(json.dumps({"counts": c, "next": board.get("next_hop")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
