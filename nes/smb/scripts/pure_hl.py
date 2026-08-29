"""Thin CLI for parked pure HappyLee isolation.

```bash
uv run python -m smb.scripts.pure_hl status
uv run python -m smb.scripts.pure_hl check-8-4-gate
```

Search/probe/export subcommands are deleted (git restores). Writes stay under
``models/pure_hl/`` and ``recordings/tas_import/pure_hl/``.
"""

from __future__ import annotations

import argparse
import json
import os
from typing import Any

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

from smb.tas import pure_hl as ph


def _print(obj: Any) -> None:
    print(json.dumps(obj, indent=2, default=str), flush=True)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description="Pure HappyLee track isolation (no hybrid / natural / skills)"
    )
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("status", help="track isolation + gate status")
    s.set_defaults(code=0, fn=lambda: ph.track_status())

    s = sub.add_parser(
        "check-8-4-gate",
        help="hard block: pure 8-4 only after pure 8-3 leave",
    )
    s.set_defaults(code=None, fn=ph.refuse_8_4_until_gate)

    args = p.parse_args(argv)
    payload = args.fn()
    _print(payload)
    if args.code is not None:
        return int(args.code)
    return 0 if payload.get("allowed") else 3


if __name__ == "__main__":
    raise SystemExit(main())
