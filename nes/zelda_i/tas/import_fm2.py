"""Import FCEUX ``.fm2`` Zelda movies into ``nes9_rle`` seeds + summary.

Primary (non-glitch): chatterbox all-items #4767M (31:52.07, 114_913 frames).
PRG1 ROM matches our integration.

```bash
uv run python -m zelda_i.tas.fetch_refs
uv run python -m zelda_i.tas.import_fm2 --summary-only

uv run python -m zelda_i.tas.import_fm2 \\
  nes/zelda_i/tas/ref/chatterbox_allitems_4767M.fm2 \\
  --out nes/zelda_i/models/zelda_allitems_raw.json
```
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from zelda_i.paths import GAME_DIR, MODELS_DIR
from zelda_i.tas.fm2 import export_rle_seed, parse_fm2

DEFAULT_REF = GAME_DIR / "tas" / "ref" / "chatterbox_allitems_4767M.fm2"
REF_DIR = GAME_DIR / "tas" / "ref"


def _summary_all() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not REF_DIR.is_dir():
        return rows
    for path in sorted(REF_DIR.glob("*.fm2")):
        movie = parse_fm2(path)
        s = movie.summary()
        s["file"] = path.name
        s["size_bytes"] = path.stat().st_size
        rows.append(s)
    return rows


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "fm2",
        nargs="?",
        type=Path,
        default=None,
        help="Path to .fm2 (default: vendored Lord Tom any%)",
    )
    p.add_argument(
        "--summary-only",
        action="store_true",
        help="Print metadata for all tas/ref/*.fm2 (or one path)",
    )
    p.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Write nes9_rle JSON seed here",
    )
    p.add_argument(
        "--route-id",
        default="zelda_tas_import",
        help="route_id field in seed JSON",
    )
    args = p.parse_args(argv)

    if args.summary_only and args.fm2 is None:
        rows = _summary_all()
        if not rows:
            print(
                "no movies in tas/ref/ — run: uv run python -m zelda_i.tas.fetch_refs",
                file=sys.stderr,
            )
            return 1
        print(json.dumps(rows, indent=2))
        return 0

    path = args.fm2 or DEFAULT_REF
    if not path.exists():
        print(
            f"missing {path} — run: uv run python -m zelda_i.tas.fetch_refs",
            file=sys.stderr,
        )
        return 1

    movie = parse_fm2(path)
    summary = movie.summary()
    print(json.dumps(summary, indent=2))

    if args.summary_only:
        return 0

    out = args.out
    if out is None:
        MODELS_DIR.mkdir(parents=True, exist_ok=True)
        stem = path.stem
        out = MODELS_DIR / f"{stem}_raw.json"

    payload = export_rle_seed(path, out=out, route_id=args.route_id)
    print(
        json.dumps(
            {
                "wrote": str(out),
                "num_frames": payload["num_frames"],
                "segments": len(payload["segments"]),
                "route_id": payload["route_id"],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
