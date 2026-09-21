"""CLI: export ALttP TAS slices from vendored movies.

```bash
uv run python -m alttp.tas.export_slices --list
uv run python -m alttp.tas.export_slices --verified
uv run python -m alttp.tas.export_slices m_riss_100_wip_menu
```

Castle / sewers / sanctuary stubs are listed but not exported.
"""

from __future__ import annotations

import argparse
import json
import sys

from alttp.paths import GAME_DIR
from alttp.tas.slice import (
    SLICE_CATALOG,
    SLICE_DIR,
    export_slice,
    load_movie_frames,
    verified_slice_ids,
)


def _write_manifest(ids: list[str]) -> None:
    SLICE_DIR.mkdir(parents=True, exist_ok=True)
    man_path = SLICE_DIR / "manifest.json"
    existing: dict = {}
    if man_path.exists():
        existing = json.loads(man_path.read_text(encoding="utf-8")).get("slices", {})
    for sid in ids:
        sp = SLICE_CATALOG[sid]
        existing[sid] = {
            "path": f"slices/{sid}.json",
            "movie": str(sp.movie.relative_to(GAME_DIR)).replace("\\", "/"),
            "start": sp.start,
            "end": sp.end,
            "tags": list(sp.tags),
            "source": sp.source,
            "notes": sp.notes,
            "verified": sp.verified,
        }
    existing = {
        k: v for k, v in existing.items() if (SLICE_DIR / f"{k}.json").exists()
    }
    man_path.write_text(
        json.dumps({"slices": existing}, indent=2) + "\n", encoding="utf-8"
    )


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("slices", nargs="*", help="Slice ids (default: verified set)")
    p.add_argument(
        "--verified",
        action="store_true",
        help="Export verified slices only (default when no ids given)",
    )
    p.add_argument("--list", action="store_true", help="List catalog and exit")
    args = p.parse_args(argv)

    if args.list:
        for sid, sp in SLICE_CATALOG.items():
            flag = "ok" if sp.verified else "stub"
            print(f"{sid:28s} {flag:4s} tags={','.join(sp.tags)}  {sp.notes[:60]}")
        return 0

    if args.slices:
        ids = list(args.slices)
    else:
        ids = verified_slice_ids()

    cache: dict = {}
    for sid in ids:
        if sid not in SLICE_CATALOG:
            print(f"unknown slice: {sid}", file=sys.stderr)
            return 2
        sp = SLICE_CATALOG[sid]
        try:
            if sp.movie not in cache:
                print(f"parse {sp.movie.name}…", file=sys.stderr)
                cache[sp.movie] = load_movie_frames(sp.movie, sp.kind)
            payload = export_slice(sp, frames=cache[sp.movie])
        except (FileNotFoundError, ValueError) as exc:
            print(f"skip {sid}: {exc}", file=sys.stderr)
            continue
        print(f"{sid}: {payload['num_frames']} frames → slices/{sid}.json")

    _write_manifest([i for i in ids if (SLICE_DIR / f"{i}.json").exists()])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
