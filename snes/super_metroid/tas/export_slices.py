"""CLI: export Super Metroid TAS slices from vendored movies.

```bash
# Finish-oriented slices (any% + 100% late/escape + menu/ceres smoke)
uv run python -m super_metroid.tas.export_slices --finish

# Everything including full any%/100% RLE (large JSON)
uv run python -m super_metroid.tas.export_slices --all

# One slice
uv run python -m super_metroid.tas.export_slices sniq_any_final_10k
```
"""

from __future__ import annotations

import argparse
import json
import sys

from super_metroid.paths import GAME_DIR
from super_metroid.tas.slice import (
    SLICE_CATALOG,
    SLICE_DIR,
    export_slice,
    finish_slice_ids,
    load_movie_frames,
)

_DEFAULT_IDS = sorted(
    set(finish_slice_ids())
    | {
        "sniq_any_menu",
        "sniq_100_menu",
        "sniq_any_ceres_open",
        "sniq_100_ceres_open",
        "sniq_any_wip_full",
        "moozooh_smtc4_full",
    }
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
        }
    existing = {
        k: v for k, v in existing.items() if (SLICE_DIR / f"{k}.json").exists()
    }
    man_path.write_text(
        json.dumps({"slices": existing}, indent=2) + "\n", encoding="utf-8"
    )


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("slices", nargs="*", help="Slice ids (default: finish set)")
    p.add_argument(
        "--finish",
        action="store_true",
        help="Export finish-tagged slices + menu/ceres/short smoke",
    )
    p.add_argument(
        "--all",
        action="store_true",
        help="Export entire catalog including full movies",
    )
    p.add_argument("--list", action="store_true", help="List catalog and exit")
    args = p.parse_args(argv)

    if args.list:
        for sid, sp in SLICE_CATALOG.items():
            print(f"{sid:28s} tags={','.join(sp.tags)}  {sp.notes[:60]}")
        return 0

    if args.all:
        ids = list(SLICE_CATALOG.keys())
    elif args.slices:
        ids = list(args.slices)
    else:
        ids = list(_DEFAULT_IDS)

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
        except FileNotFoundError as exc:
            print(f"skip {sid}: {exc}", file=sys.stderr)
            continue
        print(f"{sid}: {payload['num_frames']} frames → slices/{sid}.json")

    _write_manifest([i for i in ids if (SLICE_DIR / f"{i}.json").exists()])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
