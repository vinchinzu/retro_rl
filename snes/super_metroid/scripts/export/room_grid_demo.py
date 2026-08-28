#!/usr/bin/env python3
"""4x4 Super Metroid autobot room-grid demo.

```bash
uv run python snes/super_metroid/scripts/export/room_grid_demo.py --list
uv run python snes/super_metroid/scripts/export/room_grid_demo.py \
  --probe-parallel --workers 16
uv run python snes/super_metroid/scripts/export/room_grid_demo.py --seconds 30
uv run python snes/super_metroid/scripts/export/room_grid_demo.py \
  --workers 16 --seconds 30
```

Not continuous evidence. Default is one emulator process.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from super_metroid.paths import RECORDINGS_DIR
from super_metroid.demo.room_grid import (
    DEFAULT_COLS,
    DEFAULT_ROWS,
    DEFAULT_SECONDS,
    DEFAULT_TILES,
    NTSC_FPS,
    composite_grid,
    probe_parallel,
    record_tiles,
    tile_inventory,
)


def _print_inventory(rows: list[dict[str, object]]) -> None:
    print(
        f"{'#':>2}  {'pin':7}  {'seg':5}  {'room':8}  {'label':16}  segment"
    )
    missing = 0
    for row in rows:
        pin = "OK" if row["pinExists"] else "MISSING"
        seg = "OK" if row["segmentRegistered"] else "NO"
        if pin != "OK" or seg != "OK":
            missing += 1
        print(
            f"{int(row['index']):2d}  {pin:7}  {seg:5}  "
            f"{row['roomIdHex']:8}  {str(row['label'])[:16]:16}  "
            f"{row['segment']}"
        )
    print(f"{len(rows)} tiles, {missing} not ready")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="4x4 Super Metroid autobot room-grid demo (~30s).",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="Print the 16 tiles + pin existence and exit (no emulator)",
    )
    parser.add_argument(
        "--probe-parallel",
        action="store_true",
        help="Print host verdict for --workers and exit (no emulator)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Emulator processes (default 1; 16 is one process per tile)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Record with workers>1 even when load/RAM verdict is no",
    )
    parser.add_argument("--seconds", type=float, default=DEFAULT_SECONDS)
    parser.add_argument("--scale", type=int, default=1)
    parser.add_argument("--crf", type=int, default=23)
    parser.add_argument(
        "--output",
        type=Path,
        default=RECORDINGS_DIR / "room_grid.mp4",
    )
    parser.add_argument(
        "--tile-dir",
        type=Path,
        default=RECORDINGS_DIR / "room_grid_tiles",
    )
    parser.add_argument(
        "--composite-only",
        action="store_true",
        help="xstack existing --tile-dir mp4s; do not boot emulators",
    )
    args = parser.parse_args(argv)

    tiles = DEFAULT_TILES
    if len(tiles) != DEFAULT_COLS * DEFAULT_ROWS:
        parser.error(
            f"DEFAULT_TILES is {len(tiles)}, need "
            f"{DEFAULT_COLS * DEFAULT_ROWS}"
        )

    if args.list:
        _print_inventory(tile_inventory(tiles))
        return 0

    if args.probe_parallel:
        verdict = probe_parallel(max(1, int(args.workers)))
        print(json.dumps(verdict.to_dict(), indent=2))
        return 0 if verdict.ok else 2

    max_frames = max(1, int(round(float(args.seconds) * NTSC_FPS)))
    if args.composite_only:
        clips = sorted(args.tile_dir.glob("*.mp4"))
        if len(clips) != len(tiles):
            print(
                f"composite-only: {len(clips)} clips in {args.tile_dir}, "
                f"need {len(tiles)}",
                flush=True,
            )
            return 1
        composite_grid(clips, args.output, seconds=float(args.seconds))
        print(json.dumps({"output": str(args.output), "tiles": len(clips)}))
        return 0

    reports = record_tiles(
        tiles,
        args.tile_dir,
        workers=max(1, int(args.workers)),
        max_frames=max_frames,
        scale=int(args.scale),
        crf=int(args.crf),
        force=bool(args.force),
    )
    clips = [Path(str(row["video"])) for row in reports]
    missing = [str(p) for p in clips if not p.is_file()]
    payload = {
        "workers": int(args.workers),
        "seconds": float(args.seconds),
        "tiles": reports,
        "output": str(args.output),
        "missingClips": missing,
        "note": "room-grid demo only — not hop GREEN / not continuous evidence",
    }
    if missing:
        print(json.dumps(payload, indent=2))
        return 1
    composite_grid(clips, args.output, seconds=float(args.seconds))
    payload["bytes"] = args.output.stat().st_size if args.output.is_file() else 0
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
