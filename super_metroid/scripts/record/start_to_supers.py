#!/usr/bin/env python3
"""Record continuous power-on through Spore Super Missile collect.

```bash
uv run python super_metroid/scripts/record/start_to_supers.py --no-video
uv run python super_metroid/scripts/record/start_to_supers.py
# Opt-in per-room timing artifact (shared RoomTimer; does not affect integrity):
uv run python super_metroid/scripts/record/start_to_supers.py --no-video --room-timing
```
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from super_metroid.routes.continuous import (  # noqa: E402
    default_supers_artifact_paths,
    default_supers_room_timing_path,
    run_start_to_supers,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    default_video, default_report = default_supers_artifact_paths()
    parser.add_argument("--video", type=Path, default=default_video)
    parser.add_argument("--report", type=Path, default=default_report)
    parser.add_argument("--no-video", action="store_true")
    parser.add_argument("--no-unlimited-energy", action="store_true")
    parser.add_argument("--no-unlimited-ammo", action="store_true")
    parser.add_argument(
        "--room-timing",
        action="store_true",
        help=(
            "Opt-in: observe each frame with super_metroid.room_timer.RoomTimer "
            "and write a separate JSON under recordings/room_timings/ "
            "(does not affect continuous integrity or assists)"
        ),
    )
    parser.add_argument(
        "--room-timing-path",
        type=Path,
        default=None,
        help=(
            "Explicit room-timing JSON path (implies --room-timing). "
            f"Default when --room-timing: {default_supers_room_timing_path()}"
        ),
    )
    args = parser.parse_args()
    room_timing_path: Path | None = args.room_timing_path
    if args.room_timing and room_timing_path is None:
        room_timing_path = default_supers_room_timing_path()
    report = run_start_to_supers(
        video_path=None if args.no_video else args.video,
        report_path=args.report,
        unlimited_energy=not args.no_unlimited_energy,
        unlimited_ammo=not args.no_unlimited_ammo,
        room_timing_path=room_timing_path,
    )
    payload = report.to_dict()
    if room_timing_path is not None:
        payload["room_timing_path"] = str(room_timing_path)
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
