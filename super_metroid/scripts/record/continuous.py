#!/usr/bin/env python3
"""Record continuous power-on through a named tip on the KPDR chain.

One CLI for every milestone — do not add ``start_to_*.py`` scripts. Tips are
functions in ``routes/continuous.py`` registered in ``routes/catalog.py``.

```bash
# Current tip (Frog Savestation / KPDR K4.0)
uv run python super_metroid/scripts/record/continuous.py --no-video
uv run python super_metroid/scripts/record/continuous.py --to frog --no-video
uv run python super_metroid/scripts/record/continuous.py --to warehouse --no-video
uv run python super_metroid/scripts/record/continuous.py --to warehouse --no-video --room-timing

# Prefix milestones (shorter checks)
uv run python super_metroid/scripts/record/continuous.py --to below_spazer --no-video
uv run python super_metroid/scripts/record/continuous.py --to bat --no-video
uv run python super_metroid/scripts/record/continuous.py --to red_tower --no-video
uv run python super_metroid/scripts/record/continuous.py --to supers --no-video
uv run python super_metroid/scripts/record/continuous.py --to spore --no-video
uv run python super_metroid/scripts/record/continuous.py --to bombs --no-video
uv run python super_metroid/scripts/record/continuous.py --to morph --no-video

# List tips
uv run python super_metroid/scripts/record/continuous.py --list
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

from super_metroid.routes.catalog import (  # noqa: E402
    CONTINUOUS_TIPS,
    DEFAULT_CONTINUOUS_TIP,
    get_continuous_tip,
)
from super_metroid.routes.continuous import (  # noqa: E402
    default_tip_artifact_paths,
    default_tip_room_timing_path,
    run_to,
)


def main() -> None:
    tip_help = ", ".join(t.tip_id for t in CONTINUOUS_TIPS)
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--to",
        dest="tip",
        default=DEFAULT_CONTINUOUS_TIP,
        help=(
            f"Continuous tip to stop at (default: {DEFAULT_CONTINUOUS_TIP}). "
            f"Known: {tip_help}"
        ),
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="Print known continuous tips and exit",
    )
    parser.add_argument("--video", type=Path, default=None)
    parser.add_argument("--report", type=Path, default=None)
    parser.add_argument(
        "--state-output",
        type=Path,
        default=None,
        help=(
            "Write a reusable emulator checkpoint only after an integrity-green "
            "tip run (for continuous-like pure probes)."
        ),
    )
    parser.add_argument("--no-video", action="store_true")
    parser.add_argument("--no-unlimited-energy", action="store_true")
    parser.add_argument("--no-unlimited-ammo", action="store_true")
    parser.add_argument(
        "--room-timing",
        action="store_true",
        help=(
            "Opt-in RoomTimer JSON under recordings/room_timings/ "
            "(does not affect continuous integrity or assists; tip must support it)"
        ),
    )
    parser.add_argument(
        "--room-timing-path",
        type=Path,
        default=None,
        help="Explicit room-timing JSON path (implies --room-timing)",
    )
    args = parser.parse_args()

    if args.list:
        for tip in CONTINUOUS_TIPS:
            marker = "  (default tip)" if tip.tip_id == DEFAULT_CONTINUOUS_TIP else ""
            print(f"{tip.tip_id:12}  {tip.display_name}{marker}")
            if tip.description:
                print(f"{'':12}  {tip.description}")
        return

    tip = get_continuous_tip(args.tip)
    default_video, default_report = default_tip_artifact_paths(tip)
    video_path = args.video if args.video is not None else default_video
    report_path = args.report if args.report is not None else default_report

    room_timing_path: Path | None = args.room_timing_path
    if args.room_timing and room_timing_path is None:
        if not tip.supports_room_timing:
            parser.error(
                f"tip {tip.tip_id!r} does not support --room-timing "
                f"(use supers or red_tower)"
            )
        room_timing_path = default_tip_room_timing_path(tip)
    elif room_timing_path is not None and not tip.supports_room_timing:
        parser.error(f"tip {tip.tip_id!r} does not support room timing")

    report = run_to(
        tip.tip_id,
        video_path=None if args.no_video else video_path,
        report_path=report_path,
        unlimited_energy=not args.no_unlimited_energy,
        unlimited_ammo=not args.no_unlimited_ammo,
        room_timing_path=room_timing_path,
        state_output=args.state_output,
    )
    payload = report.to_dict()
    payload["continuous_tip"] = tip.tip_id
    if room_timing_path is not None:
        payload["room_timing_path"] = str(room_timing_path)
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
