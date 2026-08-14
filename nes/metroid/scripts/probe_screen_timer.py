#!/usr/bin/env python3
"""NES Metroid screen-timing probe (stable-retro / stock ROM).

Offline mode needs only a JSON snapshot fixture (no ROM). self-check runs a
synthetic two-cell hop for import/logic smoke.

Examples:

```bash
# Import / logic smoke (no ROM)
uv run python metroid/scripts/probe_screen_timer.py self-check

# Offline fixture → durable JSON under metroid/
uv run python metroid/scripts/probe_screen_timer.py offline \\
  --input metroid/tests/fixtures/screen_timer_sample.json \\
  --output metroid/recordings/screen_timings/offline_sample.json
```
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from metroid.paths import GAME_DIR, SCREEN_TIMINGS_DIR
from metroid.screen_timer import (
    TimingSnapshot,
    run_offline,
    snapshots_from_json,
)


def _default_output(name: str) -> Path:
    SCREEN_TIMINGS_DIR.mkdir(parents=True, exist_ok=True)
    return SCREEN_TIMINGS_DIR / name


def _write_report(path: Path, report: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {path} ({report.get('visit_count', 0)} visits)")


def cmd_offline(args: argparse.Namespace) -> int:
    raw = json.loads(Path(args.input).read_text(encoding="utf-8"))
    samples = snapshots_from_json(raw)
    report = run_offline(samples, source=str(Path(args.input).resolve()))
    out = Path(args.output) if args.output else _default_output(
        "offline_screen_timing.json"
    )
    if not out.is_absolute() and out.parent == Path("."):
        out = _default_output(out.name)
    # Keep durable artifacts under metroid/ when possible.
    try:
        out.resolve().relative_to(GAME_DIR.resolve())
    except ValueError:
        if out.parent == Path("."):
            out = _default_output(out.name)
    _write_report(out, report)
    return 0


def cmd_self_check(_args: argparse.Namespace) -> int:
    """Import/syntax smoke without ROM: synthetic two-cell hop."""
    samples = [
        TimingSnapshot(
            frame=0,
            map_x=3,
            map_y=14,
            game_mode=3,
            health_lo=0x00,
            health_hi=0x03,
            area=0x10,
        ),
        TimingSnapshot(
            frame=10,
            map_x=3,
            map_y=14,
            game_mode=3,
            health_lo=0x00,
            health_hi=0x03,
            area=0x10,
        ),
        TimingSnapshot(
            frame=11,
            map_x=3,
            map_y=14,
            game_mode=3,
            in_door=1,
            health_lo=0x00,
            health_hi=0x03,
            area=0x10,
        ),
        TimingSnapshot(
            frame=40,
            map_x=3,
            map_y=14,
            game_mode=3,
            in_door=1,
            health_lo=0x00,
            health_hi=0x03,
            area=0x10,
        ),
        TimingSnapshot(
            frame=50,
            map_x=2,
            map_y=14,
            game_mode=3,
            in_door=0,
            health_lo=0x00,
            health_hi=0x03,
            area=0x10,
        ),
    ]
    report = run_offline(samples, source="self_check")
    assert report["visit_count"] == 1, report
    visit = report["visits"][0]
    assert visit["map_cell"] == [3, 14]
    assert visit["dest_map_cell"] == [2, 14]
    assert visit["screen_frames"] == 50
    assert visit["dwell_frames"] == 11
    print(
        "self_check ok:",
        json.dumps(
            {
                "visit_count": 1,
                "screen_frames": 50,
                "map_cell": visit["map_cell"],
                "dest_map_cell": visit["dest_map_cell"],
            }
        ),
    )
    return 0


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    offline = sub.add_parser("offline", help="Process a JSON snapshot fixture")
    offline.add_argument(
        "--input", "-i", type=Path, required=True, help="JSON samples file"
    )
    offline.add_argument(
        "--output",
        "-o",
        type=Path,
        default=None,
        help=f"Output JSON (default: {SCREEN_TIMINGS_DIR}/offline_screen_timing.json)",
    )
    offline.set_defaults(func=cmd_offline)

    check = sub.add_parser(
        "self-check", help="Synthetic hop import/logic smoke (no ROM)"
    )
    check.set_defaults(func=cmd_self_check)

    args = parser.parse_args()
    raise SystemExit(args.func(args))


if __name__ == "__main__":
    main()
