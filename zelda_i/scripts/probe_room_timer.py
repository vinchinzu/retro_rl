#!/usr/bin/env python3
"""Zelda I screen/room-timing probe (stable-retro / stock ROM).

Offline mode needs only a JSON snapshot fixture (no ROM). Session mode boots
an integration save state and records hops while the emulator steps idle.

Examples:

```bash
# Offline unit-style fixture → JSON under zelda_i/
uv run python zelda_i/scripts/probe_room_timer.py offline \\
  --input zelda_i/tests/fixtures/room_timer_sample.json \\
  --output zelda_i/recordings/room_timings/offline_sample.json

# Live idle session from a named integration state (requires ROM)
uv run python zelda_i/scripts/probe_room_timer.py session \\
  --state Level1Entrance \\
  --frames 600 \\
  --output zelda_i/recordings/room_timings/session.json

# Import / logic smoke (no ROM)
uv run python zelda_i/scripts/probe_room_timer.py self-check
```
"""

# Script execution adds the repository root before importing local packages.
# ruff: noqa: E402

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from zelda_i.paths import GAME, GAME_DIR, ROOM_TIMINGS_DIR
from zelda_i.room_timer import (
    RoomTimer,
    TimingSnapshot,
    run_offline,
    snapshots_from_json,
)


def _default_output(name: str) -> Path:
    ROOM_TIMINGS_DIR.mkdir(parents=True, exist_ok=True)
    return ROOM_TIMINGS_DIR / name


def _write_report(path: Path, report: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {path} ({report.get('visit_count', 0)} visits)")


def cmd_offline(args: argparse.Namespace) -> int:
    raw = json.loads(Path(args.input).read_text(encoding="utf-8"))
    samples = snapshots_from_json(raw)
    report = run_offline(samples, source=str(Path(args.input).resolve()))
    out = Path(args.output) if args.output else _default_output("offline_room_timing.json")
    if out.parent == Path("."):
        out = _default_output(out.name)
    _write_report(out, report)
    return 0


def cmd_session(args: argparse.Namespace) -> int:
    from retro_harness.env import make_env
    from retro_harness.nes import nes_idle_action
    from snes_oneshot.segment_runner import configure_headless
    from zelda_i.ram import read_snapshot

    configure_headless()
    env = make_env(GAME, args.state, GAME_DIR, render_mode="rgb_array")
    try:
        result = env.reset()
        _obs = result[0] if isinstance(result, tuple) else result
        # One idle settle step so RAM reflects post-load play.
        for _ in range(max(1, args.settle)):
            env.step(nes_idle_action())

        timer = RoomTimer()
        snap = read_snapshot(env.get_ram())
        timer.observe(snap, frame=0)
        for frame in range(1, args.frames + 1):
            env.step(nes_idle_action())
            snap = read_snapshot(env.get_ram())
            timer.observe(snap, frame=frame)
        timer.finalize(frame=args.frames)
        report = timer.report(
            source=f"session:{args.state}",
            extra={
                "mode": "session",
                "state": args.state,
                "requested_frames": args.frames,
                "settle_frames": args.settle,
                "final_mode": snap.mode,
                "final_level": snap.level,
                "final_screen": snap.screen,
                "final_screen_hex": f"0x{snap.screen:02X}",
            },
        )
    finally:
        env.close()

    out = Path(args.output) if args.output else _default_output("session_room_timing.json")
    if out.parent == Path("."):
        out = _default_output(out.name)
    _write_report(out, report)
    return 0


def cmd_self_check(_args: argparse.Namespace) -> int:
    """Import/syntax smoke without ROM: synthetic two-screen hop."""
    samples = [
        TimingSnapshot(frame=0, mode=5, level=0, screen=0x77),
        TimingSnapshot(frame=10, mode=5, level=0, screen=0x77),
        TimingSnapshot(frame=11, mode=6, level=0, screen=0x77, next_screen=0x78),
        TimingSnapshot(frame=40, mode=7, level=0, screen=0x78, next_screen=0x78),
        TimingSnapshot(frame=50, mode=5, level=0, screen=0x78),
    ]
    report = run_offline(samples, source="self_check")
    assert report["visit_count"] == 1, report
    visit = report["visits"][0]
    assert visit["screen"] == 0x77
    assert visit["dest_screen"] == 0x78
    assert visit["location_frames"] == 50
    assert visit["dwell_frames"] == 11
    print(
        "self_check ok:",
        json.dumps({"visit_count": 1, "location_frames": 50, "dwell_frames": 11}),
    )
    return 0


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    offline = sub.add_parser("offline", help="Process a JSON snapshot fixture")
    offline.add_argument("--input", "-i", type=Path, required=True, help="JSON samples file")
    offline.add_argument(
        "--output",
        "-o",
        type=Path,
        default=None,
        help=f"Output JSON (default: {ROOM_TIMINGS_DIR}/offline_room_timing.json)",
    )
    offline.set_defaults(func=cmd_offline)

    session = sub.add_parser("session", help="Time hops while idling from a save state")
    session.add_argument(
        "--state",
        type=str,
        default="Level1",
        help="Integration state name (e.g. Level1, Level1Entrance)",
    )
    session.add_argument("--frames", type=int, default=600, help="Emulator steps after settle")
    session.add_argument("--settle", type=int, default=5, help="Post-load settle frames")
    session.add_argument(
        "--output",
        "-o",
        type=Path,
        default=None,
        help=f"Output JSON (default: {ROOM_TIMINGS_DIR}/session_room_timing.json)",
    )
    session.set_defaults(func=cmd_session)

    check = sub.add_parser("self-check", help="Synthetic hop import/logic smoke (no ROM)")
    check.set_defaults(func=cmd_self_check)

    args = parser.parse_args()
    raise SystemExit(args.func(args))


if __name__ == "__main__":
    main()
