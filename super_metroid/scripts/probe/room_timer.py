#!/usr/bin/env python3
"""Super Metroid room-timing probe (stable-retro / stock ROM).

Offline mode needs only a JSON snapshot fixture (no ROM). Session mode boots
an integration save state and records hops while the emulator steps.

Examples:

```bash
# Offline unit-style fixture → JSON under super_metroid/
uv run python super_metroid/scripts/probe/room_timer.py offline \\
  --input super_metroid/tests/fixtures/room_timer_sample.json \\
  --output super_metroid/recordings/room_timings/offline_sample.json

# Live session from a named anchor (requires ROM + integration)
uv run python super_metroid/scripts/probe/room_timer.py session \\
  --state super_metroid/custom_integrations/SuperMetroid-Snes/dev_red_tower_stable.state \\
  --frames 600 \\
  --output super_metroid/recordings/room_timings/session.json
```
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from super_metroid.paths import GAME_DIR, ROOM_TIMINGS_DIR  # noqa: E402
from super_metroid.room_timer import (  # noqa: E402
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
    if not out.is_absolute() and not str(out).startswith(str(GAME_DIR)):
        # Keep durable artifacts under super_metroid/ when a bare name is given.
        if out.parent == Path("."):
            out = _default_output(out.name)
    _write_report(out, report)
    return 0


def cmd_session(args: argparse.Namespace) -> int:
    from retro_harness.actions import idle_action
    from super_metroid.dev.common import boot_from_state, make_dev_env
    from super_metroid.ram import parse_env_state

    state_path = Path(args.state)
    if not state_path.is_file():
        print(f"error: save state not found: {state_path}", file=sys.stderr)
        return 2

    env = make_dev_env()
    try:
        boot_from_state(env, state_path, settle_frames=args.settle)
        timer = RoomTimer()
        # Seed from post-boot state at frame 0 of the timing session.
        state = parse_env_state(env, frame=0)
        timer.observe(state)
        for frame in range(1, args.frames + 1):
            env.step(idle_action())
            state = parse_env_state(env, frame=frame)
            timer.observe(state)
        timer.finalize(frame=args.frames)
        report = timer.report(
            source=str(state_path.resolve()),
            extra={
                "mode": "session",
                "requested_frames": args.frames,
                "settle_frames": args.settle,
                "final_room_id": state.room_id,
                "final_room_id_hex": f"0x{state.room_id:04X}",
                "final_game_state": state.game_state,
                "final_phase": state.phase.value,
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
    """Import/syntax smoke without ROM: synthetic two-room hop."""
    samples = [
        TimingSnapshot(frame=0, room_id=0x9AD9, area_index=1, game_state=8),
        TimingSnapshot(frame=10, room_id=0x9AD9, area_index=1, game_state=8),
        TimingSnapshot(
            frame=11,
            room_id=0x9AD9,
            area_index=1,
            game_state=9,
            door_transition=1,
            transition_direction=1,
        ),
        TimingSnapshot(
            frame=40,
            room_id=0x9AD9,
            area_index=1,
            game_state=11,
            door_transition=1,
            transition_direction=1,
        ),
        TimingSnapshot(frame=50, room_id=0x9B5B, area_index=1, game_state=8),
    ]
    report = run_offline(samples, source="self_check")
    assert report["visit_count"] == 1, report
    visit = report["visits"][0]
    assert visit["room_id"] == 0x9AD9
    assert visit["dest_room_id"] == 0x9B5B
    assert visit["room_frames"] == 50
    assert visit["dwell_frames"] == 11
    print("self_check ok:", json.dumps({"visit_count": 1, "room_frames": 50}))
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
    session.add_argument("--state", type=Path, required=True, help="Integration .state path")
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
