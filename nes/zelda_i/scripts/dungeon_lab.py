"""Probe and optimize one Zelda I dungeon room from a predecessor checkpoint.

Example::

    uv run python zelda_i/scripts/dungeon_lab.py \
      --state Level1Cleared53 --door east --expected-room 0x54 \
      --enemy-type 0x1b --alive-by type --reward auto \
      --attack-phases 0,2,4,6 --trials 2 --jobs 4 \
      --save-state Level1Cleared54
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from zelda_i.dungeon import AliveRule, ensure_default_specs, spec_for_room
from zelda_i.dungeon_lab import LabRequest, run_lab

ensure_default_specs()
from zelda_i.dungeon_trace import (
    first_trace_divergence,
    read_jsonl,
)

def _int_auto(value: str) -> int:
    return int(value, 0)

def _int_csv(value: str) -> tuple[int, ...]:
    return tuple(_int_auto(part.strip()) for part in value.split(",") if part.strip())

def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--state", default="Level1Cleared53")
    parser.add_argument(
        "--room",
        "--expected-room",
        dest="room_id",
        type=_int_auto,
    )
    parser.add_argument(
        "--door",
        choices=("north", "south", "west", "east"),
        help="Validate the room spec's entry direction",
    )
    parser.add_argument("--enemy-type", type=_int_auto, action="append", default=[])
    parser.add_argument("--alive-by", choices=("type", "hp"))
    parser.add_argument("--reward", choices=("spec", "auto", "clear"), default="spec")
    parser.add_argument("--attack-phases", type=_int_csv, default=(0,))
    parser.add_argument("--engage-distances", type=_int_csv, default=())
    parser.add_argument(
        "--trials",
        type=int,
        default=1,
        help="Trials per attack-phase/distance configuration",
    )
    parser.add_argument("--jobs", type=int, default=1)
    parser.add_argument("--max-frames", type=int)
    parser.add_argument("--tail-frames", type=int, default=120)
    parser.add_argument("--output-dir")
    parser.add_argument("--save-state")
    parser.add_argument("--no-probe-exits", action="store_true")
    parser.add_argument(
        "--diff-traces",
        nargs=2,
        metavar=("LEFT_JSONL", "RIGHT_JSONL"),
        help="Only report the first divergence between two existing traces",
    )
    return parser

def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    if args.diff_traces:
        divergence = first_trace_divergence(
            read_jsonl(Path(args.diff_traces[0])),
            read_jsonl(Path(args.diff_traces[1])),
        )
        print(json.dumps(divergence, indent=2, sort_keys=True))
        return 0

    if args.room_id is None:
        parser.error("--room/--expected-room is required")
    spec = spec_for_room(args.room_id)
    if args.door:
        expected = {
            "north": "UP",
            "south": "DOWN",
            "west": "LEFT",
            "east": "RIGHT",
        }[args.door]
        if spec.entry.direction != expected:
            parser.error(
                f"room 0x{args.room_id:02X} enters via "
                f"{spec.entry.direction}, not {expected}"
            )

    request = LabRequest(
        state=args.state,
        room_id=args.room_id,
        trials_per_config=args.trials,
        jobs=args.jobs,
        attack_phases=args.attack_phases,
        engage_distances=args.engage_distances,
        enemy_types=tuple(args.enemy_type),
        alive_rule=AliveRule(args.alive_by) if args.alive_by else None,
        reward_mode=args.reward,
        max_frames=args.max_frames,
        tail_frames=args.tail_frames,
        output_dir=args.output_dir,
        save_state=args.save_state,
        probe_exits=not args.no_probe_exits,
    )
    summary = run_lab(request)
    print(
        f"room=0x{args.room_id:02X} successes={summary['successes']}/"
        f"{summary['trial_count']} output={summary['summary_path']}"
    )
    for rank, policy in enumerate(summary["ranking"], start=1):
        print(
            f"rank={rank} phase={policy['attack_phase']} "
            f"engage={policy['engage_distance']} "
            f"success={policy['successes']}/{policy['trials']} "
            f"median_frames={policy['median_success_frames']}"
        )
    if summary["promoted_state"]:
        print(
            f"saved={summary['promoted_state']} "
            f"provenance={summary['provenance']}"
        )
    return 0 if summary["successes"] else 1

if __name__ == "__main__":
    raise SystemExit(main())
