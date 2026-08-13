#!/usr/bin/env python3
"""Compile/verify a reactive room skill from one live-anchor expert hop.

This is the planned/genetic-policy entrypoint, not frame hill-climb. It turns a
human/pure hop into sparse kinematic action spans, supports multiple takes and
equipment variants in one room policy, then requires dual-green replay before
promotion.

Examples::

  # Early Climb ascent (Morph, no Hi-Jump)
  uv run python snes/super_metroid/scripts/tools/optimize_room_policy.py \
    --body snes/super_metroid/tasks/full_start_v1_hops/hop_09_Climb.json \
    --room 0x96BA --from-room 0x975C --exit-room 0x92FD --variant base

  # Same room with late-game / Hi-Jump physics
  uv run python snes/super_metroid/scripts/tools/optimize_room_policy.py \
    --task snes/super_metroid/tasks/g4_tourian_human_mb.json --hop 5 \
    --room 0x96BA --from-room 0x975C --exit-room 0x92FD --variant hi_jump
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

ROOT = Path(__file__).resolve().parents[4]

from super_metroid.paths import GAME_DIR  # noqa: E402
from super_metroid.reactive_policy import ReactiveRoomPolicy  # noqa: E402
from super_metroid.room_policy_tools import (  # noqa: E402
    capture_reference_trajectory,
    load_button_frames,
    mark_takeovers_verified,
    mark_verified,
    merge_policy_variant,
    task_hop_frames,
    verify_reactive_policy,
    verify_takeover_sweep,
)

DEFAULT_POLICY_DIR = GAME_DIR / "policies" / "reactive_rooms"


def _integer(value: str) -> int:
    return int(value, 0)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--body", type=Path, help="hop body JSON (frames/raw_buttons)")
    source.add_argument("--task", type=Path, help="guided_human task JSON")
    parser.add_argument("--hop", type=int, help="settled hop index with --task")
    parser.add_argument("--anchor", type=Path, help="override live entry .state")
    parser.add_argument("--room", type=_integer, required=True)
    parser.add_argument("--from-room", type=_integer)
    parser.add_argument("--exit-room", type=_integer, required=True)
    parser.add_argument("--variant", default="base", help="base / hi_jump / custom")
    parser.add_argument("--trajectory-id", help="take id (default: source stem)")
    parser.add_argument("--required-items", type=_integer)
    parser.add_argument("--forbidden-items", type=_integer)
    parser.add_argument("--route", default="kpdr")
    parser.add_argument("--policy-id")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--transition-tail", type=int, default=180)
    parser.add_argument(
        "--max-span",
        type=int,
        default=8,
        help="max cached action span; feedback runs at least this often (default 8)",
    )
    parser.add_argument("--max-frames", type=int, default=10_000)
    parser.add_argument(
        "--min-fps",
        type=float,
        default=300.0,
        help="headless verification throughput gate (default 300; 0 disables)",
    )
    parser.add_argument("--no-assist", action="store_true")
    parser.add_argument("--no-verify", action="store_true")
    parser.add_argument(
        "--adapter",
        action="store_true",
        help="also use short-horizon adapter during verification",
    )
    parser.add_argument(
        "--takeover-sweep",
        action="store_true",
        help="verify 25/50/75%% human handoffs plus perturbed timing",
    )
    parser.add_argument(
        "--takeover-perturb",
        type=int,
        default=4,
        help="idle frames added at takeover to change live kinematics (default 4)",
    )
    parser.add_argument("--promote-bank", action="store_true")
    parser.add_argument("--bank", type=Path)
    parser.add_argument("--hop-key")
    return parser


def _resolve_source(args: argparse.Namespace) -> tuple[list[list[int]], dict, Path]:
    if args.body is not None:
        frames, meta = load_button_frames(args.body)
    else:
        if args.hop is None:
            raise ValueError("--hop is required with --task")
        frames, meta = task_hop_frames(
            args.task,
            args.hop,
            transition_tail=args.transition_tail,
        )
    anchor = args.anchor
    if anchor is None and meta.get("entry_anchor"):
        anchor = Path(str(meta["entry_anchor"]))
    if anchor is None:
        raise ValueError("no entry anchor in source metadata; pass --anchor")
    return frames, meta, anchor


def _promote_bank(
    args: argparse.Namespace,
    policy: ReactiveRoomPolicy,
    report: dict,
    *,
    anchor: Path,
    policy_path: Path,
    items: int,
) -> str:
    from super_metroid.skill_bank import (
        DEFAULT_BANK_PATH,
        HopSkillRecord,
        SkillBank,
        make_hop_key,
    )

    bank_path = args.bank or DEFAULT_BANK_PATH
    bank = SkillBank.load(bank_path) if bank_path.is_file() else SkillBank()
    hop_key = args.hop_key or make_hop_key(
        args.room,
        from_room_id=args.from_room,
        to_room_id=args.exit_room,
        items=items,
    )
    frames = max(int(row.get("frames", 0)) for row in report["runs"])
    current = bank.best(hop_key, require_dual_green=True)
    if current is not None and current.frames <= frames:
        return f"bank kept PB {current.frames}f <= reactive {frames}f"
    record = HopSkillRecord(
        hop_key=hop_key,
        room_id=args.room,
        name=policy.policy_id,
        frames=frames,
        source=f"reactive:{policy.policy_id}:{args.variant}",
        entry_anchor=str(anchor),
        dual_green=True,
        assist=not args.no_assist,
        notes="genetic/planned reactive policy; live-anchor dual green",
        meta={
            "policy_path": str(policy_path),
            "policy_kind": "reactive_trajectory",
            "variant": args.variant,
            "route_id": args.route,
        },
    )
    bank.add(record)
    bank.save(bank_path)
    return f"bank PB ← {hop_key} {frames}f ({bank_path})"


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        frames, source_meta, anchor = _resolve_source(args)
        policy_id = args.policy_id or (
            f"room_{args.room:04x}_from_"
            f"{args.from_room:04x}_to_{args.exit_room:04x}"
            if args.from_room is not None
            else f"room_{args.room:04x}_to_{args.exit_room:04x}"
        )
        output = args.output or DEFAULT_POLICY_DIR / f"{policy_id}.json"
        existing = ReactiveRoomPolicy.load(output) if output.is_file() else None
        trajectory_id = args.trajectory_id or (
            f"{(args.body or args.task).stem}_{args.variant}"
        )
        trajectory, capture = capture_reference_trajectory(
            anchor,
            frames,
            trajectory_id=trajectory_id,
            room_id=args.room,
            exit_room_id=args.exit_room,
            assist=not args.no_assist,
            source={
                "input": str(args.body or args.task),
                **source_meta,
            },
            max_span_frames=args.max_span,
            exit_tail_frames=args.transition_tail,
        )
        if not capture["ok"]:
            print(
                f"RED expert seed did not leave 0x{args.room:04X} for "
                f"0x{args.exit_room:04X}: room={capture['room']} xy={capture['xy']}",
                file=sys.stderr,
            )
            return 1
        policy = merge_policy_variant(
            existing,
            policy_id=policy_id,
            route_id=args.route,
            room_id=args.room,
            from_room_id=args.from_room,
            exit_room_id=args.exit_room,
            variant_id=args.variant,
            trajectory=trajectory,
            required_items=args.required_items,
            forbidden_items=args.forbidden_items,
        )
        policy.save(output)
        ratio = capture["input_frames"] / max(1, capture["samples"])
        print(
            f"CAPTURE {args.variant} {capture['input_frames']}f → "
            f"{capture['samples']} spans ({ratio:.1f}x)  items=0x{capture['items']:04X} "
            f"hi_jump={capture['hi_jump']}"
        )
        print(f"  candidate → {output}")
        if args.no_verify:
            return 0

        report = verify_reactive_policy(
            policy,
            anchor,
            dual=True,
            max_frames=args.max_frames,
            assist=not args.no_assist,
            use_adapter=args.adapter,
        )
        runs = report.get("runs") or []
        min_measured_fps = min(
            (float(row.get("fps", 0.0)) for row in runs),
            default=0.0,
        )
        performance_green = args.min_fps <= 0 or min_measured_fps >= args.min_fps
        mark = "GREEN" if report.get("green") and performance_green else "RED"
        print(
            f"{mark} dual reactive {policy_id}/{args.variant}  "
            + " ".join(
                f"run{i}={row.get('frames')}f@{row.get('fps', 0):.0f}fps:"
                f"{row.get('room')}:{row.get('xy')}"
                for i, row in enumerate(runs)
            )
        )
        if not performance_green:
            print(
                f"RED throughput {min_measured_fps:.0f}fps < "
                f"required {args.min_fps:.0f}fps",
                file=sys.stderr,
            )
            return 1
        if not report.get("green"):
            return 1
        policy = mark_verified(policy, report)
        policy.save(output)
        print(f"  {policy.status} → {output}")
        if args.takeover_sweep:
            points = tuple(
                max(1, int(len(frames) * fraction))
                for fraction in (0.25, 0.5, 0.75)
            )
            takeover = verify_takeover_sweep(
                policy,
                anchor,
                frames,
                takeover_points=points,
                perturb_frames=args.takeover_perturb,
                max_frames=args.max_frames,
                assist=not args.no_assist,
                use_adapter=True,
            )
            takeover_performance = all(
                args.min_fps <= 0 or float(row.get("fps", 0.0)) >= args.min_fps
                for row in takeover["runs"]
            )
            mark = "GREEN" if takeover["ok"] and takeover_performance else "RED"
            print(
                f"{mark} takeover sweep  "
                + " ".join(
                    f"f{row['takeover_point']}+{row['perturb_frames']}→"
                    f"{row['autopilot_frames']}f@{row['fps']:.0f}fps:"
                    f"{row['room']}"
                    for row in takeover["runs"]
                )
            )
            if not takeover["ok"] or not takeover_performance:
                return 1
            policy = mark_takeovers_verified(policy, takeover)
            policy.save(output)
            print(f"  takeover_verified/{args.variant} → {output}")
        if args.promote_bank:
            print(
                "  "
                + _promote_bank(
                    args,
                    policy,
                    report,
                    anchor=anchor,
                    policy_path=output,
                    items=int(capture["items"]),
                )
            )
        return 0
    except Exception as exc:  # noqa: BLE001 - CLI error boundary
        print(f"RED {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
