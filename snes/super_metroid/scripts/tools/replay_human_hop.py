#!/usr/bin/env python3
"""Replay one guided_human hop from a live gzip anchor (open-loop unit).

```bash
# Escape room 1 of G4/MB human take — dual green smoke
uv run python snes/super_metroid/scripts/tools/replay_human_hop.py \\
  snes/super_metroid/tasks/g4_tourian_human_mb.json \\
  --hop 1 --dual

# Explicit frame window + anchor
uv run python snes/super_metroid/scripts/tools/replay_human_hop.py \\
  snes/super_metroid/tasks/g4_tourian_human_mb.json \\
  --from-frame 10924 --frames 400 --anchor path.state

# Room id selection
uv run python snes/super_metroid/scripts/tools/replay_human_hop.py \\
  snes/super_metroid/tasks/g4_tourian_human_mb.json \\
  --room 0xDE4D --to-room 0xDE7A --dual
```

Exit 0 on GREEN (both runs if --dual). Prints a short GREEN/RED summary.
Multi-hop: ``compose_human_hops.py`` (re-pins each hop).
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

ROOT = Path(__file__).resolve().parents[4]

from super_metroid.human_tape import (  # noqa: E402
    load_anchors_index,
    resolve_hop_slice,
    run_hop_replay,
)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "task",
        type=Path,
        help="guided_human task JSON (frames + trace)",
    )
    p.add_argument(
        "--hop",
        type=int,
        default=None,
        help="Room hop index (settled hops; same as materialize bodies)",
    )
    p.add_argument(
        "--from-frame",
        type=int,
        default=None,
        help="First frame index to step (overrides hop replay_start)",
    )
    p.add_argument(
        "--to-frame",
        type=int,
        default=None,
        help="Last frame index to step (inclusive)",
    )
    p.add_argument(
        "--frames",
        type=int,
        default=None,
        dest="frames_count",
        help="Number of frames to step from --from-frame / replay_start",
    )
    p.add_argument(
        "--room",
        type=str,
        default=None,
        help="Start room id (e.g. 0xDE4D) — first matching hop",
    )
    p.add_argument(
        "--to-room",
        type=str,
        default=None,
        help="Expected leave room id (e.g. 0xDE7A)",
    )
    p.add_argument(
        "--anchor",
        type=Path,
        default=None,
        help="Explicit gzip .state (default: match from *_anchors.json)",
    )
    p.add_argument(
        "--dual",
        action="store_true",
        help="Run twice; both must be green",
    )
    p.add_argument(
        "--xy-tol",
        type=int,
        default=24,
        help="End xy band tolerance (default 24)",
    )
    p.add_argument(
        "--leave-extra",
        type=int,
        default=1,
        help="Extra frames past hop end_index to observe room leave (default 1)",
    )
    p.add_argument(
        "--settle",
        type=int,
        default=0,
        help="Idle frames after the slice",
    )
    p.add_argument(
        "--boot-settle",
        type=int,
        default=0,
        help=(
            "Idle frames after boot_from_state (default 0 — live room_enter "
            "anchors are already settled; extra idle desyncs long hops)"
        ),
    )
    p.add_argument(
        "--dry-resolve",
        action="store_true",
        help="Only resolve hop/anchor (no emulator); exit 0 if anchor found",
    )
    p.add_argument(
        "--list-hops",
        action="store_true",
        help="Print room hops and exit",
    )
    p.add_argument(
        "--no-assist",
        action="store_true",
        help=(
            "Disable UnlimitedResourcesAssist (clean-track stress). "
            "Default is assist ON — matches guided_human record contract."
        ),
    )
    p.add_argument(
        "--promote-bank",
        action="store_true",
        help="On GREEN, set dual_green=True on matching skill bank record",
    )
    p.add_argument(
        "--bank",
        type=Path,
        default=None,
        help="Skill bank path (default recordings/skill_bank/bank.json)",
    )
    return p


def _print_hops(task: Path) -> int:
    from super_metroid.human_tape.hops import load_room_hops, load_task_json

    data = load_task_json(task)
    hops = load_room_hops(task_data=data, settle=True)
    print(f"task={data.get('name') or task.stem}  hops={len(hops)}  "
          f"frames={data.get('frame_count')}  (settled)")
    for h in hops:
        print(
            f"  [{h['index']:02d}] f{h['start_index']:5d}-{h['end_index']:5d} "
            f"({h['dwell']:4d}f) {h['room']} {h.get('name', '?')} "
            f"end_xy={h.get('end_xy')}"
        )
    anchors = load_anchors_index(task)
    if anchors:
        print(f"anchors: {anchors.get('count')}  dir={anchors.get('anchors_dir')}")
        for a in (anchors.get("anchors") or [])[:12]:
            print(
                f"  f{int(a.get('frame', 0)):6d} {a.get('kind'):12s} "
                f"{a.get('room')}  {Path(str(a.get('path', ''))).name}"
            )
        n = len(anchors.get("anchors") or [])
        if n > 12:
            print(f"  … {n - 12} more")
    return 0


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    task = args.task
    if not task.is_file():
        # Relative path from cwd may miss gitignored tasks; try repo root.
        alt = ROOT / task
        if alt.is_file():
            task = alt
        else:
            print(f"RED  task not found: {args.task}", file=sys.stderr)
            return 2

    if args.list_hops:
        return _print_hops(task)

    if args.hop is None and args.room is None and args.from_frame is None:
        print(
            "RED  need --hop, --room, or --from-frame",
            file=sys.stderr,
        )
        return 2

    if args.dry_resolve:
        info = resolve_hop_slice(
            task,
            hop_index=args.hop,
            from_frame=args.from_frame,
            to_frame=args.to_frame,
            frames_count=args.frames_count,
            room=args.room,
            to_room=args.to_room,
            leave_extra=args.leave_extra,
        )
        if args.anchor:
            info["anchor_path"] = str(args.anchor)
            info["anchor_frame"] = info.get("anchor_frame")
        print(
            f"resolve hop={info.get('hop_index')} "
            f"start={info.get('start_room_hex')} → leave={info.get('leave_room_hex')} "
            f"idx {info.get('start_index')}..{info.get('end_index')} "
            f"replay_start={info.get('replay_start')} steps={info.get('steps')}"
        )
        print(
            f"  end_xy={info.get('end_xy')}  "
            f"anchor_frame={info.get('anchor_frame')}  "
            f"anchor={info.get('anchor_path')}"
        )
        ok = bool(info.get("anchor_path") or args.anchor)
        print("GREEN resolve" if ok else "RED resolve (no anchor)")
        return 0 if ok else 1

    try:
        report = run_hop_replay(
            task,
            hop_index=args.hop,
            from_frame=args.from_frame,
            to_frame=args.to_frame,
            frames_count=args.frames_count,
            room=args.room,
            to_room=args.to_room,
            anchor_path=args.anchor,
            dual=args.dual,
            xy_tol=args.xy_tol,
            settle_frames=args.settle,
            boot_settle=args.boot_settle,
            leave_extra=args.leave_extra,
            assist=not args.no_assist,
        )
    except Exception as exc:  # noqa: BLE001 — CLI surface
        print(f"RED  error: {exc}", file=sys.stderr)
        return 2

    sl = report.get("slice") or {}
    mark = "GREEN" if report.get("green") else "RED"
    dual_s = " dual" if report.get("dual") else ""
    assist_s = " assist=OFF" if args.no_assist else ""
    print(
        f"{mark}{dual_s}{assist_s}  hop={sl.get('hop_index')} "
        f"{sl.get('start_room_hex')}→{sl.get('leave_room_hex')}  "
        f"replay f{report.get('replay_start')}..{report.get('replay_end')}  "
        f"anchor={Path(str(report.get('anchor_path') or '')).name}"
    )
    check = report.get("check") or {}
    if report.get("dual") and check.get("runs"):
        for i, c in enumerate(check["runs"]):
            r = (report.get("runs") or [None, None])[i] or {}
            print(
                f"  run{i}: {'OK' if c.get('ok') else 'FAIL'}  "
                f"room={c.get('got_room_hex')} xy={c.get('got_xy')}  "
                f"pose={r.get('pose')} phase={r.get('phase')}  "
                f"{c.get('reason') or ''}"
            )
    else:
        r0 = (report.get("runs") or [{}])[0]
        print(
            f"  room={check.get('got_room_hex') or r0.get('room')} "
            f"xy={check.get('got_xy') or r0.get('xy')}  "
            f"pose={r0.get('pose')} phase={r0.get('phase')}  "
            f"tol={args.xy_tol}  {check.get('reason') or ''}"
        )
    if not report.get("green") and report.get("reason"):
        print(f"  reason: {report['reason']}")

    if report.get("green") and args.promote_bank:
        from super_metroid.skill_bank import promote_from_hop_replay

        rec = promote_from_hop_replay(
            report,
            bank_path=args.bank,
            source=str(sl.get("name") or task.stem),
        )
        if rec is not None:
            print(
                f"  bank dual_green ← {rec.hop_key}  frames={rec.frames}  "
                f"source={rec.source}"
            )
        else:
            print("  bank promote skipped (no record)", flush=True)

    return 0 if report.get("green") else 1


if __name__ == "__main__":
    raise SystemExit(main())
