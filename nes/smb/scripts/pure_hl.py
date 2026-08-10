"""CLI for pure HappyLee track (track 3) — isolated from hybrid / natural / skills.

```bash
# Track status + isolation rules
uv run python -m smb.scripts.pure_hl status

# Gate 1: pure chain → 8-3 control (must pass before body work)
SDL_VIDEODRIVER=dummy SDL_AUDIODRIVER=dummy \\
  uv run python -m smb.scripts.pure_hl verify-to-83

# Diagnostic: continuous FM2 from 8-2 control through 8-3
uv run python -m smb.scripts.pure_hl probe-83

# Search pure FM2 for 8-3 leave (gated + multi-leave 8-2 + optional continuous)
uv run python -m smb.scripts.pure_hl search-83
uv run python -m smb.scripts.pure_hl search-83 --multi-leave --with-continuous
uv run python -m smb.scripts.pure_hl search-83 --with-continuous

# Export only after a hit (also done automatically on search hit)
# uv run python -m smb.scripts.pure_hl export-83 --si … --leave …

# 8-4 is hard-blocked until gate_8_3_leave.json verifies
uv run python -m smb.scripts.pure_hl check-8-4-gate
```

Writes **only** under ``models/pure_hl/`` and ``recordings/tas_import/pure_hl/``.
Never touches natural_82, hybrid, stitchless, or flamexx seeds.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

from smb.tas import pure_hl as ph
from smb.tas.slice import HL_8_1_FM2_START, HL_8_2_FM2_START


def _print(obj: Any) -> None:
    print(json.dumps(obj, indent=2, default=str), flush=True)


def cmd_status(_: argparse.Namespace) -> int:
    _print(ph.track_status())
    return 0


def cmd_verify_to_83(args: argparse.Namespace) -> int:
    report = ph.verify_to_8_3_control(
        start_8_1=args.si_8_1,
        start_8_2=args.si_8_2,
        write_evidence=not args.no_write,
    )
    _print(report)
    return 0 if report.get("success") else 1


def cmd_probe_83(args: argparse.Namespace) -> int:
    report = ph.probe_8_3_continuous(
        start_8_1=args.si_8_1,
        si_8_2=args.si_8_2,
        max_play=args.max_play,
        write_evidence=not args.no_write,
    )
    _print(report)
    # Success here means we *ran*; leave is reported separately.
    if report.get("error"):
        return 1
    if report.get("success_leave_83"):
        print(
            "\n[pure_hl] continuous 8-3 LEAVE — export candidate; still verify gated gate",
            file=sys.stderr,
            flush=True,
        )
        return 0
    print(
        "\n[pure_hl] no 8-3 leave yet — do NOT start pure 8-4",
        file=sys.stderr,
        flush=True,
    )
    return 2


def cmd_search_83(args: argparse.Namespace) -> int:
    leads = tuple(int(x) for x in args.leads.split(",") if x.strip() != "")
    cont_min = cont_max = None
    if args.with_continuous:
        cont_min = args.cont_si82_min
        cont_max = args.cont_si82_max

    def progress(msg: str) -> None:
        print(msg, flush=True)

    report = ph.search_pure_8_3(
        start_8_1=args.si_8_1,
        start_8_2=args.si_8_2,
        si_min=args.si_min,
        si_max=args.si_max,
        si_step=args.si_step,
        lead_idles=leads,
        max_play=args.max_play,
        multi_leave=args.multi_leave,
        si82_min=args.si82_min,
        si82_max=args.si82_max,
        si82_step=args.si82_step,
        top_leaves=args.top_leaves,
        cont_si82_min=cont_min,
        cont_si82_max=cont_max,
        cont_si82_step=args.cont_si82_step,
        cont_max_play=args.cont_max_play,
        progress=progress,
        write_evidence=not args.no_write,
        export_on_hit=not args.no_export,
        stop_on_hit=not args.no_stop_on_hit,
    )
    # Compact summary to stdout (full report on disk)
    summary = {
        "success": report.get("success"),
        "multi_leave": report.get("multi_leave"),
        "n_unique_leave_classes": len(report.get("unique_leave_classes") or []),
        "n_fan_classes": len(report.get("fan_classes") or []),
        "n_gated_hits": report.get("n_gated_hits"),
        "n_gated_trials": report.get("n_gated_trials"),
        "gated_best_progress": report.get("gated_best_progress"),
        "cont_best": {
            k: (report.get("cont_best") or {}).get(k)
            for k in ("si", "left_83", "max_x_83", "death", "leave83", "enter84")
        }
        if report.get("cont_best")
        else None,
        "exported": report.get("exported"),
        "gate_written": report.get("gate_written"),
        "control_8_3_fp": report.get("control_8_3_fp"),
        "wait_8_3": report.get("wait_8_3"),
        "leave_8_2": report.get("leave_8_2"),
        "si_8_2_used": report.get("si_8_2_used"),
        "next": report.get("next"),
        "error": report.get("error"),
        "evidence": str(ph.PURE_HL_EVIDENCE / "search_8_3.json"),
    }
    _print(summary)
    return 0 if report.get("success") else 2


def cmd_export_83(args: argparse.Namespace) -> int:
    if args.si is None or args.leave is None:
        print("export-83 requires --si and --leave", file=sys.stderr)
        return 2
    out = ph.export_pure_8_3(
        start_idx=args.si,
        leave_frames=args.leave,
        lead_idle=args.lead,
        start_8_2=args.si_8_2,
        start_8_1=args.si_8_1,
        verify=not args.no_verify,
        verify_trials=args.verify_trials,
    )
    _print(out)
    return 0 if out.get("success") else 1


def cmd_check_8_4(_: argparse.Namespace) -> int:
    r = ph.refuse_8_4_until_gate()
    _print(r)
    return 0 if r["allowed"] else 3


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description="Pure HappyLee track (no hybrid / natural / skills)"
    )
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("status", help="track isolation + gate status")
    s.set_defaults(func=cmd_status)

    s = sub.add_parser("verify-to-83", help="pure HL chain → 8-3 control")
    s.add_argument("--si-8-1", type=int, default=HL_8_1_FM2_START)
    s.add_argument("--si-8-2", type=int, default=HL_8_2_FM2_START)
    s.add_argument("--no-write", action="store_true")
    s.set_defaults(func=cmd_verify_to_83)

    s = sub.add_parser(
        "probe-83",
        help="continuous pure FM2 from 8-2 control (diagnostic)",
    )
    s.add_argument("--si-8-1", type=int, default=HL_8_1_FM2_START)
    s.add_argument("--si-8-2", type=int, default=HL_8_2_FM2_START)
    s.add_argument("--max-play", type=int, default=6000)
    s.add_argument("--no-write", action="store_true")
    s.set_defaults(func=cmd_probe_83)

    s = sub.add_parser("search-83", help="grid pure FM2 for 8-3 leave")
    s.add_argument("--si-8-1", type=int, default=HL_8_1_FM2_START)
    s.add_argument("--si-8-2", type=int, default=HL_8_2_FM2_START)
    s.add_argument("--si-min", type=int, default=12950)
    s.add_argument("--si-max", type=int, default=13650)
    s.add_argument("--si-step", type=int, default=1)
    s.add_argument("--leads", type=str, default="0,1,2")
    s.add_argument(
        "--max-play",
        type=int,
        default=1500,
        help="gated probe cap (pure 8-3 dies early when desynced; 1500 is enough)",
    )
    s.add_argument(
        "--multi-leave",
        action="store_true",
        help="fan multi leave-8-2 phase classes (leave82/timer) then re-gate SI search",
    )
    s.add_argument("--si82-min", type=int, default=10840)
    s.add_argument("--si82-max", type=int, default=10940)
    s.add_argument("--si82-step", type=int, default=2)
    s.add_argument(
        "--top-leaves",
        type=int,
        default=5,
        help="how many unique leave82/timer classes to fan (plus diversity)",
    )
    s.add_argument(
        "--with-continuous",
        action="store_true",
        help="also scan continuous FM2 from nearby 8-2 starts",
    )
    s.add_argument("--cont-si82-min", type=int, default=10880)
    s.add_argument("--cont-si82-max", type=int, default=10940)
    s.add_argument("--cont-si82-step", type=int, default=2)
    s.add_argument("--cont-max-play", type=int, default=5500)
    s.add_argument(
        "--no-stop-on-hit",
        action="store_true",
        help="continue full grid even after first leave (slower)",
    )
    s.add_argument("--no-write", action="store_true")
    s.add_argument("--no-export", action="store_true")
    s.set_defaults(func=cmd_search_83)

    s = sub.add_parser("export-83", help="write pure 8-3 seed + gate (after hit)")
    s.add_argument("--si", type=int, required=False)
    s.add_argument("--leave", type=int, required=False)
    s.add_argument("--lead", type=int, default=0)
    s.add_argument("--si-8-1", type=int, default=HL_8_1_FM2_START)
    s.add_argument("--si-8-2", type=int, default=HL_8_2_FM2_START)
    s.add_argument("--verify-trials", type=int, default=2)
    s.add_argument("--no-verify", action="store_true")
    s.set_defaults(func=cmd_export_83)

    s = sub.add_parser(
        "check-8-4-gate",
        help="hard block: pure 8-4 only after pure 8-3 leave",
    )
    s.set_defaults(func=cmd_check_8_4)

    args = p.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
