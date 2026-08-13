#!/usr/bin/env python3
"""Compose multi-hop open-loop from live pins (pin → body → leave pin).

Product path for seam-safe replay of a guided_human take. Each hop boots its
own entry anchor so desync does not compound across rooms.

```bash
# All hops that have live anchors (dual green)
uv run python snes/super_metroid/scripts/tools/compose_human_hops.py \\
  snes/super_metroid/tasks/full_start_v1.json --dual

# Explicit hop list
uv run python snes/super_metroid/scripts/tools/compose_human_hops.py \\
  snes/super_metroid/tasks/full_start_v1.json --hops 0,1,2 --dual

# Dry: list planned hops only
uv run python snes/super_metroid/scripts/tools/compose_human_hops.py \\
  snes/super_metroid/tasks/full_start_v1.json --dry
```

Also: ``./play --compose full_start_v1``
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

ROOT = Path(__file__).resolve().parents[4]

from super_metroid.human_tape.compose import compose_hops  # noqa: E402
from super_metroid.human_tape.hops import (  # noqa: E402
    load_room_hops,
    load_task_json,
    resolve_hop_slice,
)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("task", type=Path, help="guided_human task JSON")
    p.add_argument(
        "--hops",
        type=str,
        default=None,
        help="Comma-separated hop indices (default: all with anchors)",
    )
    p.add_argument(
        "--all-hops",
        action="store_true",
        help="Include hops without anchors (will RED without --anchor per hop)",
    )
    p.add_argument("--dual", action="store_true", help="Dual-green each hop")
    p.add_argument("--xy-tol", type=int, default=24)
    p.add_argument("--no-assist", action="store_true")
    p.add_argument(
        "--continue-on-red",
        action="store_true",
        help="Do not stop chain on first RED hop",
    )
    p.add_argument(
        "--dry",
        action="store_true",
        help="Print planned hops + anchors only (no emulator)",
    )
    p.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Write compose report JSON",
    )
    p.add_argument(
        "--promote-bank",
        action="store_true",
        help="On each GREEN hop, set dual_green in skill bank",
    )
    p.add_argument(
        "--bank",
        type=Path,
        default=None,
        help="Skill bank path (default recordings/skill_bank/bank.json)",
    )
    return p


def _parse_hops(raw: str | None) -> list[int] | None:
    if raw is None or not str(raw).strip():
        return None
    return [int(x.strip()) for x in str(raw).split(",") if x.strip()]


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    task = args.task
    if not task.is_file():
        alt = ROOT / task
        if alt.is_file():
            task = alt
        else:
            print(f"RED  task not found: {args.task}", file=sys.stderr)
            return 2

    hop_indices = _parse_hops(args.hops)

    if args.dry:
        data = load_task_json(task)
        hops = load_room_hops(task_data=data, settle=True)
        print(f"task={data.get('name') or task.stem}  hops={len(hops)}  (settled)")
        want = set(hop_indices) if hop_indices is not None else None
        for h in hops:
            idx = int(h["index"])
            if want is not None and idx not in want:
                continue
            info = resolve_hop_slice(
                task, hop_index=idx, leave_extra=1, task_data=data, settle=True
            )
            ap = info.get("anchor_path")
            mark = "anchor" if ap else "NO_ANCHOR"
            print(
                f"  [{idx:02d}] {h.get('room')} {h.get('name')}  "
                f"f{h.get('start_index')}-{h.get('end_index')}  {mark}  "
                f"{Path(str(ap)).name if ap else ''}"
            )
        return 0

    report = compose_hops(
        task,
        hop_indices=hop_indices,
        dual=args.dual,
        xy_tol=args.xy_tol,
        assist=not args.no_assist,
        stop_on_red=not args.continue_on_red,
        require_anchor=not args.all_hops,
    )

    mark = "GREEN" if report.green else "RED"
    print(
        f"{mark}  compose  {report.task}  "
        f"green={report.hops_green}/{report.hops_run} planned={report.hops_planned}"
    )
    for r in report.results:
        m = "OK" if r.green else "FAIL"
        print(
            f"  [{r.hop_index:02d}] {m}  {r.room}→{r.leave_room}  "
            f"steps={r.steps}  {Path(str(r.anchor_path or '')).name}  "
            f"{r.reason or ''}"
        )
        if r.green and args.promote_bank:
            from super_metroid.skill_bank import promote_from_hop_replay

            rec = promote_from_hop_replay(
                r.report or {"green": True, "ok": True},
                bank_path=args.bank,
                source=report.task,
            )
            if rec is not None:
                print(f"       bank dual_green ← {rec.hop_key}")
    for note in report.notes:
        print(f"  · {note}")

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(
            json.dumps(report.to_dict(), indent=2) + "\n", encoding="utf-8"
        )
        print(f"  wrote {args.out}")

    return 0 if report.green else 1


if __name__ == "__main__":
    raise SystemExit(main())
