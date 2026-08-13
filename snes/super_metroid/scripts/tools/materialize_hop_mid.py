#!/usr/bin/env python3
"""Propose / materialize mid-pins from old guided_human recordings.

Two layers (both safe under the anti-desync rule):

1. **Offline propose** (no emulator) — parse hop ``trace`` for floor lands,
   combat poses, energy cliffs, pre-leave. Edit / re-record hints only.
2. **Lockstep materialize** — boot the hop's *live* enter anchor, step while
   matching ``trace``, dump a gzip state at last lockstep match (or an
   offline candidate that still matches). Not multi-minute full-tape invent.

```bash
# Offline candidates for Metroid Room 4 (hop 12)
uv run python snes/super_metroid/scripts/tools/materialize_hop_mid.py \\
  snes/super_metroid/tasks/g4_tourian_human.json --hop 12 --propose

# Lockstep scan (find last dual-matchable frame)
uv run python snes/super_metroid/scripts/tools/materialize_hop_mid.py \\
  snes/super_metroid/tasks/g4_tourian_human.json --hop 12 --scan

# Dump mid pin at last lockstep match + dual verify enter→mid
uv run python snes/super_metroid/scripts/tools/materialize_hop_mid.py \\
  snes/super_metroid/tasks/g4_tourian_human.json --hop 12 --materialize

# Dump a specific candidate index
uv run python snes/super_metroid/scripts/tools/materialize_hop_mid.py \\
  snes/super_metroid/tasks/g4_tourian_human.json --hop 12 \\
  --materialize --at 20222 --label floor
```
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
_SNES_IMPORT_ROOT = Path(__file__).resolve().parents[3]
for _p in (ROOT, _SNES_IMPORT_ROOT):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from super_metroid.human_tape import (  # noqa: E402
    load_room_hops,
    load_room_names,
    load_task_json,
    lockstep_scan,
    materialize_lockstep_mid,
    propose_trace_midpoints,
    resolve_hop_slice,
)
from super_metroid.dev.common import boot_from_state, make_dev_env  # noqa: E402


def _resolve_task(task: Path) -> Path:
    if task.is_file():
        return task
    alt = ROOT / task
    if alt.is_file():
        return alt
    return task


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("task", type=Path, help="guided_human task JSON")
    p.add_argument("--hop", type=int, default=None, help="Room hop index")
    p.add_argument("--from-frame", type=int, default=None)
    p.add_argument("--to-frame", type=int, default=None)
    p.add_argument(
        "--propose",
        action="store_true",
        help="Offline midpoint candidates from trace (no emulator)",
    )
    p.add_argument(
        "--scan",
        action="store_true",
        help="Emulator lockstep scan: last_match / first_mismatch",
    )
    p.add_argument(
        "--materialize",
        action="store_true",
        help="Dump gzip mid pin at last match (or --at) + dual verify",
    )
    p.add_argument(
        "--at",
        type=int,
        default=None,
        dest="target_index",
        help="Trace index to dump (must be ≤ last lockstep match)",
    )
    p.add_argument(
        "--label",
        type=str,
        default="mid",
        help="Filename slug for dumped state (default mid)",
    )
    p.add_argument("--xy-tol", type=int, default=12, help="Lockstep xy band")
    p.add_argument(
        "--pose-strict",
        action="store_true",
        help="Require pose match during lockstep (stricter)",
    )
    p.add_argument(
        "--no-assist",
        action="store_true",
        help="Disable contract assist during lockstep/materialize",
    )
    p.add_argument(
        "--boot-settle",
        type=int,
        default=0,
        help="Idle after boot_from_state (default 0)",
    )
    p.add_argument(
        "--no-dual",
        action="store_true",
        help="Skip dual-verify enter→mid after dump",
    )
    p.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Write JSON report (propose/scan/materialize summary)",
    )
    p.add_argument(
        "--all-hops",
        action="store_true",
        help="With --propose: emit candidates for every hop",
    )
    return p


def _print_candidates(cands: list[dict], *, hop: int | None) -> None:
    prefix = f"hop={hop} " if hop is not None else ""
    print(f"{prefix}midpoint candidates: {len(cands)}")
    for c in cands:
        print(
            f"  i={c['index']:6d} f{c['frame']:6d}  {c['kind']:12s}  "
            f"{c['room']} xy={c['xy']} pose={c['pose']}  {c.get('note', '')}"
        )


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    task = _resolve_task(args.task)
    if not task.is_file():
        print(f"ERROR: missing task {args.task}", file=sys.stderr)
        return 2

    if not (args.propose or args.scan or args.materialize):
        # Default: propose offline
        args.propose = True

    data = load_task_json(task)
    frames = data.get("frames") or []
    trace = list(data.get("trace") or [])
    hops = load_room_hops(task_data=data, room_names=load_room_names(), settle=True)
    report: dict = {
        "task": str(task),
        "name": data.get("name") or task.stem,
    }

    if args.propose and args.all_hops:
        all_rows = []
        for h in hops:
            cands = propose_trace_midpoints(
                trace,
                int(h["start_index"]),
                int(h["end_index"]),
                end_xy=h.get("end_xy"),
            )
            all_rows.append(
                {
                    "hop": h["index"],
                    "room": h["room"],
                    "name": h.get("name"),
                    "dwell": h.get("dwell"),
                    "candidates": cands,
                }
            )
            print(
                f"[{h['index']:02d}] {h['room']} {h.get('name', '?')} "
                f"dwell={h.get('dwell')}  mids={len(cands)}"
            )
            for c in cands[:6]:
                print(
                    f"    i={c['index']} {c['kind']:12s} xy={c['xy']} "
                    f"pose={c['pose']}"
                )
            if len(cands) > 6:
                print(f"    … {len(cands) - 6} more")
        report["hops"] = all_rows
        if args.out:
            out = args.out if args.out.is_absolute() else ROOT / args.out
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
            print(f"wrote {out}")
        return 0

    if args.hop is None and args.from_frame is None:
        print("ERROR: need --hop or --from-frame (or --all-hops with --propose)",
              file=sys.stderr)
        return 2

    slice_info = resolve_hop_slice(
        task,
        hop_index=args.hop,
        from_frame=args.from_frame,
        to_frame=args.to_frame,
        leave_extra=0,
        task_data=data,
    )
    report["slice"] = {
        k: slice_info.get(k)
        for k in (
            "hop_index",
            "start_index",
            "end_index",
            "replay_start",
            "start_room_hex",
            "leave_room_hex",
            "end_xy",
            "anchor_path",
            "anchor_frame",
            "steps",
        )
    }
    print(
        f"hop={slice_info.get('hop_index')} "
        f"{slice_info.get('start_room_hex')}→{slice_info.get('leave_room_hex')} "
        f"idx {slice_info.get('start_index')}..{slice_info.get('end_index')} "
        f"replay_start={slice_info.get('replay_start')} "
        f"anchor={Path(str(slice_info.get('anchor_path') or '')).name}"
    )

    start_i = int(slice_info["start_index"])
    end_i = int(slice_info["end_index"])
    if args.propose:
        cands = propose_trace_midpoints(
            trace,
            start_i,
            end_i,
            end_xy=slice_info.get("end_xy"),
        )
        _print_candidates(cands, hop=slice_info.get("hop_index"))
        report["candidates"] = cands

    if args.scan or args.materialize:
        ap = slice_info.get("anchor_path")
        if not ap or not Path(str(ap)).is_file():
            print("ERROR: no live enter anchor — cannot lockstep/materialize",
                  file=sys.stderr)
            print("  (offline --propose only for tapes without anchors)",
                  file=sys.stderr)
            return 1

    if args.scan and not args.materialize:
        env = make_dev_env()
        try:
            boot_from_state(
                env, Path(str(slice_info["anchor_path"])),
                settle_frames=args.boot_settle,
            )
            scan = lockstep_scan(
                env,
                frames,
                trace,
                int(slice_info["replay_start"]),
                end_i,
                xy_tol=args.xy_tol,
                pose_strict=args.pose_strict,
                sample_every=max(1, (end_i - int(slice_info["replay_start"])) // 50 or 1),
                assist=not args.no_assist,
            )
        finally:
            close = getattr(env, "close", None)
            if callable(close):
                close()
        # drop non-json blob / env state handles
        _drop = {"dumped_state", "last_ok_blob", "last_ok_st"}
        scan_out = {k: v for k, v in scan.items() if k not in _drop}
        report["scan"] = scan_out
        cont = scan.get("contiguous_last_match")
        print(
            f"lockstep contiguous_last={cont}  "
            f"last_match={scan.get('last_match')}  "
            f"first_mismatch={scan.get('first_mismatch')}"
        )
        if cont is not None:
            lm = int(cont)
            row = trace[lm] if lm < len(trace) else {}
            print(
                f"  contiguous row: room={row.get('room_hex')} "
                f"xy=[{row.get('x')},{row.get('y')}] pose={row.get('pose')}"
            )

    if args.materialize:
        result = materialize_lockstep_mid(
            task,
            hop_index=args.hop,
            from_frame=args.from_frame,
            to_frame=args.to_frame,
            target_index=args.target_index,
            xy_tol=args.xy_tol,
            pose_strict=args.pose_strict,
            boot_settle=args.boot_settle,
            leave_extra=0,
            assist=not args.no_assist,
            dual_verify=not args.no_dual,
            label=args.label,
        )
        # strip dual detail bulk for console
        mid = result.get("mid") or {}
        print(
            f"{'OK' if result.get('ok') else 'FAIL'} materialize  "
            f"dump_index={result.get('dump_index')}  "
            f"last_match={result.get('last_match')}  "
            f"dual_verify={result.get('dual_verify')}  "
            f"path={result.get('state_path')}"
        )
        if mid:
            print(
                f"  pin f{mid.get('frame')} {mid.get('room')} "
                f"xy={mid.get('xy')} pose={mid.get('pose')} kind={mid.get('kind')}"
            )
        if result.get("first_mismatch"):
            print(f"  first_mismatch: {result['first_mismatch']}")
        if not result.get("ok") and result.get("reason"):
            print(f"  reason: {result['reason']}")
        report["materialize"] = {
            k: v
            for k, v in result.items()
            if k not in ("dual_detail",)
        }
        # dual_detail is json-safe
        if result.get("dual_detail") is not None:
            report["materialize"]["dual_detail"] = result["dual_detail"]

    if args.out:
        out = args.out if args.out.is_absolute() else ROOT / args.out
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(report, indent=2, default=str) + "\n", encoding="utf-8")
        print(f"wrote {out}")

    if args.materialize:
        return 0 if report.get("materialize", {}).get("ok") else 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
