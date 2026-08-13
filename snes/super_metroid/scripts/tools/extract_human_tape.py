#!/usr/bin/env python3
"""Offline hop inventory + end-pin verify for guided_human tapes.

Offline only (no emulator). For open-loop use ``replay_human_hop.py`` (one
hop from a live pin) or ``compose_human_hops.py`` (pin→body multi-hop chain).
Uses the per-frame ``trace`` + optional live ``*_anchors.json`` dumps from
recording.

```bash
# Summary: hops + skill groups + end fingerprint check
uv run python snes/super_metroid/scripts/tools/extract_human_tape.py \\
  snes/super_metroid/tasks/maridia_grapple_human.json --summary

# Write extract JSON (hops / skills / anchors index pointer)
uv run python snes/super_metroid/scripts/tools/extract_human_tape.py \\
  snes/super_metroid/tasks/maridia_grapple_human.json \\
  --out snes/super_metroid/tasks/maridia_grapple_human_extract.json

# List live anchors from a take that recorded with anchors ON
uv run python snes/super_metroid/scripts/tools/extract_human_tape.py \\
  snes/super_metroid/tasks/some_take.json --list-anchors
```
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
_SNES_IMPORT_ROOT = Path(__file__).resolve().parents[3]
for _p in (ROOT, _SNES_IMPORT_ROOT):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from super_metroid.human_tape import default_skill_groups, extract_tape  # noqa: E402
from super_metroid.materialize import materialize_take  # noqa: E402
from super_metroid.skill_bank import DEFAULT_BANK_PATH  # noqa: E402


def _print_summary(board: dict) -> None:
    print(f"task: {board.get('name')}  frames={board.get('frame_count')}")
    print(f"  start: {board.get('start_state')}")
    print(f"  recorded_at: {board.get('recorded_at')}")
    end_fp = board.get("end_fingerprint")
    if end_fp:
        print(
            f"  end_fp: {end_fp.get('room')} xy={end_fp.get('xy')} "
            f"items={end_fp.get('items')} grapple={end_fp.get('grapple')}"
        )
    else:
        print("  end_fp: (missing — re-record with current guided_human)")
    verify = board.get("end_verify")
    if verify is not None:
        mark = "OK" if verify.get("ok") else "MISMATCH"
        print(f"  end vs trace: {mark}  {verify}")
    hops = board.get("room_hops") or []
    print(f"  room hops: {len(hops)}")
    for h in hops[:8]:
        print(
            f"    [{h['index']:02d}] f{h['frame']:5d}-{h['end_frame']:5d} "
            f"({h['dwell']:4d}f) {h['room']} {h.get('name', '?')}"
        )
    if len(hops) > 8:
        print(f"    … {len(hops) - 8} more")
        for h in hops[-3:]:
            print(
                f"    [{h['index']:02d}] f{h['frame']:5d}-{h['end_frame']:5d} "
                f"({h['dwell']:4d}f) {h['room']} {h.get('name', '?')}"
            )
    skills = board.get("skill_groups") or []
    print(f"  skill groups: {len(skills)}")
    for s in skills:
        fr = s.get("frames") or [0, 0]
        print(f"    {s.get('id')}: f{fr[0]}–{fr[1]}  {s.get('note', '')}")
    anchors = board.get("anchors")
    if isinstance(anchors, dict) and "count" in anchors:
        print(f"  live anchors: {anchors.get('count')}  dir={anchors.get('anchors_dir')}")
    elif board.get("anchors_index"):
        print(f"  anchors index: {board['anchors_index']} (unreadable or empty)")
    else:
        print("  live anchors: none (old take or --no-anchors)")


def _print_anchors(board: dict) -> None:
    anchors = board.get("anchors")
    if not isinstance(anchors, dict):
        print("no anchors index next to task JSON")
        return
    rows = anchors.get("anchors") or []
    print(f"anchors: {len(rows)}  dir={anchors.get('anchors_dir')}")
    for a in rows:
        print(
            f"  f{a.get('frame', 0):6d}  {a.get('kind', '?'):12s}  "
            f"{a.get('room')} xy={a.get('xy')} items={a.get('items', '-')}  "
            f"{Path(str(a.get('path') or '')).name}"
        )


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("task", type=Path, help="Path to guided_human task JSON")
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Write extract board JSON (default: <task>_extract.json if not --summary-only)",
    )
    parser.add_argument(
        "--summary",
        action="store_true",
        help="Print hop/skill summary (still writes --out if set)",
    )
    parser.add_argument(
        "--list-anchors",
        action="store_true",
        help="List live anchors from <task>_anchors.json",
    )
    parser.add_argument(
        "--no-write",
        action="store_true",
        help="Never write extract JSON (summary/list only)",
    )
    parser.add_argument(
        "--materialize",
        action="store_true",
        help=(
            "Settle hops + write <task>_run_timing.json (and extract). "
            "One leaf clock: settled ordinary entry + hop dwell."
        ),
    )
    parser.add_argument(
        "--bank",
        action="store_true",
        help="With --materialize: merge hop records into skill bank (dual_green=False)",
    )
    parser.add_argument(
        "--bank-path",
        type=Path,
        default=None,
        help=f"Skill bank JSON (default: {DEFAULT_BANK_PATH})",
    )
    args = parser.parse_args()

    task_path = args.task
    if not task_path.is_file():
        print(f"ERROR: missing task {task_path}", file=sys.stderr)
        return 1

    if args.materialize:
        result = materialize_take(
            task_path,
            write=not args.no_write,
            write_extract=not args.no_write,
            write_run_timing=not args.no_write,
            merge_bank=bool(args.bank) and not args.no_write,
            bank_path=args.bank_path,
        )
        board = extract_tape(task_path)
        # Prefer settled hops for summary when materialize ran.
        board["room_hops"] = result.hops_settled
        board["skill_groups"] = default_skill_groups(result.hops_settled)
        if args.list_anchors:
            _print_anchors(board)
            return 0
        _print_summary(board)
        summary = (result.run_timing or {}).get("summary") or {}
        print(
            f"  materialize: rooms={summary.get('room_visits')} "
            f"items={summary.get('item_splits')} bosses={summary.get('boss_splits')} "
            f"bank_recs={len(result.bank_records)}"
        )
        if result.run_timing_path:
            print(f"  run_timing → {result.run_timing_path}")
        if result.extract_path:
            print(f"  extract → {result.extract_path}")
        if result.bank_path:
            print(f"  bank → {result.bank_path}")
        for note in result.notes:
            print(f"  note: {note}")
        if result.bank_records:
            print("  hop_keys (first 6):")
            for rec in result.bank_records[:6]:
                pin = "pin" if rec.entry_anchor else "no-pin"
                print(f"    {rec.hop_key}  {rec.frames}f  {rec.name}  [{pin}]")
        # Optional override --out still respected for extract path rename
        if args.out is not None and not args.no_write and result.extract_path:
            if args.out.resolve() != result.extract_path.resolve():
                args.out.write_text(
                    result.extract_path.read_text(encoding="utf-8"),
                    encoding="utf-8",
                )
                print(f"wrote {args.out}")
        return 0

    board = extract_tape(task_path)

    if args.list_anchors:
        _print_anchors(board)
        return 0

    if args.summary or args.no_write:
        _print_summary(board)

    out = args.out
    if out is None and not args.no_write and not args.summary:
        out = task_path.with_name(task_path.stem + "_extract.json")
    if out is not None and not args.no_write:
        # Drop bulky nested anchors file body if huge; keep pointer + count.
        slim = dict(board)
        anc = slim.get("anchors")
        if isinstance(anc, dict) and isinstance(anc.get("anchors"), list):
            slim["anchors"] = {
                "task": anc.get("task"),
                "anchors_dir": anc.get("anchors_dir"),
                "count": anc.get("count"),
                "index_path": slim.get("anchors_index"),
                # keep fingerprints only (paths) — full list still in *_anchors.json
                "anchors": anc.get("anchors"),
            }
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(slim, indent=2) + "\n", encoding="utf-8")
        print(f"wrote {out}")
        if not args.summary:
            _print_summary(board)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
