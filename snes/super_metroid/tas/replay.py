"""CLI: replay Super Metroid TAS movies/slices under stable-retro and annotate.

```bash
# Menu smoke (fast)
SDL_VIDEODRIVER=dummy SDL_AUDIODRIVER=dummy \\
  uv run python -m super_metroid.tas.replay --slice sniq_any_menu --annotate

# Short contest BK2 end-to-end
uv run python -m super_metroid.tas.replay --slice moozooh_smtc4_full \\
  --annotate --series-stride 4 --states-on room_enter,control

# Full any% power-on (long; expect core desync — still useful for milestones)
uv run python -m super_metroid.tas.replay --slice sniq_any_full \\
  --annotate --series-stride 8 --states-on room_enter,item_gain,beam_gain,control \\
  --out snes/super_metroid/recordings/tas_import/sniq_any_full

# Movie path + window
uv run python -m super_metroid.tas.replay \\
  --movie snes/super_metroid/tas/ref/sniq_any_3653M.lsmv \\
  --start 0 --end 20000 --annotate --series-stride 2
```
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

from super_metroid.ram import SuperMetroidState
from super_metroid.tas.slice import SLICE_CATALOG
from super_metroid.tas.trace import (
    DEFAULT_OUT_ROOT,
    resolve_frames,
    trace_frames,
    write_trace_artifacts,
)


def _parse_kinds(raw: str | None) -> list[str]:
    if not raw:
        return []
    return [p.strip() for p in raw.split(",") if p.strip()]


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    src = p.add_mutually_exclusive_group(required=False)
    src.add_argument("--slice", dest="slice_id", help="Catalog slice id")
    src.add_argument("--movie", type=Path, help="Path to .lsmv / .bk2")
    src.add_argument("--seed", type=Path, help="Path to snes12_rle JSON seed")
    p.add_argument("--list-slices", action="store_true", help="Print catalog and exit")
    p.add_argument("--start", type=int, default=None, help="Frame window start")
    p.add_argument("--end", type=int, default=None, help="Frame window end (exclusive)")
    p.add_argument("--max-frames", type=int, default=None, help="Cap frames played")
    p.add_argument(
        "--state",
        default=None,
        help="Integration state stem (default: power-on NONE)",
    )
    p.add_argument(
        "--state-path",
        type=Path,
        default=None,
        help="Explicit .state file loaded after reset",
    )
    p.add_argument(
        "--annotate",
        action="store_true",
        default=True,
        help="Enable event annotation (default on)",
    )
    p.add_argument(
        "--series-stride",
        type=int,
        default=0,
        help="Record kinematics every N frames (0=off, 1=every frame)",
    )
    p.add_argument(
        "--parse-mode",
        choices=("nav", "full"),
        default="nav",
        help="WRAM parse mode (nav=fast low WRAM; full=bank $7E)",
    )
    p.add_argument(
        "--stall-frames",
        type=int,
        default=90,
        help="Frozen pose+xy threshold for desync_suspect",
    )
    p.add_argument(
        "--states-on",
        default="",
        help="Comma event kinds that dump .state (room_enter,control,item_gain,…)",
    )
    p.add_argument(
        "--dump-every",
        type=int,
        default=0,
        help="Also dump .state every N frames",
    )
    p.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Output directory (default recordings/tas_import/<run_id>/)",
    )
    p.add_argument(
        "--no-write",
        action="store_true",
        help="Do not write artifacts (stdout summary only)",
    )
    p.add_argument(
        "--progress-every",
        type=int,
        default=5000,
        help="Print progress every N frames (0=off)",
    )
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    if args.list_slices:
        for sid, spec in sorted(SLICE_CATALOG.items()):
            tags = ",".join(spec.tags)
            exists = "ok" if spec.movie.exists() else "MISSING"
            print(f"{sid:28s}  [{tags}]  {exists}  {spec.notes[:60]}")
        return 0

    if not args.slice_id and not args.movie and not args.seed:
        print(
            "error: provide --slice, --movie, or --seed (or --list-slices)",
            file=sys.stderr,
        )
        return 2

    t0 = time.perf_counter()
    try:
        frames, source = resolve_frames(
            movie=args.movie,
            slice_id=args.slice_id,
            seed_path=args.seed,
            start=args.start,
            end=args.end,
        )
    except (FileNotFoundError, KeyError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    run_id = args.slice_id or (
        args.movie.stem if args.movie else Path(args.seed).stem  # type: ignore[arg-type]
    )
    if args.start is not None or args.end is not None:
        run_id = f"{run_id}_{args.start or 0}_{args.end or 'end'}"
    out_dir = args.out or (DEFAULT_OUT_ROOT / run_id)
    states_dir = None
    dump_on = _parse_kinds(args.states_on)
    if dump_on or args.dump_every:
        states_dir = out_dir / "states"

    n = len(frames)
    cap = args.max_frames if args.max_frames is not None else n
    print(
        f"replay source={source} frames={n} play={min(n, cap)} "
        f"start_mode={args.state or args.state_path or 'poweron'}",
        flush=True,
    )

    t_chunk = time.perf_counter()
    last_mark = 0

    def _progress(
        frame_i: int, total: int, state: SuperMetroidState, event_count: int
    ) -> None:
        nonlocal t_chunk, last_mark
        dt = time.perf_counter() - t_chunk
        delta = frame_i - last_mark
        rate = delta / dt if dt > 0 else 0.0
        print(
            f"  … f={frame_i}/{total} room=0x{state.room_id:04X} "
            f"pose={state.pose} xy=({state.samus_x},{state.samus_y}) "
            f"events={event_count} ({rate:.0f} f/s)",
            flush=True,
        )
        t_chunk = time.perf_counter()
        last_mark = frame_i

    trace = trace_frames(
        frames,
        source=source,
        state_name=args.state,
        state_path=args.state_path,
        max_frames=args.max_frames,
        series_stride=int(args.series_stride),
        parse_mode=args.parse_mode,
        stall_frames=args.stall_frames,
        dump_states_on=dump_on,
        dump_every=args.dump_every,
        states_dir=states_dir,
        progress_every=int(args.progress_every),
        on_progress=_progress if args.progress_every > 0 else None,
    )

    elapsed = time.perf_counter() - t0
    summary = trace.summary()
    summary["elapsed_s"] = round(elapsed, 2)
    summary["fps"] = round(trace.frames_played / elapsed, 1) if elapsed > 0 else 0

    if not args.no_write:
        written = write_trace_artifacts(
            trace,
            out_dir,
            write_series=args.series_stride > 0,
        )
        print(f"wrote {out_dir}", flush=True)
        for k, path in written.items():
            print(f"  {k}: {path}", flush=True)
        if trace.state_dumps:
            print(f"  states: {len(trace.state_dumps)} under {states_dir}", flush=True)

    ann = summary.get("annotate") or {}
    print(
        f"done frames={trace.frames_played} events={len(trace.events)} "
        f"rooms={len(trace.rooms)} series={len(trace.series)} "
        f"elapsed={elapsed:.1f}s",
        flush=True,
    )
    print(f"  by_kind: {ann.get('by_kind')}", flush=True)
    print(f"  first_control: {ann.get('first_control_frame')}", flush=True)
    if ann.get("item_gains"):
        print(f"  gains: {len(ann['item_gains'])}", flush=True)
        for g in ann["item_gains"][:25]:
            print(
                f"    f{g['frame']} {g['kind']} {g['detail']} "
                f"room=0x{g['room_id']:04X}",
                flush=True,
            )
    if ann.get("desync_suspects"):
        print(f"  desync_suspects: {len(ann['desync_suspects'])}", flush=True)
        for d in ann["desync_suspects"][:10]:
            print(f"    f{d['frame']} {d['detail']}", flush=True)
    if trace.final:
        fin = trace.final
        print(
            f"  final: room={fin.get('room')} pose={fin.get('pose')} "
            f"xy=({fin.get('x')},{fin.get('y')}) phase={fin.get('phase')} "
            f"items={fin.get('items')} beams={fin.get('beams')}",
            flush=True,
        )

    if not args.no_write:
        print(
            json.dumps(
                {
                    "ok": True,
                    "out": str(out_dir),
                    "summary": {
                        "frames_played": trace.frames_played,
                        "events": len(trace.events),
                        "rooms": len(trace.rooms),
                        "first_control": ann.get("first_control_frame"),
                        "by_kind": ann.get("by_kind"),
                        "elapsed_s": summary["elapsed_s"],
                    },
                },
                separators=(",", ":"),
            )
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
