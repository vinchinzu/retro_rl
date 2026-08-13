#!/usr/bin/env python3
"""Rank Red Tower → Hellway human practice takes (multi-attempt splice board).

Does **not** open-loop replay or concatenate button streams. Scans take JSONs
under a series folder (or explicit paths) and ranks by:

1. Reached Hellway ``0xA2F7`` (required for success)
2. Shorter Red dwell / total frames (cleaner climb)
3. End room still Hellway (or past Hellway)

```bash
# Rank a practice series
uv run python snes/super_metroid/scripts/tools/rank_red_climb_takes.py \\
  --series red_climb_v1

# Rank any globs
uv run python snes/super_metroid/scripts/tools/rank_red_climb_takes.py \\
  snes/super_metroid/tasks/red_climb_v1/*.json

# Write splice manifest (best take path + sibling tape pins)
uv run python snes/super_metroid/scripts/tools/rank_red_climb_takes.py \\
  --series red_climb_v1 --write-manifest \\
  snes/super_metroid/tasks/red_climb_v1/splice_manifest.json
```
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]

from super_metroid.paths import GAME_DIR  # noqa: E402

ROOM_RED = 0xA253
ROOM_HELLWAY = 0xA2F7
TASKS = GAME_DIR / "tasks"


@dataclass
class TakeScore:
    path: str
    frames: int
    reached_hellway: bool
    red_enter_index: int | None
    hellway_enter_index: int | None
    red_dwell: int | None
    end_room: str
    end_xy: tuple[int, int] | None
    n_room_enters: int
    grade: str  # GREEN | YELLOW | RED


def _room(fr: dict) -> int | None:
    if "room" in fr and fr["room"] is not None:
        return int(fr["room"])
    hx = fr.get("room_hex")
    if isinstance(hx, str) and hx.startswith("0x"):
        return int(hx, 16)
    return None


def score_take(path: Path) -> TakeScore:
    data = json.loads(path.read_text(encoding="utf-8"))
    trace = data.get("trace") or []
    frames = int(data.get("frame_count") or len(trace) or len(data.get("frames") or []))
    red_enter: int | None = None
    hell_enter: int | None = None
    n_enters = 0
    prev: int | None = None
    for i, fr in enumerate(trace):
        r = _room(fr)
        if r is None:
            continue
        if r != prev:
            n_enters += 1
            if r == ROOM_RED and red_enter is None:
                red_enter = i
            if r == ROOM_HELLWAY and hell_enter is None:
                hell_enter = i
            prev = r
    end = trace[-1] if trace else {}
    er = _room(end)
    end_room = f"0x{er:04X}" if er is not None else "?"
    end_xy = None
    if "x" in end and "y" in end:
        end_xy = (int(end["x"]), int(end["y"]))
    red_dwell = None
    if red_enter is not None and hell_enter is not None:
        red_dwell = hell_enter - red_enter
    elif red_enter is not None:
        red_dwell = frames - red_enter
    reached = hell_enter is not None
    if reached and red_dwell is not None and red_dwell <= 4500:
        grade = "GREEN"
    elif reached:
        grade = "YELLOW"  # Hellway but long thrash
    else:
        grade = "RED"
    return TakeScore(
        path=str(path),
        frames=frames,
        reached_hellway=reached,
        red_enter_index=red_enter,
        hellway_enter_index=hell_enter,
        red_dwell=red_dwell,
        end_room=end_room,
        end_xy=end_xy,
        n_room_enters=n_enters,
        grade=grade,
    )


def _sort_key(s: TakeScore) -> tuple:
    # Prefer: reached Hellway, shorter red_dwell, fewer total frames
    return (
        0 if s.reached_hellway else 1,
        s.red_dwell if s.red_dwell is not None else 10**9,
        s.frames,
        s.path,
    )


def _collect_paths(series: str | None, globs: list[str]) -> list[Path]:
    paths: list[Path] = []
    if series:
        d = TASKS / series
        if d.is_dir():
            paths.extend(sorted(d.glob(f"{series}_take*.json")))
            paths.extend(sorted(d.glob("take*.json")))
        # Also flat series_takeNN under tasks/
        paths.extend(sorted(TASKS.glob(f"{series}_take*.json")))
    for g in globs:
        p = Path(g)
        if p.is_file():
            paths.append(p)
        else:
            paths.extend(sorted(Path().glob(g)))
            paths.extend(sorted(ROOT.glob(g)))
    # Dedupe preserve order
    seen: set[str] = set()
    out: list[Path] = []
    for p in paths:
        key = str(p.resolve())
        if key in seen:
            continue
        if p.suffix != ".json":
            continue
        # Skip anchors indexes / extracts
        if p.name.endswith("_anchors.json") or p.name.endswith("_extract.json"):
            continue
        if "manifest" in p.name:
            continue
        seen.add(key)
        out.append(p)
    return out


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument(
        "paths",
        nargs="*",
        help="Take JSON paths or globs (optional if --series)",
    )
    ap.add_argument(
        "--series",
        default=None,
        help="Practice series stem under tasks/<series>/",
    )
    ap.add_argument(
        "--write-manifest",
        type=Path,
        default=None,
        help="Write splice board manifest JSON (best + all ranked)",
    )
    ap.add_argument(
        "--top",
        type=int,
        default=20,
        help="Print top N (default 20)",
    )
    args = ap.parse_args()
    paths = _collect_paths(args.series, list(args.paths))
    if not paths:
        print("No take JSONs found. Pass --series red_climb_v1 or paths.", file=sys.stderr)
        return 1

    scores = []
    for p in paths:
        try:
            scores.append(score_take(p))
        except (OSError, json.JSONDecodeError, ValueError, TypeError) as exc:
            print(f"  skip {p}: {exc}", file=sys.stderr)
    scores.sort(key=_sort_key)

    print(f"Ranked {len(scores)} takes  (Hellway=0xA2F7, Red=0xA253)")
    print(f"{'grade':6} {'red_dwell':>9} {'frames':>7} {'end':8} path")
    for s in scores[: max(1, args.top)]:
        dwell = f"{s.red_dwell}" if s.red_dwell is not None else "-"
        print(
            f"{s.grade:6} {dwell:>9} {s.frames:7d} {s.end_room:8} {s.path}"
        )
    ok = [s for s in scores if s.reached_hellway]
    print()
    if ok:
        best = ok[0]
        print(f"BEST: {best.path}")
        print(
            f"  grade={best.grade} red_dwell={best.red_dwell} "
            f"frames={best.frames} end={best.end_room} xy={best.end_xy}"
        )
        print("  hop-replay from take boot/enter anchors — do not full open-loop.")
    else:
        print("No take reached Hellway yet.")

    if args.write_manifest is not None:
        best = ok[0] if ok else None
        manifest = {
            "pipeline": "hop-board-splice (anchors) — not button-stream concat",
            "segment": "red-to-hellway",
            "room_red": "0xA253",
            "room_hellway": "0xA2F7",
            "best": asdict(best) if best else None,
            "ranked": [asdict(s) for s in scores],
            "spine_siblings": {
                "ice_to_warehouse": "tasks/ice_to_red_human.json",
                "warehouse_to_alpha_pb_sloppy_red": "tasks/warehouse_to_red_human.json",
                "red_enter_live": (
                    "tasks/warehouse_to_red_human_anchors/"
                    "f002012_enter_0xA253_0xA253.state"
                ),
                "pure_red_bottom": "scratch/post_ice_bat_to_red_pure.state",
                "alpha_pb_to_moat": "tasks/alpha_pb_to_moat_human.json",
                "moat_end": "tasks/alpha_pb_to_moat_human_end.state",
            },
            "how_to_splice": [
                "Prefix hops: pure dual Ice→…→Red OR ice_to_red + warehouse hops 0–5",
                "Red climb: best take from this series (live boot = Red enter)",
                "Suffix: Hellway enter pin → Alpha PB → alpha_pb_to_moat_human",
                "Each piece hop-replays from its own live anchor — never concat frames",
            ],
        }
        out = args.write_manifest
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
        print(f"\n[manifest] {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
