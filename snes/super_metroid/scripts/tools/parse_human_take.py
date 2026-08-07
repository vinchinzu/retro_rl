#!/usr/bin/env python3
"""Parse a guided_human task JSON into phase pins + path JSON for DC Wave practice.

Default target: Double Chamber missile → Wave (room 0xADAD → 0xADDE).

```bash
# Reference take04 → product path data
uv run python snes/super_metroid/scripts/tools/parse_human_take.py \\
  snes/super_metroid/tasks/dc_missile_v1/dc_missile_v1_take04.json

# Custom out path
uv run python snes/super_metroid/scripts/tools/parse_human_take.py \\
  snes/super_metroid/tasks/dc_missile_v1/dc_missile_v1_take04.json \\
  --out snes/super_metroid/routes/kpdr/data/dc_missile_wave_take04_paths.json

# Summary only
uv run python snes/super_metroid/scripts/tools/parse_human_take.py \\
  snes/super_metroid/tasks/dc_missile_v1/dc_missile_v1_take04.json --summary
```
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
DEFAULT_OUT = (
    ROOT
    / "snes/super_metroid/routes/kpdr/data/dc_missile_wave_take04_paths.json"
)

ROOM_DC = "0xADAD"
ROOM_WAVE = "0xADDE"


def _load_task(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _slice_f(trace: list[dict], a: int, b: int) -> list[dict]:
    return [t for t in trace if a <= int(t["frame"]) <= b]


def _rle_buttons(rows: list[dict]) -> list[dict]:
    rle: list[dict] = []
    for t in rows:
        key = list(t.get("buttons") or [])
        if rle and rle[-1]["buttons"] == key:
            rle[-1]["n"] += 1
        else:
            rle.append({"n": 1, "buttons": key})
    return rle


def _downsample(
    rows: list[dict],
    *,
    dx: int = 28,
    dy: int = 16,
    max_y: int | None = None,
    room: str = ROOM_DC,
) -> list[dict]:
    out: list[dict] = []
    last: tuple[int, int] | None = None
    for t in rows:
        if t.get("room_hex") != room:
            continue
        x, y = int(t["x"]), int(t["y"])
        if max_y is not None and y > max_y:
            continue
        if last is None or abs(x - last[0]) >= dx or abs(y - last[1]) >= dy:
            out.append(
                {
                    "x": x,
                    "y": y,
                    "frame": int(t["frame"]),
                    "pose": int(t.get("pose") or 0),
                }
            )
            last = (x, y)
    return out


def _find_phases(trace: list[dict]) -> dict[str, list[int]]:
    """Heuristic phase bounds from events (works for DC Wave takes)."""
    n = len(trace) - 1
    # defaults
    p1_end = n
    past_gate = None
    free_510 = None
    edge = None
    wave = None
    gate_seat = None

    for t in trace:
        f, x, y = int(t["frame"]), int(t["x"]), int(t["y"])
        room = t.get("room_hex")
        if room == ROOM_DC and gate_seat is None and 370 <= x <= 390 and y <= 145:
            gate_seat = f
        if room == ROOM_DC and past_gate is None and x >= 480 and y <= 165:
            past_gate = f
        if (
            room == ROOM_DC
            and free_510 is None
            and x >= 510
            and y <= 160
            and int(t.get("missiles") or 0) >= 20
        ):
            free_510 = f
        if room == ROOM_DC and edge is None and x >= 590 and y <= 160 and f > (past_gate or 0):
            edge = f
        if room == ROOM_WAVE and wave is None:
            wave = f

    if gate_seat is not None:
        p1_end = gate_seat
    if past_gate is None:
        past_gate = min(p1_end + 1, n)
    if free_510 is None:
        free_510 = past_gate
    if edge is None:
        edge = free_510
    if wave is None:
        wave = n

    return {
        "P1_entry_hop": [0, p1_end],
        "P2_gate_open": [p1_end, past_gate],
        "P3_missile_free": [past_gate, free_510],
        "P4_runway_dash": [free_510, edge],
        "P5_launch_super": [edge, wave],
        "P6_wave_collect": [wave, n],
    }


def _floor_recover(trace: list[dict]) -> tuple[list[dict], list[dict]]:
    """First floor drop → reseat interval for recovery path."""
    start = None
    end = None
    for t in trace:
        if t.get("room_hex") != ROOM_DC:
            continue
        y = int(t["y"])
        if start is None and y >= 300:
            start = t
        if start is not None and y <= 160 and int(t.get("vy") or 0) == 0:
            end = t
            break
    if start is None or end is None:
        return [], []
    rows = _slice_f(trace, int(start["frame"]), int(end["frame"]))
    path = _downsample(rows, dx=25, dy=20, max_y=None)
    pins = [
        {
            "x": int(start["x"]),
            "y": int(start["y"]),
            "label": "fall-pin",
            "phase": "P1-recover",
        },
        {
            "x": int(end["x"]),
            "y": int(end["y"]),
            "label": "reseat-P2",
            "phase": "P1-recover",
        },
    ]
    # Add mid samples
    for t in rows:
        if int(t["y"]) >= 400:
            pins.insert(
                1,
                {
                    "x": int(t["x"]),
                    "y": int(t["y"]),
                    "label": "floor",
                    "phase": "P1-recover",
                },
            )
            break
    return path, pins


def parse_take(task_path: Path) -> dict:
    d = _load_task(task_path)
    trace = list(d.get("trace") or [])
    if not trace:
        raise ValueError(f"no trace in {task_path}")

    bounds = _find_phases(trace)
    end = trace[-1]
    missile_collect = next(
        (t for t in trace if int(t.get("missiles") or 0) >= 20), None
    )
    free_510 = next(
        (
            t
            for t in trace
            if int(t["x"]) >= 510
            and int(t.get("missiles") or 0) >= 20
            and int(t["y"]) <= 160
        ),
        None,
    )
    wave = next((t for t in trace if t.get("room_hex") == ROOM_WAVE), None)

    past = bounds["P3_missile_free"][0]
    p2_end = bounds["P5_launch_super"][1]
    p2_rle = _rle_buttons(_slice_f(trace, past, p2_end))

    recover_path, recover_pins = _floor_recover(trace)
    main_gate = _downsample(
        _slice_f(trace, bounds["P2_gate_open"][0], p2_end),
        dx=22,
        dy=14,
        max_y=280,
    )

    floor_f = sum(
        1 for t in trace if t.get("room_hex") == ROOM_DC and int(t["y"]) >= 360
    )

    return {
        "schema": "dc_missile_wave_paths/v1",
        "source_task": str(task_path.as_posix()),
        "source_state": d.get("start_state"),
        "recorded_at": d.get("recorded_at"),
        "frame_count": d.get("frame_count") or len(trace),
        "result": {
            "end_room": end.get("room_hex"),
            "end_xy": [end.get("x"), end.get("y")],
            "missiles_end": end.get("missiles"),
            "floor_frames": floor_f,
            "wave_ok": end.get("room_hex") == ROOM_WAVE,
        },
        "phases": {
            name: {
                "frames": fr,
                "duration": fr[1] - fr[0],
            }
            for name, fr in bounds.items()
        },
        "pins": {
            "main": [
                {"x": 61, "y": 139, "label": "P1-entry", "phase": "P1"},
                {"x": 214, "y": 122, "label": "P1-hop2", "phase": "P1"},
                {"x": 379, "y": 139, "label": "P2-gate-seat", "phase": "P2"},
                {"x": 480, "y": 139, "label": "P2-past-gate", "phase": "P2"},
                {"x": 494, "y": 139, "label": "P3-missile", "phase": "P3"},
                {"x": 510, "y": 139, "label": "P3-free", "phase": "P3"},
                {"x": 437, "y": 139, "label": "P4-runway", "phase": "P4"},
                {"x": 600, "y": 139, "label": "P4-edge", "phase": "P4"},
                {"x": 647, "y": 60, "label": "P5-peak", "phase": "P5"},
                {"x": 903, "y": 248, "label": "P5-door-WJ", "phase": "P5"},
                {"x": 929, "y": 139, "label": "P5-sill", "phase": "P5"},
            ],
            "recover": recover_pins,
        },
        "paths": {
            "main_gate_to_wave": main_gate,
            "recover_floor_to_seat": recover_path,
        },
        "timings": {
            "missile_collect_f": missile_collect["frame"] if missile_collect else None,
            "missile_collect_xy": (
                [missile_collect["x"], missile_collect["y"]]
                if missile_collect
                else None
            ),
            "free_past_510_f": free_510["frame"] if free_510 else None,
            "free_duration_f": (
                int(free_510["frame"]) - int(missile_collect["frame"])
                if free_510 and missile_collect
                else None
            ),
            "wave_room_f": wave["frame"] if wave else None,
        },
        "human_rle_p2_to_wave_door": p2_rle,
        "fallback_rules": [
            {
                "when": "y >= 300 and room == 0xADAD and x < 480",
                "do": "follow recover path to reseat ~(379,139); do not Super from floor",
                "rejoin_phase": "P2_gate_open",
            },
            {
                "when": "missiles >= 20 and x in [488,505] and vx == 0",
                "do": "RIGHT+B free until x >= 510",
                "rejoin_phase": "P3_missile_free",
            },
            {
                "when": "y >= 280 and x >= 700",
                "do": "abort door WJ; return to ledge runway",
                "rejoin_phase": "P4_runway_dash",
            },
        ],
    }


def _print_summary(data: dict) -> None:
    print(f"source:  {data.get('source_task')}")
    print(f"frames:  {data.get('frame_count')}")
    print(f"result:  {data.get('result')}")
    print(f"timings: {data.get('timings')}")
    print("phases:")
    for name, info in (data.get("phases") or {}).items():
        fr = info.get("frames")
        print(f"  {name:20s} f={fr[0]}..{fr[1]} ({info.get('duration')}f)")
    rle = data.get("human_rle_p2_to_wave_door") or []
    print(f"P2→door RLE segments: {len(rle)}")
    print(f"recover path pts: {len((data.get('paths') or {}).get('recover_floor_to_seat') or [])}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("task", type=Path, help="Path to *takeNN.json")
    ap.add_argument(
        "--out",
        type=Path,
        default=None,
        help=f"Write paths JSON (default: {DEFAULT_OUT} for take04 name)",
    )
    ap.add_argument(
        "--summary",
        action="store_true",
        help="Print phase summary only (no write unless --out set)",
    )
    args = ap.parse_args()
    task = args.task
    if not task.is_file():
        # allow repo-relative
        alt = ROOT / task
        if alt.is_file():
            task = alt
        else:
            print(f"ERROR: missing {task}", file=sys.stderr)
            return 1

    data = parse_take(task)
    _print_summary(data)

    out = args.out
    if out is None and not args.summary:
        # default product path for take04-like names
        out = DEFAULT_OUT
    if out is not None:
        out = Path(out)
        if not out.is_absolute():
            out = ROOT / out
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
        print(f"\nwrote {out} ({out.stat().st_size} bytes)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
