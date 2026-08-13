#!/usr/bin/env python3
"""Build late-spine hop inventory + thrash ranking from human tape extracts.

Reads ``*_extract.json`` (or runs ``extract_tape`` if missing) for the G4/MB
chain and optional late-game free-records. Writes a single thrash board used by
hop-replay / trim / pure-green waves (epic rr-7thf).

```bash
uv run python snes/super_metroid/scripts/tools/build_late_spine_board.py
uv run python snes/super_metroid/scripts/tools/build_late_spine_board.py --summary
```
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[4]
_SNES_IMPORT_ROOT = Path(__file__).resolve().parents[3]
_GAME = Path(__file__).resolve().parents[2]
_TASKS = _GAME / "tasks"
for _p in (ROOT, _SNES_IMPORT_ROOT):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from super_metroid.human_tape import extract_tape  # noqa: E402

# Boss / fight rooms: mode=combat. Big Boy is optional combat (long dwell).
COMBAT_ROOM_IDS: frozenset[int] = frozenset(
    {
        0x9804,  # Bomb Torizo
        0x9DC7,  # Spore Spawn
        0xA59F,  # Kraid
        0xA98D,  # Crocomire
        0xB283,  # Golden Torizo
        0xB32E,  # Ridley
        0xB62B,  # Metal Pirates
        0xCD13,  # Phantoon
        0xD95E,  # Botwoon
        0xDA60,  # Draygon
        0xDAE1,  # Metroid Room 1
        0xDB31,  # Metroid Room 2
        0xDB7D,  # Metroid Room 3
        0xDBCD,  # Metroid Room 4
        0xDCB1,  # Big Boy (optional combat)
        0xDD58,  # Mother Brain
    }
)

# G4 / Tourian / escape path rooms get elevated priority even at modest dwell.
G4_PATH_ROOM_IDS: frozenset[int] = frozenset(
    {
        0xA5ED,  # Statues Hallway
        0xA66A,  # Statues Room
        0xDAAE,  # Tourian Elevator
        0xDF1B,  # Upper Tourian Save
        0xDAE1,
        0xDB31,
        0xDB7D,
        0xDBCD,
        0xDC19,  # Hopper
        0xDC65,  # Dust Torizo
        0xDCB1,  # Big Boy
        0xDCFF,  # Seaweed
        0xDD2E,  # Recharge
        0xDDC4,  # Eye Door
        0xDDF3,  # Rinka Shaft
        0xDE23,  # Lower Tourian Save
        0xDD58,  # Mother Brain
        0xDE4D,  # Escape 1
        0xDE7A,  # Escape 2
        0xDEA7,  # Escape 3
        0xDEDE,  # Escape 4
        0x96BA,  # The Climb
        0x92FD,  # Parlor
        0x91F8,  # Landing Site
    }
)

G4_TAPE_NAMES = frozenset(
    {
        "g4_tourian_human",
        "g4_tourian_human_bb",
        "g4_tourian_human_mb",
    }
)

# Primary late spine (always include when task/extract present).
PRIMARY_TAPES: list[str] = [
    "g4_tourian_human",
    "g4_tourian_human_bb",
    "g4_tourian_human_mb",
    "post-main-hall",
    "post_sj_exit_human",
    "maridia_botwoon_path_human",
]

# Known residual / gap rows (static + runtime-detected).
STATIC_GAPS: list[dict[str, str]] = [
    {
        "tape": "ws_ship_human",
        "issue": "no_anchors",
        "note": "Old free-record before live anchors; re-record short take (rr-7thf.8)",
    },
    {
        "tape": "maridia_grapple_human",
        "issue": "end_state_lost",
        "note": "Open-loop desync overwrote end; use post-grapple pin + extract hops only",
    },
    {
        "tape": "gravity_path_human",
        "issue": "legacy_extract_format",
        "note": "Not a guided_human extract board; snapshots-only JSON",
    },
]

THRASH_DWELL_HIGH = 3000
DEFAULT_OUT = _TASKS / "LATE_SPINE_HOP_BOARD.json"


def _rel_to_game(path: Path | str | None) -> str | None:
    if path is None:
        return None
    p = Path(path)
    try:
        return str(p.resolve().relative_to(_GAME.resolve()))
    except ValueError:
        # Already relative or outside game dir.
        s = str(path)
        if s.startswith("snes/super_metroid/"):
            return s[len("snes/super_metroid/") :]
        return s


def _room_id(hop: dict[str, Any]) -> int:
    if "room_id" in hop and hop["room_id"] is not None:
        return int(hop["room_id"])
    room = hop.get("room") or "0x0"
    if isinstance(room, int):
        return int(room)
    return int(str(room), 16)


def _hop_mode(room_id: int) -> str:
    return "combat" if room_id in COMBAT_ROOM_IDS else "traversal"


def _is_enter_or_boot(kind: str) -> bool:
    k = str(kind or "")
    return k in ("boot", "room_enter", "enter") or k.startswith("enter")


def _match_anchor(
    anchors: list[dict[str, Any]] | None,
    *,
    room: str,
    hop_start_frame: int,
    hop_end_frame: int | None = None,
) -> dict[str, Any] | None:
    """Best enter/boot anchor for a hop visit.

    Live dumps fire on the first *settled ordinary* frame, usually shortly
    after the trace room change. Prefer:

    1. enter/boot inside [hop_start, hop_end] nearest hop_start (this visit)
    2. enter/boot with frame ≤ hop_start nearest hop_start (boot / pre-dump)
    3. any same-room pin inside the dwell window
    """
    if not anchors:
        return None
    end_fr = int(hop_end_frame) if hop_end_frame is not None else hop_start_frame

    def _enter_boot_rows() -> list[dict[str, Any]]:
        return [
            a
            for a in anchors
            if a.get("room") == room
            and _is_enter_or_boot(str(a.get("kind") or ""))
        ]

    # 1) This visit's settle dump (frame in dwell, prefer closest to start).
    best: dict[str, Any] | None = None
    best_delta = 10**12
    for a in _enter_boot_rows():
        fr = int(a.get("frame", 0))
        if fr < hop_start_frame or fr > end_fr:
            continue
        delta = fr - hop_start_frame
        if delta < best_delta:
            best = a
            best_delta = delta
    if best is not None:
        return best

    # 2) Spec fallback: frame ≤ hop start, nearest (covers boot at 0).
    best = None
    best_frame = -1
    for a in _enter_boot_rows():
        fr = int(a.get("frame", 0))
        if fr > hop_start_frame:
            continue
        if fr >= best_frame:
            best = a
            best_frame = fr
    if best is not None:
        return best

    # 3) Any same-room anchor in dwell (item_delta / manual / end).
    best = None
    best_delta = 10**12
    for a in anchors:
        if a.get("room") != room:
            continue
        fr = int(a.get("frame", 0))
        if fr < hop_start_frame or fr > end_fr:
            continue
        delta = abs(fr - hop_start_frame)
        if delta < best_delta:
            best = a
            best_delta = delta
    return best


def _priority(
    *,
    tape_name: str,
    room_id: int,
    dwell: int,
    mode: str,
) -> int:
    """1 = highest (thrash / G4-MB spine), 2 = elevated, 3 = normal."""
    if dwell > THRASH_DWELL_HIGH:
        return 1
    if tape_name in G4_TAPE_NAMES:
        return 1
    if room_id in G4_PATH_ROOM_IDS or mode == "combat":
        return 2
    if dwell >= 1500:
        return 2
    return 3


def _load_extract(name: str, *, refresh: bool) -> dict[str, Any] | None:
    task = _TASKS / f"{name}.json"
    extract_path = _TASKS / f"{name}_extract.json"
    if not task.is_file() and not extract_path.is_file():
        return None
    if refresh or not extract_path.is_file():
        if not task.is_file():
            return None
        board = extract_tape(task)
        # Slim anchors like extract_human_tape CLI.
        slim = dict(board)
        anc = slim.get("anchors")
        if isinstance(anc, dict) and isinstance(anc.get("anchors"), list):
            slim["anchors"] = {
                "task": anc.get("task"),
                "anchors_dir": anc.get("anchors_dir"),
                "count": anc.get("count"),
                "index_path": slim.get("anchors_index"),
                "anchors": anc.get("anchors"),
            }
        extract_path.write_text(json.dumps(slim, indent=2) + "\n", encoding="utf-8")
        return slim
    return json.loads(extract_path.read_text(encoding="utf-8"))


def _anchors_list(extract: dict[str, Any]) -> list[dict[str, Any]]:
    anc = extract.get("anchors")
    if isinstance(anc, dict):
        rows = anc.get("anchors")
        if isinstance(rows, list):
            return rows
    # Fall back to side-car anchors index.
    idx = extract.get("anchors_index")
    if idx and Path(idx).is_file():
        try:
            data = json.loads(Path(idx).read_text(encoding="utf-8"))
            rows = data.get("anchors")
            if isinstance(rows, list):
                return rows
        except (OSError, json.JSONDecodeError):
            pass
    return []


def _leave_room(
    hops: list[dict[str, Any]], index: int
) -> str | None:
    if index + 1 < len(hops):
        return hops[index + 1].get("room")
    return None


def build_tape_entry(name: str, extract: dict[str, Any]) -> dict[str, Any]:
    hops_raw = list(extract.get("room_hops") or [])
    anchors = _anchors_list(extract)
    has_anchors = bool(anchors) or bool(
        isinstance(extract.get("anchors"), dict)
        and int((extract.get("anchors") or {}).get("count") or 0) > 0
    )
    task_rel = _rel_to_game(extract.get("task") or (_TASKS / f"{name}.json"))
    out_hops: list[dict[str, Any]] = []
    for pos, h in enumerate(hops_raw):
        rid = _room_id(h)
        room = h.get("room") or f"0x{rid:04X}"
        dwell = int(h.get("dwell") or 0)
        mode = _hop_mode(rid)
        start_frame = int(h.get("frame", h.get("start_index", 0)))
        end_frame = int(h.get("end_frame", start_frame))
        anchor = _match_anchor(
            anchors,
            room=room,
            hop_start_frame=start_frame,
            hop_end_frame=end_frame,
        )
        anchor_path = None
        if anchor and anchor.get("path"):
            anchor_path = _rel_to_game(anchor["path"])
        idx = int(h.get("index", pos))
        out_hops.append(
            {
                "index": idx,
                "room": room,
                "name": h.get("name") or "?",
                "start_index": int(h.get("start_index", 0)),
                "end_index": int(h.get("end_index", 0)),
                "frame": start_frame,
                "end_frame": int(h.get("end_frame", 0)),
                "dwell": dwell,
                "mode": mode,
                "leave_room": _leave_room(hops_raw, pos),
                "end_xy": h.get("end_xy") or h.get("xy"),
                "anchor_path": anchor_path,
                "priority": _priority(
                    tape_name=name, room_id=rid, dwell=dwell, mode=mode
                ),
            }
        )
    end_fp = extract.get("end_fingerprint")
    if isinstance(end_fp, dict):
        end_fp = dict(end_fp)
        if end_fp.get("path"):
            end_fp["path"] = _rel_to_game(end_fp["path"])
    return {
        "name": name,
        "task": task_rel or f"tasks/{name}.json",
        "extract": f"tasks/{name}_extract.json",
        "frames": int(extract.get("frame_count") or 0),
        "anchors": has_anchors,
        "anchor_count": len(anchors)
        or (
            int((extract.get("anchors") or {}).get("count") or 0)
            if isinstance(extract.get("anchors"), dict)
            else 0
        ),
        "end_fingerprint": end_fp,
        "end_verify_ok": (extract.get("end_verify") or {}).get("ok")
        if isinstance(extract.get("end_verify"), dict)
        else None,
        "hops": out_hops,
    }


def detect_gaps(tape_entries: list[dict[str, Any]]) -> list[dict[str, str]]:
    gaps = [dict(g) for g in STATIC_GAPS]
    seen = {g["tape"] for g in gaps}
    for entry in tape_entries:
        name = entry["name"]
        if not entry.get("anchors"):
            if name not in seen:
                gaps.append(
                    {
                        "tape": name,
                        "issue": "no_anchors",
                        "note": "Extract present but no live *_anchors.json",
                    }
                )
                seen.add(name)
        hops = entry.get("hops") or []
        missing = sum(1 for h in hops if not h.get("anchor_path"))
        if hops and missing and entry.get("anchors"):
            # Partial coverage (re-enters without second enter dump is OK; flag high miss).
            if missing > max(2, len(hops) // 3):
                gaps.append(
                    {
                        "tape": name,
                        "issue": "partial_anchors",
                        "note": f"{missing}/{len(hops)} hops lack matched enter/boot anchor",
                    }
                )
        if entry.get("end_verify_ok") is False:
            gaps.append(
                {
                    "tape": name,
                    "issue": "end_verify_mismatch",
                    "note": "end_fingerprint vs last trace row",
                }
            )
    # Tapes that never produced a guided extract.
    for name in ("ws_ship_human",):
        if name not in seen and not (_TASKS / f"{name}_extract.json").is_file():
            # already in STATIC_GAPS
            pass
    return gaps


def build_board(
    *,
    tape_names: list[str] | None = None,
    refresh: bool = False,
) -> dict[str, Any]:
    names = list(tape_names or PRIMARY_TAPES)
    tapes: list[dict[str, Any]] = []
    missing: list[str] = []
    for name in names:
        extract = _load_extract(name, refresh=refresh)
        if extract is None:
            missing.append(name)
            continue
        # Skip legacy non-hop extracts (gravity_path style).
        if not extract.get("room_hops") and extract.get("snapshots") is not None:
            missing.append(f"{name}(legacy)")
            continue
        tapes.append(build_tape_entry(name, extract))

    thrash: list[dict[str, Any]] = []
    for t in tapes:
        for h in t.get("hops") or []:
            thrash.append(
                {
                    "dwell": int(h["dwell"]),
                    "room": h["room"],
                    "name": h.get("name"),
                    "tape": t["name"],
                    "hop": int(h["index"]),
                    "mode": h["mode"],
                    "priority": h["priority"],
                    "anchor_path": h.get("anchor_path"),
                    "leave_room": h.get("leave_room"),
                }
            )
    thrash.sort(key=lambda r: (-int(r["dwell"]), r["tape"], int(r["hop"])))

    gaps = detect_gaps(tapes)
    if missing:
        for m in missing:
            base = m.split("(")[0]
            if not any(g.get("tape") == base for g in gaps):
                gaps.append(
                    {
                        "tape": base,
                        "issue": "missing_extract",
                        "note": f"no task/extract for {m}",
                    }
                )

    return {
        "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "pipeline": "hop-replay → trim → pure dual green",
        "epic": "rr-7thf",
        "source": "scripts/tools/build_late_spine_board.py",
        "combat_rooms": [f"0x{r:04X}" for r in sorted(COMBAT_ROOM_IDS)],
        "tapes": tapes,
        "thrash_ranking": thrash,
        "gaps": gaps,
        "wave_order": [
            {
                "wave": "A",
                "bead": "rr-7thf.4",
                "focus": "Tourian Escape 1–4",
                "tape": "g4_tourian_human_mb",
                "rooms": ["0xDE4D", "0xDE7A", "0xDEA7", "0xDEDE"],
            },
            {
                "wave": "B",
                "bead": "rr-7thf.5",
                "focus": "Climb + Parlor + Landing Site / ship",
                "tape": "g4_tourian_human_mb",
                "rooms": ["0x96BA", "0x92FD", "0x91F8"],
            },
            {
                "wave": "C",
                "bead": "rr-7thf.6",
                "focus": "G4 statues → Metroids → Big Boy",
                "tape": "g4_tourian_human",
                "rooms": [
                    "0xA66A",
                    "0xDAE1",
                    "0xDB31",
                    "0xDB7D",
                    "0xDBCD",
                    "0xDCB1",
                ],
            },
            {
                "wave": "D",
                "bead": "rr-7thf.7",
                "focus": "MB approach + Mother Brain fight",
                "tapes": ["g4_tourian_human_bb", "g4_tourian_human_mb"],
                "rooms": ["0xDD58"],
            },
            {
                "wave": "thrash_queue",
                "bead": "rr-7thf.9",
                "focus": "Long-dwell thrash rooms from ranking",
                "note": "Ridley / Worst / Metal Pirates / Draygon / Colosseum / Everest",
            },
        ],
        "stats": {
            "tape_count": len(tapes),
            "hop_count": sum(len(t.get("hops") or []) for t in tapes),
            "thrash_count": len(thrash),
            "priority_1": sum(
                1
                for t in tapes
                for h in (t.get("hops") or [])
                if h.get("priority") == 1
            ),
        },
    }


def _print_summary(board: dict[str, Any]) -> None:
    print(f"generated_at: {board.get('generated_at')}")
    print(f"pipeline: {board.get('pipeline')}")
    stats = board.get("stats") or {}
    print(
        f"tapes={stats.get('tape_count')} hops={stats.get('hop_count')} "
        f"p1={stats.get('priority_1')}"
    )
    for t in board.get("tapes") or []:
        print(
            f"  {t['name']}: frames={t.get('frames')} hops={len(t.get('hops') or [])} "
            f"anchors={t.get('anchors')} ({t.get('anchor_count')})"
        )
    print("thrash_ranking (top 12):")
    for row in (board.get("thrash_ranking") or [])[:12]:
        print(
            f"  dwell={row['dwell']:5d}  {row['room']}  {row.get('name')}  "
            f"tape={row['tape']} hop={row['hop']} mode={row['mode']} p={row['priority']}"
        )
    print("gaps:")
    for g in board.get("gaps") or []:
        print(f"  {g.get('tape')}: {g.get('issue')} — {g.get('note', '')}")


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=DEFAULT_OUT,
        help=f"Output board JSON (default: {DEFAULT_OUT})",
    )
    parser.add_argument(
        "--refresh",
        action="store_true",
        help="Re-run extract_tape for each primary tape before building",
    )
    parser.add_argument(
        "--summary",
        action="store_true",
        help="Print thrash/gap summary after write",
    )
    parser.add_argument(
        "--tapes",
        nargs="*",
        default=None,
        help="Override tape stem list (default: primary late spine set)",
    )
    args = parser.parse_args()

    board = build_board(tape_names=args.tapes, refresh=args.refresh)
    out = args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(board, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {out}")
    if args.summary:
        _print_summary(board)
    else:
        stats = board.get("stats") or {}
        thrash = board.get("thrash_ranking") or []
        top = thrash[0] if thrash else None
        top_s = (
            f"top_thrash={top['dwell']}f {top['room']}@{top['tape']}"
            if top
            else "thrash=empty"
        )
        print(
            f"  tapes={stats.get('tape_count')} hops={stats.get('hop_count')} "
            f"p1={stats.get('priority_1')} {top_s}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
