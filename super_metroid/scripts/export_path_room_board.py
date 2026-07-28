#!/usr/bin/env python3
"""Export the completion-path room board (no door-warp clearance claims).

Walks ``maps/full_route_hops.json`` in research completion order and emits a
machine-readable board of every unique room + every directed hop that must
eventually be *played* (controller/policy), not door-warped.

```bash
uv run python super_metroid/scripts/export_path_room_board.py
uv run python super_metroid/scripts/export_path_room_board.py --json PATH
```
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from super_metroid.paths import GAME_DIR, MAPS_DIR, RECORDINGS_DIR  # noqa: E402
from super_metroid.route_dev import FULL_LEG_ORDER, leg_key, load_full_hops  # noqa: E402

FULL_GRAPH_PATH = MAPS_DIR / "full_room_graph.json"
ROOM_PROBLEMS_PATH = MAPS_DIR / "room_problems.json"
DEFAULT_JSON = MAPS_DIR / "path_room_board.json"
DEFAULT_MD = GAME_DIR / "docs" / "PATH_ROOM_BOARD.md"

# Hand-curated clearance status for path rooms (update when a room promotes).
# continuous = natural power-on evidence exists through this room on some accepted route
# controller_dev = controller crosses from a natural/dev entry (not continuous yet)
# policy_verified = isolated room_clears verified_development_state
# open = not cleared by play
# boss_deferred = boss fight intentionally later; still need natural entry eventually
ROOM_STATUS: dict[str, dict[str, str]] = {
    # Continuous power-on → Super collect covers Ceres + early Zebes + Spore.
    # Exact intermediate rooms match start_to_* controllers (not every research
    # hop on bomb_torizo→spore may match any% order — marked continuous_prefix
    # where the continuous product path owns them).
    "0xDF45": {"status": "continuous", "note": "Ceres start"},
    "0xDF8D": {"status": "continuous", "note": "Ceres"},
    "0xDFD7": {"status": "continuous", "note": "Ceres"},
    "0xE021": {"status": "continuous", "note": "Ceres"},
    "0xE06B": {"status": "continuous", "note": "Ceres"},
    "0xE0B5": {"status": "continuous", "note": "Ceres Ridley escape sequence"},
    "0x91F8": {"status": "continuous", "note": "Landing Site (early + finish)"},
    "0x92FD": {"status": "continuous", "note": "Parlor"},
    "0x96BA": {"status": "continuous", "note": "Climb"},
    "0x975C": {"status": "continuous", "note": "Pit Room"},
    "0x97B5": {"status": "continuous", "note": "Elev to Morph"},
    "0x9E9F": {"status": "continuous", "note": "Morph Ball"},
    "0x9F11": {"status": "continuous", "note": "Construction Zone"},
    "0xA107": {"status": "continuous", "note": "First Missiles"},
    "0x9879": {"status": "continuous", "note": "Flyway (+ room_clears policy)"},
    "0x9804": {"status": "continuous", "note": "Bomb Torizo fight continuous"},
    "0x990D": {"status": "continuous", "note": "Terminator (post-Torizo controller)"},
    "0x99BD": {"status": "continuous", "note": "Green Pirates Shaft"},
    "0x9DC7": {"status": "continuous", "note": "Spore Spawn fight continuous"},
    "0x9B5B": {
        "status": "continuous",
        "note": "Spore Super room collect continuous (furthest continuous)",
    },
    # Post-Super controller (dev from natural_post_spore; not full power-on yet)
    "0xA0A4": {
        "status": "controller_dev",
        "note": "Farming via play_super_room_to_farming",
    },
    "0x9D19": {
        "status": "controller_dev",
        "note": "Big Pink: crest+tunnel→main shaft (play_big_pink_into_main_shaft); PB climb OPEN",
    },
    # Boss rooms — entry required eventually; fights deferred
    "0xA59F": {"status": "boss_deferred", "note": "Kraid — dev spray exists; natural entry open"},
    "0xCD13": {"status": "boss_deferred", "note": "Phantoon — entry warp-only; fight open"},
    "0xD95E": {"status": "boss_deferred", "note": "Botwoon"},
    "0xDA60": {"status": "boss_deferred", "note": "Draygon"},
    "0xB32E": {"status": "boss_deferred", "note": "Ridley"},
    "0xDD58": {"status": "boss_deferred", "note": "Mother Brain"},
}

# Directed hops with known controller (from→to), independent of room status.
HOP_STATUS: dict[str, dict[str, str]] = {
    "0x9B5B->0xA0A4": {
        "status": "controller_dev",
        "note": "play_super_room_to_farming",
    },
    "0xA0A4->0x9D19": {
        "status": "controller_dev",
        "note": "play_farming_to_big_pink",
    },
    "0x9D19->0x9E11": {
        "status": "open",
        "note": "★ NEXT: climb shaft to PB door 0x8E02 / block [32,71]",
    },
}


def _room_names() -> dict[str, str]:
    if not FULL_GRAPH_PATH.is_file():
        return {}
    graph = json.loads(FULL_GRAPH_PATH.read_text(encoding="utf-8"))
    out: dict[str, str] = {}
    for room in graph.get("rooms", []):
        rid = int(room["roomId"])
        out[f"0x{rid:04X}"] = str(room.get("name") or room.get("roomName") or "?")
    return out


def _problem_meta() -> dict[str, dict[str, object]]:
    if not ROOM_PROBLEMS_PATH.is_file():
        return {}
    catalog = json.loads(ROOM_PROBLEMS_PATH.read_text(encoding="utf-8"))
    out: dict[str, dict[str, object]] = {}
    for problem in catalog.get("problems", []):
        rid = str(problem.get("roomIdHex") or "")
        if not rid:
            continue
        out[rid] = {
            "problemId": problem.get("problemId"),
            "tier": problem.get("tier"),
            "queue": problem.get("queue"),
            "practiceStatus": (problem.get("practice") or {}).get("status"),
        }
    return out


def build_board() -> dict[str, object]:
    hops_data = load_full_hops()
    names = _room_names()
    problems = _problem_meta()

    ordered_rooms: list[str] = []
    hops: list[dict[str, object]] = []
    for source, target in FULL_LEG_ORDER:
        key = leg_key(source, target)
        chain = hops_data[key]
        for hop in chain:
            fr = str(hop["from"])
            to = str(hop["to"])
            for room in (fr, to):
                if room not in ordered_rooms:
                    ordered_rooms.append(room)
            hop_key = f"{fr}->{to}"
            hop_meta = HOP_STATUS.get(hop_key, {"status": "open", "note": ""})
            hops.append(
                {
                    "from": fr,
                    "to": to,
                    "door": hop.get("door"),
                    "leg": key,
                    "legSource": source,
                    "legTarget": target,
                    "status": hop_meta["status"],
                    "note": hop_meta.get("note", ""),
                    "fromName": names.get(fr, "?"),
                    "toName": names.get(to, "?"),
                }
            )

    rooms_out: list[dict[str, object]] = []
    status_counts: dict[str, int] = {}
    for index, room in enumerate(ordered_rooms):
        meta = ROOM_STATUS.get(room, {"status": "open", "note": ""})
        status = str(meta["status"])
        status_counts[status] = status_counts.get(status, 0) + 1
        prob = problems.get(room, {})
        rooms_out.append(
            {
                "index": index,
                "roomIdHex": room,
                "name": names.get(room, "?"),
                "status": status,
                "note": meta.get("note", ""),
                "tier": prob.get("tier"),
                "queue": prob.get("queue"),
                "problemId": prob.get("problemId"),
            }
        )

    # First open hop on the board (skip hops already continuous/controller).
    first_open_hop = next(
        (h for h in hops if h["status"] == "open"),
        None,
    )
    # Prefer the explicit PB climb hop if present.
    pb_hop = next((h for h in hops if h["from"] == "0x9D19" and h["to"] == "0x9E11"), None)

    return {
        "schemaVersion": 1,
        "principle": (
            "Clearance requires playing the room (controller/policy). "
            "Door-warps are topology diagnostics only — not route evidence."
        ),
        "hopCount": len(hops),
        "uniqueRoomCount": len(rooms_out),
        "statusCounts": status_counts,
        "furthestContinuous": {
            "roomIdHex": "0x9B5B",
            "name": names.get("0x9B5B", "Spore Spawn Super Room"),
            "evidence": "recordings/start_to_supers.json",
        },
        "furthestControllerDev": {
            "roomIdHex": "0x9D19",
            "name": names.get("0x9D19", "Big Pink"),
            "position": {"samusX": 746, "samusY": 1465},
            "note": "main shaft after play_big_pink_into_main_shaft from natural_post_spore",
            "probe": "probe_post_spore_pb.py --to main",
        },
        "nextOpenHop": pb_hop or first_open_hop,
        "rooms": rooms_out,
        "hops": hops,
        "waves": _waves(),
    }


def _waves() -> list[dict[str, object]]:
    """Ordered work waves for clearing the path by play."""
    return [
        {
            "id": "W0",
            "title": "Continuous prefix (done)",
            "goal": "Power-on → Super collect",
            "status": "done",
            "roomsApprox": 20,
            "bosses": "Bomb Torizo + Spore Spawn natural",
        },
        {
            "id": "W1",
            "title": "Super → Power Bombs by play",
            "goal": "0x9B5B → 0xA0A4 → 0x9D19 → 0x9E11 + PB collect",
            "status": "in_progress",
            "done": ["Super collect continuous", "farming hop controller_dev", "Big Pink to main shaft"],
            "open": ["Big Pink climb to PB door", "natural PB collect", "continuous power-on through PB"],
            "hops": 3,
        },
        {
            "id": "W2",
            "title": "PB → Kraid approach (research path)",
            "goal": "Play GHZ/Noob/Red Tower/Warehouse rooms into Kraid entry",
            "status": "open",
            "hops": 15,
            "note": "Any% ship route can diverge after PB; prefer research hops if single board",
        },
        {
            "id": "W3",
            "title": "Kraid → Varia → Speed → Ice by play",
            "goal": "Natural Kraid (or deferred fight) + item rooms + Norfair halls",
            "status": "open",
            "bosses": "Kraid fight when spine reaches door",
        },
        {
            "id": "W4",
            "title": "Ice → Phantoon approach by play",
            "goal": "Maridia tube / elev / Moat / Ocean / WS shaft rooms",
            "status": "open",
            "hardRooms": ["Moat", "West Ocean", "WS Main Shaft", "WS Basement"],
        },
        {
            "id": "W5",
            "title": "Phantoon → Gravity → Botwoon → Draygon by play",
            "goal": "WS + Maridia sand/hall rooms",
            "status": "open",
            "bosses": "Phantoon, Botwoon, Draygon deferred until rooms approachable",
        },
        {
            "id": "W6",
            "title": "Draygon → Ridley (LN) by play",
            "goal": "Lower Norfair path rooms",
            "status": "open",
            "bosses": "Ridley deferred",
        },
        {
            "id": "W7",
            "title": "Ridley → Statues → Tourian → MB by play",
            "goal": "Statues, elevator, metroid rooms, MB approach",
            "status": "open",
            "bosses": "Mother Brain deferred",
        },
        {
            "id": "W8",
            "title": "Escape → Landing Site by play",
            "goal": "Escape 1–4, Climb, Parlor, ship/credits",
            "status": "open",
            "note": "Requires MB defeat event + escape timer init",
        },
        {
            "id": "W9",
            "title": "Boss fight scripts (parallel after room entry exists)",
            "goal": "Natural fights for remaining bosses + credits predicate",
            "status": "deferred",
            "bosses": ["Kraid", "Phantoon", "Botwoon", "Draygon", "Ridley", "Mother Brain"],
        },
    ]


def render_markdown(board: dict[str, object]) -> str:
    lines: list[str] = [
        "# Path room board — clear by play (no door-warp evidence)",
        "",
        f"Generated by `scripts/export_path_room_board.py`. "
        f"Machine copy: `maps/path_room_board.json`.",
        "",
        "## Principle",
        "",
        str(board["principle"]),
        "",
        "Door-warp tools (`probe_route.py full` / `full-hybrid`) remain useful "
        "only for topology debugging. **Route progress** = natural room exit "
        "under a controller/policy, eventually chained from power-on.",
        "",
        "## How far we are (empirical)",
        "",
        "| Layer | Furthest | Evidence |",
        "|-------|----------|----------|",
    ]
    fc = board["furthestContinuous"]
    fd = board["furthestControllerDev"]
    lines.append(
        f"| **Continuous** | `{fc['roomIdHex']}` {fc['name']} | `{fc['evidence']}` |"
    )
    lines.append(
        f"| **Controller (dev)** | `{fd['roomIdHex']}` {fd['name']} "
        f"~({fd['position']['samusX']},{fd['position']['samusY']}) | `{fd['probe']}` |"
    )
    nxt = board.get("nextOpenHop") or {}
    if nxt:
        lines.extend(
            [
                "",
                "### ★ Next hop to play",
                "",
                f"- **{nxt.get('from')} → {nxt.get('to')}** "
                f"({nxt.get('fromName')} → {nxt.get('toName')})",
                f"- Door: `{nxt.get('door')}`",
                f"- Note: {nxt.get('note') or 'open'}",
                f"- Leg: `{nxt.get('leg')}`",
            ]
        )

    counts = board["statusCounts"]
    lines.extend(
        [
            "",
            "## Status counts (unique path rooms)",
            "",
            f"| Status | Count |",
            f"|--------|------:|",
        ]
    )
    for key in (
        "continuous",
        "controller_dev",
        "policy_verified",
        "boss_deferred",
        "open",
    ):
        if key in counts:
            lines.append(f"| {key} | {counts[key]} |")
    lines.append(f"| **total** | **{board['uniqueRoomCount']}** |")
    lines.append(f"| directed hops | {board['hopCount']} |")

    lines.extend(
        [
            "",
            "## Clearance method (per room / hop)",
            "",
            "1. **Natural entry** — arrive from predecessor by play (preferred) or",
            "   capture a one-time entry state *from that natural exit*.",
            "2. **Attempt** — controller or compact room policy (not door-warp).",
            "3. **Exit predicate** — ordinary gameplay in the next room (or item delta).",
            "4. **Promote** — `controller_dev` → continuous suffix → power-on chain.",
            "5. **Boss rooms** — natural *entry* first; fight script is a separate gate.",
            "",
            "Isolated practice loop (still valid):",
            "",
            "```bash",
            "uv run python super_metroid/scripts/run_room_problem.py scaffold PROBLEM_ID",
            "uv run python super_metroid/scripts/run_room_problem.py capture PROBLEM_ID STATE",
            "uv run python super_metroid/scripts/run_room_problem.py run PROBLEM_ID",
            "```",
            "",
            "Path priority: only the **~107 completion-path rooms** first, not all 262.",
            "",
            "## Work waves",
            "",
        ]
    )
    for wave in board["waves"]:
        lines.append(f"### {wave['id']} — {wave['title']} ({wave['status']})")
        lines.append("")
        lines.append(f"**Goal:** {wave['goal']}")
        if wave.get("done"):
            lines.append("")
            lines.append("Done:")
            for item in wave["done"]:
                lines.append(f"- [x] {item}")
        if wave.get("open"):
            lines.append("")
            lines.append("Open:")
            for item in wave["open"]:
                lines.append(f"- [ ] {item}")
        if wave.get("hardRooms"):
            lines.append("")
            lines.append("Hard rooms: " + ", ".join(wave["hardRooms"]))
        if wave.get("note"):
            lines.append("")
            lines.append(f"_{wave['note']}_")
        lines.append("")

    lines.extend(
        [
            "## Full room list (path order of first visit)",
            "",
            "| # | Room | Name | Status | Tier | Note |",
            "|--:|------|------|--------|------|------|",
        ]
    )
    for room in board["rooms"]:
        note = str(room.get("note") or "").replace("|", "/")
        lines.append(
            f"| {room['index']} | `{room['roomIdHex']}` | {room['name']} | "
            f"**{room['status']}** | {room.get('tier') or '?'} | {note} |"
        )

    lines.extend(
        [
            "",
            "## Immediate actions",
            "",
            "1. Solve **Big Pink → Pink PB** (`0x9D19 → 0x9E11`) by play from main shaft.",
            "2. Compose Super→PB controller; re-prove on continuous power-on.",
            "3. For each following hop: capture natural entry → attempt policy → promote.",
            "4. Boss fights only after natural entry to that boss room exists on the chain.",
            "5. Refresh this board after every promotion:",
            "",
            "```bash",
            "uv run python super_metroid/scripts/export_path_room_board.py",
            "```",
            "",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", type=Path, default=DEFAULT_JSON)
    parser.add_argument("--markdown", type=Path, default=DEFAULT_MD)
    parser.add_argument("--no-markdown", action="store_true")
    parser.add_argument("--print", action="store_true", help="Print summary JSON to stdout")
    args = parser.parse_args()

    board = build_board()
    args.json.parent.mkdir(parents=True, exist_ok=True)
    args.json.write_text(json.dumps(board, indent=2) + "\n", encoding="utf-8")

    if not args.no_markdown:
        args.markdown.parent.mkdir(parents=True, exist_ok=True)
        args.markdown.write_text(render_markdown(board), encoding="utf-8")

    summary = {
        "json": str(args.json),
        "markdown": None if args.no_markdown else str(args.markdown),
        "uniqueRoomCount": board["uniqueRoomCount"],
        "hopCount": board["hopCount"],
        "statusCounts": board["statusCounts"],
        "nextOpenHop": board.get("nextOpenHop"),
        "furthestContinuous": board["furthestContinuous"]["roomIdHex"],
        "furthestControllerDev": board["furthestControllerDev"]["roomIdHex"],
    }
    print(json.dumps(summary, indent=2))
    if args.print:
        print(json.dumps(board, indent=2))


if __name__ == "__main__":
    main()
