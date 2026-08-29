#!/usr/bin/env python3
"""Export the completion-path room board (no door-warp clearance claims).

Walks ``maps/full_route_hops.json`` in research completion order and emits a
machine-readable board of every unique room + every directed hop that must
eventually be *played* (controller/policy), not door-warped.

```bash
uv run python snes/super_metroid/scripts/export/path_room_board.py
uv run python snes/super_metroid/scripts/export/path_room_board.py --json PATH
```
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
from super_metroid.paths import GAME_DIR, MAPS_DIR  # noqa: E402
from super_metroid.dev.route_dev import FULL_LEG_ORDER, leg_key, load_full_hops  # noqa: E402

FULL_GRAPH_PATH = MAPS_DIR / "full_room_graph.json"
ROOM_PROBLEMS_PATH = MAPS_DIR / "room_problems.json"
DEFAULT_JSON = MAPS_DIR / "path_room_board.json"
DEFAULT_MD = GAME_DIR / "docs" / "research" / "PATH_ROOM_BOARD.md"

# Hand-curated clearance status for path rooms (update when a room promotes).
# continuous = natural power-on evidence exists through this room on some accepted route
# controller_dev = controller crosses from a natural/dev entry (not continuous yet)
# policy_verified = isolated room_clears verified_development_state
# open = not cleared by play
# boss_deferred = boss fight intentionally later; still need natural entry eventually
ROOM_STATUS: dict[str, dict[str, str]] = {
    # Continuous power-on → Super collect covers Ceres + early Zebes + Spore.
    # Exact intermediate rooms match continuous tip controllers (not every research
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
        "note": "Spore Super room collect continuous",
    },
    # Continuous KPDR through the post-Varia return to Business (K3).
    "0xA0A4": {
        "status": "continuous",
        "note": "Farming via play_super_room_to_farming continuous",
    },
    "0x9D19": {
        "status": "continuous",
        "note": "Big Pink main continuous; Pink PB still controller_dev side trip",
    },
    "0x9E11": {
        "status": "controller_dev",
        "note": "Pink PB: wall@437 pure break; collect from x≤225; mid-maze 405→225 OPEN",
    },
    "0x9E52": {"status": "continuous", "note": "GHZ continuous on K1 tip"},
    "0x9FBA": {"status": "continuous", "note": "Noob Bridge continuous on K1 tip"},
    "0xA253": {"status": "continuous", "note": "Red Tower continuous on K1 tip"},
    "0xA3DD": {"status": "continuous", "note": "Bat Room continuous"},
    "0xA408": {"status": "continuous", "note": "Below Spazer continuous"},
    "0xCF54": {
        "status": "continuous",
        "note": "West Tunnel continuous on warehouse tip",
    },
    "0xCEFB": {
        "status": "continuous",
        "note": "Glass Tunnel continuous on warehouse tip",
    },
    "0xCF80": {
        "status": "continuous",
        "note": "East Tunnel continuous on warehouse tip",
    },
    "0xA6A1": {
        "status": "continuous",
        "note": "Warehouse Entrance continuous on both K2 and post-Varia return",
    },
    "0xA7DE": {
        "status": "continuous",
        "note": "Business Center return continuous (frog, 114923f x2)",
    },
    "0xB167": {
        "status": "continuous",
        "note": "Frog Savestation continuous (frog, 114923f x2)",
    },
    "0xAA41": {
        "status": "continuous",
        "note": "Hi-Jump Shaft: real E-Tank, intended ledges, ordinary bomb tunnel",
    },
    "0xA9E5": {
        "status": "continuous",
        "note": "Hi-Jump Boots real PLM collect; item bit 0x0100",
    },
    "0xA471": {"status": "continuous", "note": "Warehouse Zeela return continuous"},
    "0xA4DA": {"status": "continuous", "note": "Warehouse Kihunter return continuous"},
    "0xA521": {"status": "continuous", "note": "Baby Kraid return continuous"},
    "0xA56B": {"status": "continuous", "note": "Kraid Eye Door return continuous"},
    "0xA59F": {
        "status": "continuous",
        "note": "Kraid fight and post-fight return continuous",
    },
    "0xA6E2": {"status": "continuous", "note": "Varia Suit collect continuous"},
    # Cathedral-first Bubble pure stack (CATH-01…04) + Bubble→Bat R19.
    # Continuous tip still Frog Save until planner compose.
    "0xA7B3": {
        "status": "controller_dev",
        "note": "Cathedral Entrance pure (CATH-01); first Bubble path",
    },
    "0xA788": {
        "status": "controller_dev",
        "note": "Cathedral pure (CATH-02/03); first Bubble path",
    },
    "0xAFA3": {
        "status": "controller_dev",
        "note": "Rising Tide pure (CATH-03/04); first Bubble path",
    },
    "0xACB3": {
        "status": "controller_dev",
        "note": "Bubble Mountain pure GREEN (CATH-04 entry + R19 climb→Bat)",
    },
    "0xB07A": {
        "status": "controller_dev",
        "note": "Bat Cave pure GREEN R19 (2012f from post_rising_tide_to_bubble_pure)",
    },
    # Boss rooms — entry required eventually; fights deferred
    "0xCD13": {
        "status": "boss_deferred",
        "note": "Phantoon — entry warp-only; fight open",
    },
    "0xD95E": {"status": "boss_deferred", "note": "Botwoon"},
    "0xDA60": {"status": "boss_deferred", "note": "Draygon"},
    "0xB32E": {"status": "boss_deferred", "note": "Ridley"},
    "0xDD58": {"status": "boss_deferred", "note": "Mother Brain"},
}

# Directed hops with known controller (from→to), independent of room status.
HOP_STATUS: dict[str, dict[str, str]] = {
    "0x9B5B->0xA0A4": {
        "status": "continuous",
        "note": "play_super_room_to_farming",
    },
    "0xA0A4->0x9D19": {
        "status": "continuous",
        "note": "play_farming_to_big_pink",
    },
    "0x9D19->0x9E11": {
        "status": "controller_dev",
        "note": "sill entry green; ★ pure sill approach + mid-maze 405→225 still open",
    },
    "0x9D19->0x9E52": {"status": "continuous", "note": "play_big_pink_to_ghz"},
    "0x9E52->0x9FBA": {"status": "continuous", "note": "play_ghz_to_noob"},
    "0x9FBA->0xA253": {"status": "continuous", "note": "play_noob_to_red_tower"},
    "0xA253->0xA3DD": {"status": "continuous", "note": "play_red_tower_to_bat"},
    "0xA3DD->0xA408": {"status": "continuous", "note": "play_bat_to_below_spazer"},
    "0xA408->0xCF54": {"status": "continuous", "note": "play_below_spazer_to_west"},
    "0xCF54->0xCEFB": {"status": "continuous", "note": "play_west_to_glass"},
    "0xCEFB->0xCF80": {"status": "continuous", "note": "play_glass_to_east"},
    "0xCF80->0xA6A1": {"status": "continuous", "note": "play_east_to_warehouse"},
    "0xA6A1->0xA7DE": {
        "status": "continuous",
        "note": "play_warehouse_to_business; business return",
    },
    "0xA7DE->0xB167": {
        "status": "continuous",
        "note": "play_business_to_frog_save; frog",
    },
    "0xA7DE->0xAA41": {"status": "continuous", "note": "play_business_to_hj_shaft"},
    "0xAA41->0xA9E5": {"status": "continuous", "note": "play_hj_shaft_to_hj_room"},
    "0xA9E5->0xAA41": {"status": "continuous", "note": "play_hj_room_to_shaft"},
    "0xAA41->0xA7DE": {"status": "continuous", "note": "play_hj_shaft_to_business"},
    "0xA7DE->0xA6A1": {
        "status": "continuous",
        "note": "play_business_to_warehouse",
    },
    "0xA6A1->0xA471": {
        "status": "continuous",
        "note": "play_warehouse_to_zeela_with_hijump",
    },
    "0xA471->0xA4DA": {"status": "continuous", "note": "play_zeela_to_kihunter"},
    "0xA4DA->0xA521": {
        "status": "continuous",
        "note": "play_kihunter_to_baby_kraid",
    },
    "0xA521->0xA56B": {"status": "continuous", "note": "play_baby_kraid_to_eye"},
    "0xA56B->0xA59F": {"status": "continuous", "note": "play_eye_to_kraid"},
    "0xA59F->0xA6E2": {"status": "continuous", "note": "play_kraid_entry_to_varia"},
    "0xA6E2->0xA59F": {"status": "continuous", "note": "play_varia_to_kraid"},
    "0xA59F->0xA56B": {"status": "continuous", "note": "play_kraid_to_eye_return"},
    "0xA56B->0xA521": {"status": "continuous", "note": "play_eye_to_baby_return"},
    "0xA521->0xA4DA": {"status": "continuous", "note": "play_baby_to_kihunter_return"},
    "0xA4DA->0xA471": {"status": "continuous", "note": "play_kihunter_to_zeela_return"},
    "0xA471->0xA6A1": {
        "status": "continuous",
        "note": "play_zeela_to_warehouse_return",
    },
    # Cathedral first Bubble + Bubble→Bat (pure GREEN; not continuous tip yet)
    "0xA7DE->0xA7B3": {
        "status": "controller_dev",
        "note": "play_business_to_cathedral_entrance (CATH-01 pure)",
    },
    "0xA7B3->0xA788": {
        "status": "controller_dev",
        "note": "play_cathedral_entrance_to_cathedral (CATH-02 pure)",
    },
    "0xA788->0xAFA3": {
        "status": "controller_dev",
        "note": "play_cathedral_to_rising_tide (CATH-03 pure)",
    },
    "0xAFA3->0xACB3": {
        "status": "controller_dev",
        "note": "play_rising_tide_to_bubble (CATH-04 pure)",
    },
    "0xACB3->0xB07A": {
        "status": "controller_dev",
        "note": "play_bubble_to_bat_cave (R19 pure GREEN 2012f)",
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

    # The accepted KPDR tip reaches Frog Save. The first K4 forward gate is
    # Frog Save → Frog Speedway; fall back to topology order if it disappears.
    first_open_hop = next((h for h in hops if h["status"] == "open"), None)
    frog_hop = next(
        (h for h in hops if h["from"] == "0xB167" and h["to"] == "0xB106"),
        None,
    )

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
            "roomIdHex": "0xB167",
            "name": names.get("0xB167", "Frog Savestation"),
            "evidence": "recordings/frog*.json (114923f x2)",
        },
        "furthestControllerDev": None,
        "nextOpenHop": frog_hop or first_open_hop,
        "rooms": rooms_out,
        "hops": hops,
        "waves": _waves(),
    }


def _waves() -> list[dict[str, object]]:
    """Ordered work waves for clearing the path by play."""
    return [
        {
            "id": "W0",
            "title": "Continuous prefix",
            "goal": "Power-on → Super collect",
            "status": "done",
            "roomsApprox": 20,
            "bosses": "Bomb Torizo + Spore Spawn natural",
        },
        {
            "id": "W1",
            "title": "Super → Red Tower (KPDR K1)",
            "goal": "0x9B5B → farm → Big Pink → GHZ → Noob → Red Tower",
            "status": "done",
            "done": [
                "Super collect continuous",
                "farming hop continuous",
                "Big Pink to main shaft continuous",
                "direct Big Pink → GHZ → Noob → Red Tower continuous",
            ],
            "open": [
                "Charge Beam conventional return (no IBJ)",
            ],
        },
        {
            "id": "W2",
            "title": "Red Tower → Hi-Jump → Kraid entry (KPDR K2)",
            "goal": "Warehouse → Hi-Jump real PLM → return → natural Kraid entry",
            "status": "done",
            "done": [
                "continuous power-on → Bat / Below Spazer / Warehouse",
                "Red Tower → Warehouse continuous (warehouse 83512f)",
                "Hi-Jump E-Tank + Boots real PLMs (controller)",
                "Hi-Jump intended ledges + ordinary bomb-tunnel return",
                "Warehouse → Zeela → Kihunter → Baby Kraid → Eye → Kraid",
                "15356f composed Warehouse suffix; no IBJ",
                "continuous Warehouse → Hi-Jump → Kraid",
            ],
            "open": [],
        },
        {
            "id": "W3",
            "title": "Kraid → Varia → Speed → Ice by play",
            "goal": "Frog Save → Speedway → Bubble → Speed → Ice",
            "status": "in_progress",
            "done": ["Kraid fight + Varia + reverse return + Frog Save continuous"],
            "open": ["Frog Save → Speedway pure controller"],
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
            "bosses": [
                "Kraid",
                "Phantoon",
                "Botwoon",
                "Draygon",
                "Ridley",
                "Mother Brain",
            ],
        },
    ]


def render_markdown(board: dict[str, object]) -> str:
    lines: list[str] = [
        "# Path room board — clear by play (no door-warp evidence)",
        "",
        "Generated by `scripts/export/path_room_board.py`. "
        "Machine copy: `maps/path_room_board.json`.",
        "",
        "## Principle",
        "",
        str(board["principle"]),
        "",
        "Door-warp tools (`dev/route_dev.py`) remain useful "
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
    if fd:
        lines.append(
            f"| **Controller (dev)** | `{fd['roomIdHex']}` {fd['name']} "
            f"~({fd['position']['samusX']},{fd['position']['samusY']}) | `{fd['probe']}` |"
        )
    else:
        lines.append(
            "| **Controller (dev)** | — | No controller-only clearance beyond the accepted tip |"
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
            "| Status | Count |",
            "|--------|------:|",
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
            "uv run python snes/super_metroid/scripts/room/run_problem.py scaffold PROBLEM_ID",
            "uv run python snes/super_metroid/scripts/room/run_problem.py capture PROBLEM_ID STATE",
            "uv run python snes/super_metroid/scripts/room/run_problem.py run PROBLEM_ID",
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
            "1. Compose the Kraid fight from the natural controller entry.",
            "2. Take Kraid's rear door and collect Varia from the real PLM.",
            "3. Finish Charge with a conventional return; do not route an IBJ.",
            "4. Compose K1→K2, then re-prove it from continuous power-on.",
            "5. Refresh this board after every promotion:",
            "",
            "```bash",
            "uv run python snes/super_metroid/scripts/export/path_room_board.py",
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
    parser.add_argument(
        "--print", action="store_true", help="Print summary JSON to stdout"
    )
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
        "furthestControllerDev": (
            None
            if board["furthestControllerDev"] is None
            else board["furthestControllerDev"]["roomIdHex"]
        ),
    }
    print(json.dumps(summary, indent=2))
    if args.print:
        print(json.dumps(board, indent=2))


if __name__ == "__main__":
    main()
