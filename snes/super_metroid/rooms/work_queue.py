"""Easiest-first room-clear work queue and percent-complete tracking.

Canonical units (practice harness only — not continuous evidence):
- **262 room problems** — one development problem per room.
- **583 directed edges** — full physical topology (research graph).
- **199 completion hops / 107 path rooms** — any% path board (topology).

Ranks isolated practice: teleport in, clear, promote. Bosses and large/tough
rooms stay at the bottom so percent-complete grows from the easy class first.

**Product next-work (continuous tip) is not owned here.** That lives in
``docs/STATUS.md`` + beads ``bd ready -l super_metroid`` (+ KPDR tracker). This board is
dual-track metrics only.
"""

from __future__ import annotations

from collections import Counter
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from super_metroid.paths import (
    FULL_ROOM_GRAPH_PATH,
    GAME_DIR,
    MAPS_DIR,
    ROOM_PROBLEMS_PATH,
)
from super_metroid.rooms.room_graph import load_problem_catalog

DEFAULT_QUEUE_JSON = MAPS_DIR / "room_work_queue.json"
DEFAULT_QUEUE_CSV = GAME_DIR / "docs" / "routes" / "ROOM_WORK_QUEUE.csv"
DEFAULT_QUEUE_MD = GAME_DIR / "docs" / "routes" / "ROOM_WORK_QUEUE.md"
PATH_BOARD_PATH = MAPS_DIR / "path_room_board.json"
KPDR_TRACKER_PATH = MAPS_DIR / "kpdr_tracker.json"

# Lower objective weight = easier to verify in isolation.
_OBJECTIVE_WEIGHT: dict[str, int] = {
    "visit_station_and_return": 0,
    "enter_objective_and_return": 1,
    "traverse_to_exit": 2,
    "collect_and_return": 3,
    "collect_items_and_exit": 4,
    "scripted_escape": 8,
    "defeat_boss_and_exit": 10,
}

_TIER_ORDER = {
    "easy": 0,
    "standard": 1,
    "tough": 2,
    "late_special": 3,
    "boss_late": 4,
}

_STATUS_DONE = frozenset({"ready", "verified_development_state"})


def resolve_practice_status(
    problem: Mapping[str, Any],
    *,
    state_ok: bool | None = None,
    policy_status: str | None = None,
) -> str:
    """Single practice-status resolver for catalog rows and live board ranking.

    Order of truth:
    1. Live policy ``verified_development_state`` + entry state → ``ready``
    2. Entry state on disk → ``state_ready``
    3. Catalog ``practice.status`` (else ``unstarted``)
    """
    practice = problem.get("practice") or {}
    if state_ok is None or policy_status is None:
        state_exists, policy_exists, _ = _practice_paths_exist(problem)
        if state_ok is None:
            state_ok = state_exists
        if policy_status is None and policy_exists:
            policy_path = GAME_DIR / str(practice.get("policyFile", ""))
            try:
                payload = json.loads(policy_path.read_text(encoding="utf-8"))
                if isinstance(payload, dict):
                    policy_status = str(payload.get("status") or "")
            except (OSError, json.JSONDecodeError, TypeError):
                policy_status = None

    if policy_status == "verified_development_state" and state_ok:
        return "ready"
    catalog_status = str(practice.get("status", "unstarted"))
    if catalog_status == "ready" and state_ok:
        return "ready"
    if state_ok:
        return "state_ready"
    return catalog_status if catalog_status else "unstarted"


def difficulty_score(problem: Mapping[str, Any]) -> int:
    """Return a rough lower-is-easier score for isolated room practice.

    Primary order is still catalog ``queue`` (0 ready … 4 boss). Within a
    class, smaller geometry, fewer enemies, a static air path, and simpler
    objectives rank earlier. Does **not** inject continuous-spine priority.
    """
    queue = int(problem.get("queue", 3))
    geo = problem.get("geometry") or {}
    screens = int(geo.get("widthScreens", 1)) * int(geo.get("heightScreens", 1))
    enemies = int(geo.get("enemyCount", 0))
    plan = problem.get("staticPlan") or {}
    plan_status = str(plan.get("status", "unresolved"))
    path_blocks = int(plan.get("pathBlocks") or 0)
    objective = str(problem.get("objective", "traverse_to_exit"))
    reasons = list(problem.get("difficultyReasons") or [])

    score = queue * 10_000
    score += _TIER_ORDER.get(str(problem.get("tier", "tough")), 2) * 1_000
    score += _OBJECTIVE_WEIGHT.get(objective, 5) * 50
    score += screens * 25
    score += enemies * 12
    if plan_status == "planned_static":
        score += min(path_blocks, 200)
    elif plan_status == "unresolved":
        score += 400
    else:
        score += 600
    if "large/vertical geometry" in reasons:
        score += 150
    if "dense enemies" in reasons:
        score += 80
    if "gated exit" in reasons:
        score += 40
    if "static collision path unresolved" in reasons:
        score += 100
    # Ready / state-ready sort first so open work sits below finished practice.
    status = str((problem.get("practice") or {}).get("status", "unstarted"))
    if status == "ready":
        score = min(score, 0)
    elif status == "state_ready":
        score = min(score, 50)
    return score


def _practice_paths_exist(problem: Mapping[str, Any]) -> tuple[bool, bool, bool]:
    practice = problem.get("practice") or {}
    state_path = GAME_DIR / str(practice.get("stateFile", ""))
    policy_path = GAME_DIR / str(practice.get("policyFile", ""))
    report_path = GAME_DIR / str(practice.get("reportFile", ""))
    return state_path.is_file(), policy_path.is_file(), report_path.is_file()


def _load_path_room_ids(path: Path = PATH_BOARD_PATH) -> set[int]:
    if not path.is_file():
        return set()
    board = json.loads(path.read_text(encoding="utf-8"))
    rooms = board.get("rooms") or []
    ids: set[int] = set()
    for room in rooms:
        if "roomId" in room:
            ids.add(int(room["roomId"]))
        elif "roomIdHex" in room:
            ids.add(int(str(room["roomIdHex"]), 0))
    return ids


def _load_kpdr_room_ids(path: Path = KPDR_TRACKER_PATH) -> set[int]:
    if not path.is_file():
        return set()
    payload = json.loads(path.read_text(encoding="utf-8"))
    segments = payload.get("segments") or []
    ids: set[int] = set()
    for segment in segments:
        raw = segment.get("room_id_hex") or segment.get("roomIdHex")
        if raw:
            try:
                ids.add(int(str(raw), 0))
            except ValueError:
                continue
    return ids


def annotate_problem(
    problem: Mapping[str, Any],
    *,
    path_room_ids: set[int],
    kpdr_room_ids: set[int],
    rank: int,
) -> dict[str, Any]:
    """Attach ranking + teleport readiness fields for one problem."""
    state_ok, policy_ok, report_ok = _practice_paths_exist(problem)
    practice = problem.get("practice") or {}
    status = resolve_practice_status(problem, state_ok=state_ok)
    room_id = int(problem["roomId"])
    # Score with live practice status so state_ready sorts near the top.
    scored = dict(problem)
    scored_practice = dict(practice)
    scored_practice["status"] = status
    scored["practice"] = scored_practice
    score = difficulty_score(scored)
    geo = problem.get("geometry") or {}
    screens = int(geo.get("widthScreens", 1)) * int(geo.get("heightScreens", 1))
    entry = problem.get("entry")
    exit_ = problem.get("exit")
    return {
        "rank": rank,
        "difficultyScore": score,
        "problemId": problem["problemId"],
        "roomId": room_id,
        "roomIdHex": problem["roomIdHex"],
        "roomName": problem["roomName"],
        "area": problem.get("area"),
        "objective": problem.get("objective"),
        "tier": problem.get("tier"),
        "queue": int(problem.get("queue", 3)),
        "difficultyReasons": list(problem.get("difficultyReasons") or []),
        "screens": screens,
        "enemyCount": int(geo.get("enemyCount", 0)),
        "staticPlanStatus": (problem.get("staticPlan") or {}).get("status"),
        "pathBlocks": (problem.get("staticPlan") or {}).get("pathBlocks"),
        "onCompletionPath": room_id in path_room_ids,
        "onKpdrTracker": room_id in kpdr_room_ids,
        "practiceStatus": status,
        "hasEntryState": state_ok,
        "hasPolicy": policy_ok,
        "hasReport": report_ok,
        "teleportReady": state_ok,
        "runReady": state_ok and policy_ok and status == "ready",
        "entrySourceRoomIdHex": (
            entry.get("sourceRoomIdHex") if isinstance(entry, dict) else None
        ),
        "exitTargetRoomIdHex": (
            exit_.get("targetRoomIdHex") if isinstance(exit_, dict) else None
        ),
        "stateFile": practice.get("stateFile"),
        "policyFile": practice.get("policyFile"),
        "reportFile": practice.get("reportFile"),
    }


def build_work_queue(
    *,
    catalog_path: Path = ROOM_PROBLEMS_PATH,
    path_board_path: Path = PATH_BOARD_PATH,
    kpdr_tracker_path: Path = KPDR_TRACKER_PATH,
    graph_path: Path = FULL_ROOM_GRAPH_PATH,
) -> dict[str, Any]:
    """Build easiest-first ranked rows + progress summary."""
    catalog = load_problem_catalog(catalog_path)
    problems = list(catalog["problems"])
    path_ids = _load_path_room_ids(path_board_path)
    kpdr_ids = _load_kpdr_room_ids(kpdr_tracker_path)

    # Annotate first (live teleport status), then sort by resulting score.
    annotated = [
        annotate_problem(
            problem,
            path_room_ids=path_ids,
            kpdr_room_ids=kpdr_ids,
            rank=0,
        )
        for problem in problems
    ]
    ordered = sorted(
        annotated,
        key=lambda row: (
            int(row["difficultyScore"]),
            int(row.get("queue", 3)),
            str(row.get("area", "")),
            int(row["roomId"]),
        ),
    )
    rows = [{**row, "rank": index + 1} for index, row in enumerate(ordered)]

    edge_count = 0
    if graph_path.is_file():
        graph = json.loads(graph_path.read_text(encoding="utf-8"))
        edge_count = len(graph.get("edges") or [])
        if not edge_count and isinstance(graph.get("summary"), dict):
            edge_count = int(graph["summary"].get("directedEdgeCount") or 0)

    summary = _summarize(
        rows, edge_count=edge_count, catalog_summary=catalog.get("summary")
    )
    return {
        "schemaVersion": 1,
        "catalogId": "super_metroid_room_work_queue",
        "generatedAt": datetime.now(timezone.utc).isoformat(),
        "source": {
            "roomProblems": str(catalog_path),
            "pathBoard": str(path_board_path) if path_board_path.is_file() else None,
            "kpdrTracker": (
                str(kpdr_tracker_path) if kpdr_tracker_path.is_file() else None
            ),
            "fullRoomGraph": str(graph_path) if graph_path.is_file() else None,
        },
        "unitNote": (
            "Ranked units are the 262 canonical room problems (one per room). "
            f"The full graph has {edge_count or 583} directed edges; those are "
            "topology hops, not separate practice harness entries. This board "
            "is dual-track practice metrics only — not continuous evidence and "
            "not the product work queue. Product next-work: docs/STATUS.md + "
            "`bd ready -l super_metroid`. Continuous segment board: KPDR_TRACKER."
        ),
        "summary": summary,
        "queuePolicy": (catalog.get("queuePolicy") or []),
        "problems": rows,
    }


def _summarize(
    rows: Sequence[Mapping[str, Any]],
    *,
    edge_count: int,
    catalog_summary: Mapping[str, Any] | None,
) -> dict[str, Any]:
    total = len(rows)
    by_queue = Counter(int(r["queue"]) for r in rows)
    by_tier = Counter(str(r["tier"]) for r in rows)
    by_status = Counter(str(r["practiceStatus"]) for r in rows)
    teleport_ready = sum(1 for r in rows if r["teleportReady"])
    run_ready = sum(1 for r in rows if r["runReady"])
    on_path = [r for r in rows if r["onCompletionPath"]]
    non_boss = [r for r in rows if int(r["queue"]) < 4]
    easy_std = [r for r in rows if int(r["queue"]) in {0, 1, 2}]

    def _pct(done: int, denom: int) -> float:
        if denom <= 0:
            return 0.0
        return round(100.0 * done / denom, 1)

    ready_like = sum(
        1 for r in rows if r["practiceStatus"] in _STATUS_DONE or r["runReady"]
    )
    easy_ready = sum(
        1 for r in easy_std if r["practiceStatus"] in _STATUS_DONE or r["runReady"]
    )
    path_ready = sum(
        1 for r in on_path if r["practiceStatus"] in _STATUS_DONE or r["runReady"]
    )
    non_boss_ready = sum(
        1 for r in non_boss if r["practiceStatus"] in _STATUS_DONE or r["runReady"]
    )
    teleport_easy = sum(1 for r in easy_std if r["teleportReady"])

    open_easy = [
        r
        for r in rows
        if int(r["queue"]) in {0, 1}
        and not r["runReady"]
        and str(r["tier"]) != "boss_late"
    ][:25]

    return {
        "problemCount": total,
        "directedEdgeCount": edge_count or None,
        "completionPathRoomCount": len(on_path),
        "queueCounts": {str(k): by_queue[k] for k in sorted(by_queue)},
        "tierCounts": dict(sorted(by_tier.items())),
        "practiceStatusCounts": dict(sorted(by_status.items())),
        "teleportReadyCount": teleport_ready,
        "runReadyCount": run_ready,
        "percentComplete": {
            "allProblemsReady": _pct(ready_like, total),
            "easyAndStandardReady": _pct(easy_ready, len(easy_std)),
            "nonBossReady": _pct(non_boss_ready, len(non_boss)),
            "completionPathReady": _pct(path_ready, len(on_path)),
            "easyAndStandardTeleportReady": _pct(teleport_easy, len(easy_std)),
            "allTeleportReady": _pct(teleport_ready, total),
        },
        "workFocus": {
            "easyAndStandardTotal": len(easy_std),
            "easyAndStandardReady": easy_ready,
            "easyAndStandardTeleportReady": teleport_easy,
            "bossDeferred": by_queue.get(4, 0),
            "toughOrLate": by_queue.get(3, 0),
            "nextOpenEasyProblemIds": [r["problemId"] for r in open_easy],
            "productNextWork": (
                "Continuous tip order is owned by docs/STATUS.md and "
                "beads (`bd ready -l super_metroid`) — not this practice board."
            ),
        },
        "catalogSummary": dict(catalog_summary or {}),
    }


def work_queue_to_csv_rows(payload: Mapping[str, Any]) -> list[dict[str, str]]:
    """Flatten ranked problems for CSV export."""
    field_order = [
        "rank",
        "difficultyScore",
        "queue",
        "tier",
        "problemId",
        "roomIdHex",
        "roomName",
        "area",
        "objective",
        "screens",
        "enemyCount",
        "staticPlanStatus",
        "pathBlocks",
        "onCompletionPath",
        "onKpdrTracker",
        "practiceStatus",
        "teleportReady",
        "runReady",
        "entrySourceRoomIdHex",
        "exitTargetRoomIdHex",
        "difficultyReasons",
        "stateFile",
        "policyFile",
    ]
    rows: list[dict[str, str]] = []
    for problem in payload.get("problems") or []:
        row: dict[str, str] = {}
        for key in field_order:
            value = problem.get(key)
            if key == "difficultyReasons" and isinstance(value, list):
                row[key] = ";".join(str(item) for item in value)
            elif isinstance(value, bool):
                row[key] = "1" if value else "0"
            elif value is None:
                row[key] = ""
            else:
                row[key] = str(value)
        rows.append(row)
    return rows


def work_queue_to_markdown(payload: Mapping[str, Any]) -> str:
    """Human board: summary + top open easy rooms + full count tables."""
    summary = payload["summary"]
    pct = summary["percentComplete"]
    focus = summary["workFocus"]
    lines = [
        "# Room work queue — easiest first",
        "",
        "Isolated room-clear practice board (teleport → policy → promote).",
        "**Not continuous-run evidence** and **not** the product work queue.",
        "Product next-work: `docs/STATUS.md` + `bd ready -l super_metroid`.",
        "Source catalog: `maps/room_problems.json`.",
        "",
        "Regenerate:",
        "",
        "```bash",
        "uv run python snes/super_metroid/scripts/export/room_work_queue.py",
        "```",
        "",
        "## Units",
        "",
        payload.get("unitNote", ""),
        "",
        "## Percent complete (practice harness only)",
        "",
        "| Scope | Ready % |",
        "|-------|--------:|",
        f"| Easy + standard (classes 0–2) | **{pct['easyAndStandardReady']}%** |",
        f"| Non-boss (classes 0–3) | {pct['nonBossReady']}% |",
        f"| All 262 room problems | {pct['allProblemsReady']}% |",
        f"| Completion-path rooms only | {pct['completionPathReady']}% |",
        f"| Easy+standard teleport fixtures | {pct['easyAndStandardTeleportReady']}% |",
        f"| All teleport fixtures | {pct['allTeleportReady']}% |",
        "",
        "## Counts",
        "",
        "| Metric | Count |",
        "|--------|------:|",
        f"| Room problems | {summary['problemCount']} |",
        f"| Directed edges (full graph) | {summary.get('directedEdgeCount') or '—'} |",
        f"| Completion-path rooms | {summary['completionPathRoomCount']} |",
        f"| Teleport-ready (entry `.state`) | {summary['teleportReadyCount']} |",
        f"| Run-ready (state + verified policy) | {summary['runReadyCount']} |",
        f"| Easy+standard total | {focus['easyAndStandardTotal']} |",
        f"| Easy+standard ready | {focus['easyAndStandardReady']} |",
        f"| Tough/late (class 3) | {focus['toughOrLate']} |",
        f"| Boss deferred (class 4) | {focus['bossDeferred']} |",
        "",
        "### By difficulty class (`queue` field)",
        "",
        "| Class | Meaning | Count |",
        "|------:|---------|------:|",
    ]
    meanings = {
        "0": "state + verified policy ready",
        "1": "easy / small rooms",
        "2": "standard traversal",
        "3": "tough / late / unresolved geometry",
        "4": "bosses held for later",
    }
    for queue, count in (summary.get("queueCounts") or {}).items():
        lines.append(f"| {queue} | {meanings.get(str(queue), '')} | {count} |")

    lines += [
        "",
        "### By tier",
        "",
        "| Tier | Count |",
        "|------|------:|",
    ]
    for tier, count in (summary.get("tierCounts") or {}).items():
        lines.append(f"| `{tier}` | {count} |")

    lines += [
        "",
        "## How to work top-down",
        "",
        "1. Export / refresh this board.",
        "2. Bootstrap entry states for easy rooms (door-warp fixtures):",
        "",
        "```bash",
        "uv run python snes/super_metroid/scripts/room/run_problem.py bootstrap --queue 1",
        "```",
        "",
        "3. Scaffold a policy, iterate, then promote on a green isolated run:",
        "",
        "```bash",
        "uv run python snes/super_metroid/scripts/room/run_problem.py scaffold PROBLEM_ID",
        "uv run python snes/super_metroid/scripts/room/run_problem.py teleport PROBLEM_ID",
        "uv run python snes/super_metroid/scripts/room/run_problem.py run PROBLEM_ID --promote",
        "```",
        "",
        "4. Leave class 3 large rooms and class 4 bosses until easy+standard % is solid.",
        "5. Never claim practice greens as continuous tip evidence.",
        "",
        "## Next open easy (practice only)",
        "",
        "| Rank | Score | Room | Problem | Teleport |",
        "|-----:|------:|------|---------|:--------:|",
    ]
    by_id = {r["problemId"]: r for r in payload.get("problems") or []}
    for problem_id in focus.get("nextOpenEasyProblemIds") or []:
        row = by_id.get(problem_id)
        if row is None:
            continue
        lines.append(
            f"| {row['rank']} | {row['difficultyScore']} | "
            f"{row['roomName']} `{row['roomIdHex']}` | `{problem_id}` | "
            f"{'yes' if row['teleportReady'] else 'no'} |"
        )

    lines += [
        "",
        "Full ranked table: `docs/routes/ROOM_WORK_QUEUE.csv` · "
        "machine JSON: `maps/room_work_queue.json`.",
        "",
        f"_Generated {payload.get('generatedAt', '')}_",
        "",
    ]
    return "\n".join(lines)


def export_work_queue(
    *,
    catalog_path: Path = ROOM_PROBLEMS_PATH,
    json_output: Path = DEFAULT_QUEUE_JSON,
    csv_output: Path = DEFAULT_QUEUE_CSV,
    md_output: Path = DEFAULT_QUEUE_MD,
) -> dict[str, Any]:
    """Write JSON + CSV + markdown work queue artifacts."""
    import csv

    payload = build_work_queue(catalog_path=catalog_path)
    json_output.parent.mkdir(parents=True, exist_ok=True)
    csv_output.parent.mkdir(parents=True, exist_ok=True)
    md_output.parent.mkdir(parents=True, exist_ok=True)

    json_output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    rows = work_queue_to_csv_rows(payload)
    if rows:
        with csv_output.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
    md_output.write_text(work_queue_to_markdown(payload), encoding="utf-8")
    return payload
