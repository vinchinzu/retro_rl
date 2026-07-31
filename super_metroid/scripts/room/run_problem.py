#!/usr/bin/env python3
"""List, teleport to, capture, route, and run isolated room problems."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from super_metroid.paths import (  # noqa: E402
    FULL_ROOM_GRAPH_PATH,
    ROOM_PROBLEMS_PATH,
)
from super_metroid.rooms.room_graph import (  # noqa: E402
    load_problem_catalog,
    shortest_room_path,
)
from super_metroid.rooms.entry_bootstrap import (  # noqa: E402
    bootstrap_entry_state,
    bootstrap_entry_states,
)
from super_metroid.rooms.room_practice import (  # noqa: E402
    capture_room_state,
    promote_room_policy,
    ready_problem_ids,
    run_room_problem,
    scaffold_room_policy,
    teleport_room_problem,
)
from super_metroid.rooms.work_queue import (  # noqa: E402
    build_work_queue,
    difficulty_score,
)


def _room_id(value: str) -> int:
    return int(value, 0)


def _list_problems(
    tier: str | None,
    status: str | None,
    *,
    queue: int | None = None,
    easiest_first: bool = False,
    limit: int | None = None,
) -> None:
    catalog = load_problem_catalog(ROOM_PROBLEMS_PATH)
    rows = [
        problem
        for problem in catalog["problems"]
        if (tier is None or problem["tier"] == tier)
        and (status is None or problem["practice"]["status"] == status)
        and (queue is None or int(problem.get("queue", 3)) == queue)
    ]
    if easiest_first:
        rows = sorted(
            rows,
            key=lambda problem: (
                difficulty_score(problem),
                int(problem.get("queue", 3)),
                int(problem["roomId"]),
            ),
        )
        rows = [
            {**problem, "difficultyScore": difficulty_score(problem)}
            for problem in rows
        ]
    if limit is not None:
        rows = rows[: max(0, limit)]
    print(json.dumps(rows, indent=2))


def _queue_board(limit: int | None) -> None:
    payload = build_work_queue()
    summary = payload["summary"]
    problems = payload["problems"]
    if limit is not None:
        problems = problems[: max(0, limit)]
    print(
        json.dumps(
            {
                "summary": summary,
                "problems": problems,
            },
            indent=2,
        )
    )


def _route(source: int, target: int, capabilities: list[str]) -> None:
    graph = json.loads(FULL_ROOM_GRAPH_PATH.read_text(encoding="utf-8"))
    path = shortest_room_path(graph, source, target, capabilities)
    if path is None:
        raise SystemExit(
            f"no route 0x{source:04X}->0x{target:04X} with {sorted(capabilities)}"
        )
    room_names = {int(room["roomId"]): room["name"] for room in graph["rooms"]}
    route = [
        {"roomIdHex": f"0x{source:04X}", "roomName": room_names[source]},
        *(
            {
                "roomIdHex": edge["target"]["roomIdHex"],
                "roomName": room_names[int(edge["target"]["roomId"])],
                "viaEdge": edge["edgeId"],
                "requires": edge["requires"],
                "localRequirements": edge["localRequirements"],
            }
            for edge in path
        ),
    ]
    print(json.dumps(route, indent=2))


def _scaffold(
    problem_id: str | None,
    *,
    all_problems: bool,
    output_dir: Path | None,
    force: bool,
) -> None:
    catalog = load_problem_catalog(ROOM_PROBLEMS_PATH)
    if all_problems:
        problem_ids = [str(problem["problemId"]) for problem in catalog["problems"]]
    elif problem_id is not None:
        problem_ids = [problem_id]
    else:
        raise SystemExit("scaffold requires a problem_id or --all")

    results = []
    for identifier in problem_ids:
        output = output_dir / f"{identifier}.json" if output_dir is not None else None
        try:
            results.append(
                scaffold_room_policy(
                    identifier,
                    output_path=output,
                    overwrite=force,
                )
            )
        except FileExistsError as exc:
            results.append(
                {
                    "problemId": identifier,
                    "status": "existing_skipped",
                    "message": str(exc),
                }
            )
    print(json.dumps(results, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    list_parser = sub.add_parser("list")
    list_parser.add_argument("--tier")
    list_parser.add_argument("--status")
    list_parser.add_argument("--queue", type=int)
    list_parser.add_argument(
        "--easiest-first",
        action="store_true",
        help="sort by rough difficulty score (easiest first)",
    )
    list_parser.add_argument("--limit", type=int)

    queue_parser = sub.add_parser(
        "queue",
        help="print easiest-first work board summary (+ ranked problems)",
    )
    queue_parser.add_argument("--limit", type=int, default=30)

    ready_parser = sub.add_parser("ready")
    ready_parser.add_argument("--run", action="store_true")

    capture_parser = sub.add_parser("capture")
    capture_parser.add_argument("problem_id")
    capture_parser.add_argument("source_state", type=Path)

    bootstrap_parser = sub.add_parser(
        "bootstrap",
        help="door-warp a development entry .state for teleport practice",
    )
    bootstrap_parser.add_argument(
        "problem_id",
        nargs="?",
        help="single problem id; omit with --queue for a bulk slice",
    )
    bootstrap_parser.add_argument(
        "--queue",
        type=int,
        default=None,
        help="bootstrap all problems in this difficulty queue (e.g. 1 = easy)",
    )
    bootstrap_parser.add_argument("--max", type=int, default=None)
    bootstrap_parser.add_argument("--overwrite", action="store_true")
    bootstrap_parser.add_argument("--boot-state", type=Path)
    bootstrap_parser.add_argument(
        "--boot-idle-frames",
        type=int,
        default=0,
        help="extra idle frames before door-warp (recorded for RNG re-rolls)",
    )

    teleport_parser = sub.add_parser("teleport")
    teleport_parser.add_argument("problem_id")
    teleport_parser.add_argument("--screenshot", type=Path)

    run_parser = sub.add_parser("run")
    run_parser.add_argument("problem_id")
    run_parser.add_argument("--report", type=Path)
    run_parser.add_argument(
        "--promote",
        action="store_true",
        help="on success, mark policy verified_development_state (sha-gated)",
    )

    promote_parser = sub.add_parser(
        "promote",
        help="mark policy verified from an existing green report (sha-gated)",
    )
    promote_parser.add_argument("problem_id")
    promote_parser.add_argument("--report", type=Path)
    promote_parser.add_argument(
        "--allow-sha-mismatch",
        action="store_true",
        help="skip state/policy sha match against the report (not recommended)",
    )

    scaffold_parser = sub.add_parser("scaffold")
    scaffold_parser.add_argument("problem_id", nargs="?")
    scaffold_parser.add_argument("--all", action="store_true")
    scaffold_parser.add_argument("--output-dir", type=Path)
    scaffold_parser.add_argument("--force", action="store_true")

    route_parser = sub.add_parser("route")
    route_parser.add_argument("source", type=_room_id)
    route_parser.add_argument("target", type=_room_id)
    route_parser.add_argument("--capability", action="append", default=[])

    args = parser.parse_args()
    if args.command == "list":
        _list_problems(
            args.tier,
            args.status,
            queue=args.queue,
            easiest_first=args.easiest_first,
            limit=args.limit,
        )
    elif args.command == "queue":
        _queue_board(args.limit)
    elif args.command == "bootstrap":
        if args.problem_id:
            print(
                json.dumps(
                    bootstrap_entry_state(
                        args.problem_id,
                        boot_state=args.boot_state,
                        overwrite=args.overwrite,
                        boot_idle_frames=args.boot_idle_frames,
                    ),
                    indent=2,
                )
            )
        elif args.queue is not None:
            print(
                json.dumps(
                    bootstrap_entry_states(
                        queue=args.queue,
                        max_problems=args.max,
                        overwrite=args.overwrite,
                        boot_state=args.boot_state,
                        boot_idle_frames=args.boot_idle_frames,
                    ),
                    indent=2,
                )
            )
        else:
            raise SystemExit("bootstrap requires problem_id or --queue N")
    elif args.command == "ready":
        identifiers = ready_problem_ids()
        if not args.run:
            print("\n".join(identifiers))
        else:
            reports = [run_room_problem(problem_id) for problem_id in identifiers]
            print(json.dumps(reports, indent=2))
            if not all(report["success"] for report in reports):
                raise SystemExit(1)
    elif args.command == "capture":
        print(
            json.dumps(
                capture_room_state(args.problem_id, args.source_state),
                indent=2,
            )
        )
    elif args.command == "teleport":
        print(
            json.dumps(
                teleport_room_problem(
                    args.problem_id,
                    screenshot_path=args.screenshot,
                ),
                indent=2,
            )
        )
    elif args.command == "run":
        report = run_room_problem(
            args.problem_id,
            report_path=args.report,
            promote=args.promote,
        )
        print(json.dumps(report, indent=2))
        if not report["success"]:
            raise SystemExit(1)
        if args.promote and not report.get("promoted"):
            raise SystemExit(1)
    elif args.command == "promote":
        result = promote_room_policy(
            args.problem_id,
            report_path=args.report,
            require_matching_sha=not args.allow_sha_mismatch,
        )
        print(json.dumps(result, indent=2))
        if not result.get("promoted"):
            raise SystemExit(1)
    elif args.command == "scaffold":
        _scaffold(
            args.problem_id,
            all_problems=args.all,
            output_dir=args.output_dir,
            force=args.force,
        )
    else:
        _route(args.source, args.target, args.capability)


if __name__ == "__main__":
    main()
