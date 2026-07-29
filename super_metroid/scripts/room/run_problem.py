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
from super_metroid.rooms.room_practice import (  # noqa: E402
    capture_room_state,
    ready_problem_ids,
    run_room_problem,
    scaffold_room_policy,
    teleport_room_problem,
)


def _room_id(value: str) -> int:
    return int(value, 0)


def _list_problems(tier: str | None, status: str | None) -> None:
    catalog = load_problem_catalog(ROOM_PROBLEMS_PATH)
    rows = [
        problem
        for problem in catalog["problems"]
        if (tier is None or problem["tier"] == tier)
        and (status is None or problem["practice"]["status"] == status)
    ]
    print(json.dumps(rows, indent=2))


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

    ready_parser = sub.add_parser("ready")
    ready_parser.add_argument("--run", action="store_true")

    capture_parser = sub.add_parser("capture")
    capture_parser.add_argument("problem_id")
    capture_parser.add_argument("source_state", type=Path)

    teleport_parser = sub.add_parser("teleport")
    teleport_parser.add_argument("problem_id")
    teleport_parser.add_argument("--screenshot", type=Path)

    run_parser = sub.add_parser("run")
    run_parser.add_argument("problem_id")
    run_parser.add_argument("--report", type=Path)

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
        _list_problems(args.tier, args.status)
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
        report = run_room_problem(args.problem_id, report_path=args.report)
        print(json.dumps(report, indent=2))
        if not report["success"]:
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
