#!/usr/bin/env python3
"""Export the full room graph and one canonical problem per room."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from super_metroid.paths import (  # noqa: E402
    FULL_ROOM_GRAPH_PATH,
    INTEGRATION_DIR,
    ROOM_CLEAR_POLICY_DIR,
    ROOM_PROBLEMS_PATH,
)
from super_metroid.rooms.room_graph import export_full_room_catalog  # noqa: E402


def _editor_nav() -> Path:
    configured = os.environ.get("SUPER_METROID_EDITOR_NAV")
    if configured:
        return Path(configured).expanduser()
    return (
        ROOT.parent
        / "snes_editor"
        / "super_metroid_rl"
        / "super_metroid_editor"
        / "export"
        / "sm_nav"
        / "nav_graph.json"
    )


def _reference_root() -> Path:
    configured = os.environ.get("SUPER_METROID_JSON_DATA")
    if configured:
        return Path(configured).expanduser()
    return ROOT.parent / "snes_editor" / "super_metroid_rl" / "refs" / "sm-json-data"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--editor-nav", type=Path, default=_editor_nav())
    parser.add_argument("--reference-root", type=Path, default=_reference_root())
    parser.add_argument(
        "--legacy-route",
        type=Path,
        default=ROOT / "super_metroid/maps/legacy/full_game_route.json",
    )
    parser.add_argument(
        "--graph-output",
        type=Path,
        default=FULL_ROOM_GRAPH_PATH,
    )
    parser.add_argument(
        "--problems-output",
        type=Path,
        default=ROOM_PROBLEMS_PATH,
    )
    args = parser.parse_args()
    graph, catalog = export_full_room_catalog(
        editor_nav=args.editor_nav,
        reference_root=args.reference_root,
        legacy_route=args.legacy_route,
        graph_output=args.graph_output,
        problems_output=args.problems_output,
        states_dir=INTEGRATION_DIR,
        policy_dir=ROOM_CLEAR_POLICY_DIR,
    )
    print(
        f"wrote {args.graph_output}: {graph['summary']['roomCount']} rooms, "
        f"{graph['summary']['physicalConnectionCount']} connections, "
        f"{graph['summary']['directedEdgeCount']} directed edges"
    )
    print(
        f"wrote {args.problems_output}: "
        f"{catalog['summary']['problemCount']} room problems"
    )


if __name__ == "__main__":
    main()
