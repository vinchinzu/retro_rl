#!/usr/bin/env python3
"""Pre-calculate the post-Torizo route from a Super Metroid editor export."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from super_metroid.map_planning import EditorNavigationGraph, sha256_file  # noqa: E402
from super_metroid.paths import MAPS_DIR, SHARED_ROM  # noqa: E402
from super_metroid.routes.spore_spawn_route import (  # noqa: E402
    POST_TORIZO_CAPABILITIES,
    POST_TORIZO_ROUTE_PATCHES,
    POST_TORIZO_TO_SPORE_SPAWN,
)


def _default_editor_nav() -> Path:
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


def _default_reference_route() -> Path:
    return (
        ROOT.parent
        / "snes_editor"
        / "super_metroid_rl"
        / "maps"
        / "spore_spawn_practice_route.json"
    )


def export_plan(
    editor_nav: Path,
    output: Path,
    *,
    reference_route: Path | None = None,
) -> dict[str, object]:
    base_graph = EditorNavigationGraph.load(editor_nav)
    graph = base_graph.add_patches(POST_TORIZO_ROUTE_PATCHES)
    planned = graph.plan_legs(
        POST_TORIZO_TO_SPORE_SPAWN,
        initial_capabilities=POST_TORIZO_CAPABILITIES,
    )
    reference = reference_route.resolve() if reference_route else None
    payload: dict[str, object] = {
        "schemaVersion": 1,
        "planId": "post_torizo_to_spore_spawn",
        "status": "planned_not_continuous",
        "acceptanceWarning": (
            "This is a pre-calculated route. Editor edges and route patches "
            "are not continuous-run evidence."
        ),
        "generatedAt": datetime.now(timezone.utc).isoformat(),
        "source": {
            "editorNavPath": str(base_graph.source_path),
            "editorNavSha256": base_graph.source_sha256,
            "editorRoomCount": len(base_graph.rooms),
            "editorEdgeCount": len(base_graph.edges),
            "referenceRoutePath": str(reference) if reference else None,
            "referenceRouteSha256": (
                sha256_file(reference)
                if reference is not None and reference.is_file()
                else None
            ),
            "romPath": str(SHARED_ROM.resolve()),
            "romSha256": sha256_file(SHARED_ROM),
        },
        "initialCapabilities": sorted(POST_TORIZO_CAPABILITIES),
        "routePatches": [
            patch.as_edge().to_dict() for patch in POST_TORIZO_ROUTE_PATCHES
        ],
        "roomPath": [
            f"0x{POST_TORIZO_TO_SPORE_SPAWN[0].source_room_id:04X}",
            *(f"0x{item.leg.target_room_id:04X}" for item in planned),
        ],
        "legs": [item.to_dict(graph.rooms) for item in planned],
        "terminal": {
            "roomId": 0x9B5B,
            "roomIdHex": "0x9B5B",
            "condition": (
                "first ordinary-gameplay frame after natural Spore Spawn defeat"
            ),
            "requiredCapability": "spore_spawn_defeated",
        },
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--editor-nav", type=Path, default=_default_editor_nav())
    parser.add_argument(
        "--reference-route",
        type=Path,
        default=_default_reference_route(),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=MAPS_DIR / "post_torizo_to_spore_spawn_plan.json",
    )
    args = parser.parse_args()
    payload = export_plan(
        args.editor_nav,
        args.output,
        reference_route=args.reference_route,
    )
    print(
        f"wrote {args.output} with {len(payload['legs'])} planned legs "
        f"from {payload['source']['editorEdgeCount']} editor edges"
    )


if __name__ == "__main__":
    main()

