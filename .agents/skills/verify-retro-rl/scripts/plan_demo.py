#!/usr/bin/env python3
"""Drive the public adventure planner on the morph-door fixture."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
sys.path[:0] = [str(ROOT), str(ROOT / "snes"), str(ROOT / "nes")]

from retro_harness.adventure import (  # noqa: E402
    GraphEdge,
    PlanRequest,
    inventory_aware_path,
    plan,
)


def main() -> int:
    edges = (
        GraphEdge("start", "item", edge_id="collect", acquires={"morph"}),
        GraphEdge("item", "goal", edge_id="open", requires={"morph"}),
    )
    result = plan(PlanRequest(edges, "start", "goal"))
    compatible = inventory_aware_path(edges, "start", "goal")
    record = result.to_record()
    payload = {
        "status": record["status"],
        "path_edge_ids": record["path_edge_ids"],
        "total_cost": record["total_cost"],
        "inventory_aware_path_ids": [edge.edge_id for edge in compatible],
        "found": result.found,
    }
    print(json.dumps(payload, indent=2))
    return 0 if result.found and payload["path_edge_ids"] == ["collect", "open"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
