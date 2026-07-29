"""Export the typed room graph as a JSON navigation artifact."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from super_metroid.paths import MAPS_DIR  # noqa: E402
from super_metroid.progression import (  # noqa: E402
    EARLY_GAME_GRAPH,
    START_TO_MORPH_GRAPH,
    START_TO_SPORE_SPAWN_GRAPH,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--graph",
        choices=(
            "start_to_morph",
            "start_to_bomb_torizo",
            "start_to_spore_spawn",
        ),
        default="start_to_spore_spawn",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
    )
    args = parser.parse_args()
    graphs = {
        "start_to_morph": START_TO_MORPH_GRAPH,
        "start_to_bomb_torizo": EARLY_GAME_GRAPH,
        "start_to_spore_spawn": START_TO_SPORE_SPAWN_GRAPH,
    }
    graph = graphs[args.graph]
    output = args.output or MAPS_DIR / f"{args.graph}_graph.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = graph.to_dict()
    output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"graph: {output}")


if __name__ == "__main__":
    main()
