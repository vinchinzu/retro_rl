"""Export the typed room graph as a JSON navigation artifact."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[4]
from super_metroid.paths import MAPS_DIR  # noqa: E402
from super_metroid.progression import (  # noqa: E402
    EARLY_GAME_GRAPH,
    MORPH_GRAPH,
    SPORE_GRAPH,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--graph",
        choices=(
            "morph",
            "bombs",
            "spore",
        ),
        default="spore",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
    )
    args = parser.parse_args()
    graphs = {
        "morph": MORPH_GRAPH,
        "bombs": EARLY_GAME_GRAPH,
        "spore": SPORE_GRAPH,
    }
    graph = graphs[args.graph]
    output = args.output or MAPS_DIR / f"{args.graph}_graph.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = graph.to_dict()
    output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"graph: {output}")


if __name__ == "__main__":
    main()
