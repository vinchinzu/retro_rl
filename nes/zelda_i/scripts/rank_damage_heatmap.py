"""Rank Survival assist damage_by_location. No ROM. Does not write files.

    uv run python nes/zelda_i/scripts/rank_damage_heatmap.py path/to.json [more.json]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_NES = Path(__file__).resolve().parents[2]
if str(_NES) not in sys.path:
    sys.path.insert(0, str(_NES))

from zelda_i.route.heatmap import format_heatmap_table, rank_report_paths


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "reports",
        nargs="+",
        type=Path,
        help="Survival assist JSON report(s)",
    )
    args = parser.parse_args(argv)
    missing = [path for path in args.reports if not path.is_file()]
    if missing:
        for path in missing:
            print(f"missing report: {path}", file=sys.stderr)
        return 2
    print(format_heatmap_table(rank_report_paths(args.reports)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
