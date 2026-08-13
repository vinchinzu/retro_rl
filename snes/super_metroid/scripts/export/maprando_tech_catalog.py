#!/usr/bin/env python3
"""Rebuild Map Rando / sm-json-data tech catalog under maps/.

Tech definitions: ``refs/sm-json-data/tech.json``
Difficulty tiers: embedded from https://maprando.com/logic
Bot builder status: ``super_metroid.rooms.tech_catalog``

```bash
uv run python snes/super_metroid/scripts/export/maprando_tech_catalog.py
uv run python snes/super_metroid/scripts/export/maprando_tech_catalog.py --summary
uv run python snes/super_metroid/scripts/export/maprando_tech_catalog.py --builders
```
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
_SNES_IMPORT_ROOT = Path(__file__).resolve().parents[3]
for _p in (ROOT, _SNES_IMPORT_ROOT):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from super_metroid.rooms.tech_catalog import (  # noqa: E402
    MAPRANDO_TECH_CATALOG_PATH,
    builder_coverage_summary,
    clear_tech_cache,
    write_catalog,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--catalog",
        type=Path,
        default=MAPRANDO_TECH_CATALOG_PATH,
        help="Output catalog path",
    )
    parser.add_argument(
        "--summary",
        action="store_true",
        help="Print difficulty + builder counts",
    )
    parser.add_argument(
        "--builders",
        action="store_true",
        help="Print Implicit/Basic/Medium bot coverage",
    )
    args = parser.parse_args(argv)

    path, payload = write_catalog(catalog_path=args.catalog)
    clear_tech_cache()
    counts = payload.get("counts") or {}
    print(
        f"wrote {path} techs={counts.get('total')} "
        f"core={counts.get('builderCore')} try={counts.get('builderTry')}"
    )
    if args.summary:
        by_diff = counts.get("byDifficulty") or {}
        for diff, n in sorted(by_diff.items(), key=lambda kv: (-kv[1], kv[0])):
            print(f"  {diff}: {n}")
        print(
            f"  bot green={counts.get('botGreen')} "
            f"partial={counts.get('botPartial')} "
            f"missing={counts.get('botMissing')}"
        )
    if args.builders:
        cov = builder_coverage_summary()
        print(f"builder targets: {cov['total']}")
        for status in ("green", "partial", "missing"):
            names = cov.get(status) or []
            print(f"  [{status}] ({len(names)})")
            for name in names:
                print(f"    {name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
