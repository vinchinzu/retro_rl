#!/usr/bin/env python3
"""Rebuild Map Rando / sm-json-data room name catalog under maps/.

Source of truth: ``refs/sm-json-data`` (same names as https://maprando.com/logic).

```bash
uv run python snes/super_metroid/scripts/export/maprando_catalog.py
uv run python snes/super_metroid/scripts/export/maprando_catalog.py --summary
```
"""

from __future__ import annotations

import argparse
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]

from super_metroid.rooms.canonical_names import (  # noqa: E402
    MAPRANDO_CATALOG_PATH,
    MAPRANDO_NAMES_PATH,
    clear_name_cache,
    write_catalog,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--catalog",
        type=Path,
        default=MAPRANDO_CATALOG_PATH,
        help="Output catalog path",
    )
    parser.add_argument(
        "--names",
        type=Path,
        default=MAPRANDO_NAMES_PATH,
        help="Output names index path",
    )
    parser.add_argument(
        "--summary",
        action="store_true",
        help="Print area counts after write",
    )
    args = parser.parse_args(argv)

    catalog_path, names_path, catalog = write_catalog(
        catalog_path=args.catalog,
        names_path=args.names,
    )
    clear_name_cache()
    summary = catalog.get("summary") or {}
    print(
        f"wrote {catalog_path} rooms={summary.get('roomCount')} "
        f"and {names_path}"
    )
    if args.summary:
        for area, count in sorted((summary.get("areaCounts") or {}).items()):
            print(f"  {area}: {count}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
