#!/usr/bin/env python3
"""Regenerate ``maps/practice_repertoire.json`` from sm_practice_hack menus.

Fetches preset ``*_menu.asm`` / ``*_data.asm`` from GitHub (or uses ``--cache``)
and rebuilds the full practice-hack repertoire catalog used by
``super_metroid.practice_repertoire``.

```bash
uv run python snes/super_metroid/scripts/export/practice_repertoire.py
uv run python snes/super_metroid/scripts/export/practice_repertoire.py --cache /tmp/sm_practice/presets
```
"""

from __future__ import annotations

import argparse
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[4]

from super_metroid.paths import (  # noqa: E402
    PRACTICE_REPERTOIRE_PATH,
    SHARED_PRACTICE_ROM,
)
from super_metroid.practice_repertoire.export_catalog import (  # noqa: E402
    DEFAULT_CACHE,
    UPSTREAM_COMMIT,
    build_catalog,
    ensure_cache,
    write_catalog,
)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--cache",
        type=Path,
        default=DEFAULT_CACHE,
        help="directory of *_menu.asm / *_data.asm",
    )
    p.add_argument(
        "--out",
        type=Path,
        default=PRACTICE_REPERTOIRE_PATH,
        help="output JSON path",
    )
    p.add_argument(
        "--mainmenu",
        type=Path,
        default=None,
        help="upstream src/mainmenu.asm (auto-detected beside --cache)",
    )
    p.add_argument(
        "--practice-rom",
        type=Path,
        default=SHARED_PRACTICE_ROM,
        help="pinned practice ROM used to resolve Save Stations records",
    )
    p.add_argument("--source-commit", default=UPSTREAM_COMMIT)
    p.add_argument(
        "--no-fetch",
        action="store_true",
        help="do not download missing asm files",
    )
    args = p.parse_args(argv)
    if not args.no_fetch:
        ensure_cache(args.cache)
    catalog = build_catalog(
        args.cache,
        mainmenu=args.mainmenu,
        practice_rom=args.practice_rom,
        source_commit=args.source_commit,
    )
    write_catalog(catalog, args.out)
    print(
        f"wrote {args.out} sessions={len(catalog['sessions'])} "
        f"teleports={len(catalog['teleports'])} cats={len(catalog['categories'])}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
