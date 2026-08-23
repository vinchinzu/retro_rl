#!/usr/bin/env python3
"""Point the custom integration at repo ``roms/Mortal Kombat.smc``."""

from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[3]
for _p in (_ROOT, _ROOT / "snes"):
    _t = str(_p)
    if _t not in sys.path:
        sys.path.insert(0, _t)

from mortal_kombat.paths import INTEGRATION_DIR  # noqa: E402

ROM_CANDIDATES = (
    _ROOT / "roms" / "Mortal Kombat.smc",
    _ROOT / "roms" / "Mortal Kombat.zip",
)


def main() -> int:
    target = INTEGRATION_DIR / "rom.sfc"
    for src in ROM_CANDIDATES:
        if src.exists():
            if target.is_symlink() or target.exists():
                target.unlink()
            target.symlink_to(src)
            print(f"ROM ready: {target} -> {src}")
            return 0
    print("No Mortal Kombat ROM under roms/", file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
