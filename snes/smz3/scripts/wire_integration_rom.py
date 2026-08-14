"""Point SMZ3-Snes/rom.sfc at a seed combo ROM and refresh rom.sha.

  uv run python smz3/scripts/wire_integration_rom.py
  uv run python smz3/scripts/wire_integration_rom.py smz3/seeds/test_seed/smz3.sfc
"""

from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path

from smz3.paths import INTEGRATION_DIR, TEST_SEED_DIR  # noqa: E402

def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "rom",
        nargs="?",
        type=Path,
        default=TEST_SEED_DIR / "smz3.sfc",
        help="Combo ROM path (default: test seed)",
    )
    args = parser.parse_args(argv)
    rom = args.rom.resolve()
    if not rom.is_file():
        print(f"Missing ROM: {rom}", file=sys.stderr)
        return 1

    INTEGRATION_DIR.mkdir(parents=True, exist_ok=True)
    link = INTEGRATION_DIR / "rom.sfc"
    if link.exists() or link.is_symlink():
        link.unlink()
    # Relative symlink from integration dir → rom for portability.
    try:
        rel = Path(os_relpath(rom, INTEGRATION_DIR))
    except Exception:
        rel = rom
    link.symlink_to(rel)

    digest = hashlib.sha1(rom.read_bytes()).hexdigest()
    (INTEGRATION_DIR / "rom.sha").write_text(f"{digest}\n", encoding="utf-8")
    print(f"rom.sfc -> {link.resolve()}")
    print(f"rom.sha = {digest}")
    return 0

def os_relpath(target: Path, start: Path) -> str:
    import os

    return os.path.relpath(target, start)

if __name__ == "__main__":
    raise SystemExit(main())
