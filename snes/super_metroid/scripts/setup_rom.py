"""Link the verified shared Super Metroid ROM into the local integration."""

from __future__ import annotations

import os
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[3]
from super_metroid.paths import INTEGRATION_DIR, SHARED_ROM  # noqa: E402


def main() -> None:
    if not SHARED_ROM.is_file():
        raise FileNotFoundError(SHARED_ROM)
    target = INTEGRATION_DIR / "rom.sfc"
    target.parent.mkdir(parents=True, exist_ok=True)
    rel = Path(os.path.relpath(SHARED_ROM, target.parent))
    if target.exists() or target.is_symlink():
        try:
            same = target.resolve() == SHARED_ROM.resolve()
        except OSError:
            same = False
        if same:
            print(f"ROM ready: {target}")
            return
        # Replace broken or stale relative links after layout moves.
        target.unlink()
    target.symlink_to(rel)
    print(f"ROM ready: {target}")


if __name__ == "__main__":
    main()
