"""Link shared Super Metroid + ALttP ROMs into smz3/roms/."""

from __future__ import annotations

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from smz3.paths import (  # noqa: E402
    LOCAL_SM_ROM,
    LOCAL_Z3_ROM,
    ROMS_DIR,
    SHARED_SM_ROM,
    SHARED_Z3_ROM,
)


def _link(shared: Path, local: Path) -> None:
    if not shared.is_file():
        raise FileNotFoundError(f"Missing shared ROM: {shared}")
    ROMS_DIR.mkdir(parents=True, exist_ok=True)
    if local.exists() or local.is_symlink():
        if local.resolve() == shared.resolve():
            print(f"OK: {local} -> {shared}")
            return
        local.unlink()
    local.symlink_to(shared)
    print(f"Linked: {local} -> {shared}")


def main() -> None:
    _link(SHARED_SM_ROM, LOCAL_SM_ROM)
    _link(SHARED_Z3_ROM, LOCAL_Z3_ROM)
    print("SMZ3 vanilla ROMs ready (combo ROM built per seed).")


if __name__ == "__main__":
    main()
