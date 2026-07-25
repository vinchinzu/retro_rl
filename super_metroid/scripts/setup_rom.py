"""Link the verified shared Super Metroid ROM into the local integration."""

from __future__ import annotations

from pathlib import Path
import sys

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from super_metroid.paths import INTEGRATION_DIR, SHARED_ROM  # noqa: E402


def main() -> None:
    if not SHARED_ROM.is_file():
        raise FileNotFoundError(SHARED_ROM)
    target = INTEGRATION_DIR / "rom.sfc"
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists() or target.is_symlink():
        if target.resolve() == SHARED_ROM.resolve():
            print(f"ROM ready: {target}")
            return
        raise FileExistsError(f"refusing to replace {target}")
    target.symlink_to(Path("../../..") / "roms" / SHARED_ROM.name)
    print(f"ROM ready: {target}")


if __name__ == "__main__":
    main()
