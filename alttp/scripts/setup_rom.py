"""Link the shared zelda3.sfc ROM into the ALTTP integration."""

from __future__ import annotations

import hashlib
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from alttp.paths import GAME_DIR, INTEGRATION_DIR, ROMS_DIR, REPO_ROOT

EXPECTED_SHA1 = "6d4f10a8b10e10dbe624cb23cf03b88bb8252973"


def main() -> None:
    """Symlink roms/zelda3.sfc into the game integration."""
    shared = REPO_ROOT / "roms" / "zelda3.sfc"
    if not shared.is_file():
        raise FileNotFoundError(f"Missing shared ROM: {shared}")

    digest = hashlib.sha1(shared.read_bytes()).hexdigest()
    if digest != EXPECTED_SHA1:
        raise RuntimeError(
            f"Unexpected ROM sha1 {digest} (expected {EXPECTED_SHA1})"
        )

    ROMS_DIR.mkdir(parents=True, exist_ok=True)
    local = ROMS_DIR / "zelda3.sfc"
    if local.exists() or local.is_symlink():
        local.unlink()
    local.symlink_to(shared)

    INTEGRATION_DIR.mkdir(parents=True, exist_ok=True)
    rom_link = INTEGRATION_DIR / "rom.sfc"
    if rom_link.exists() or rom_link.is_symlink():
        rom_link.unlink()
    rom_link.symlink_to(Path("../../roms/zelda3.sfc"))

    sha_path = INTEGRATION_DIR / "rom.sha"
    sha_path.write_text(f"{EXPECTED_SHA1}\n", encoding="utf-8")
    print(f"ROM ready: {rom_link} -> {shared}")
    print(f"Game dir: {GAME_DIR}")


if __name__ == "__main__":
    main()
