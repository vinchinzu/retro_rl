"""Extract and link the shared Star Fox ROM."""

from __future__ import annotations

from pathlib import Path
import sys

GAME_DIR = Path(__file__).resolve().parent.parent
REPO_ROOT = GAME_DIR.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from snes_oneshot.rom_setup import setup_game_rom


def main() -> None:
    """Wire the shared USA Rev 2 ROM into the custom integration."""
    rom = setup_game_rom(
        shared_zip=REPO_ROOT / "roms" / "Super Nintendo" / "Star Fox.zip",
        game_dir=GAME_DIR,
        integration_name="StarFox-Snes",
    )
    print(f"ROM ready: {rom}")


if __name__ == "__main__":
    main()
