"""Extract and link the shared NES ROM for this game."""

from __future__ import annotations

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from zelda_i.paths import GAME, GAME_DIR, SHARED_ROM_ZIP
from snes_oneshot.rom_setup import NES_EXTENSIONS, setup_game_rom


def main() -> None:
    """Wire the shared zip into this game's integration."""
    rom = setup_game_rom(
        shared_zip=SHARED_ROM_ZIP,
        game_dir=GAME_DIR,
        integration_name=GAME,
        extensions=NES_EXTENSIONS,
    )
    print(f"ROM ready: {rom}")


if __name__ == "__main__":
    main()
