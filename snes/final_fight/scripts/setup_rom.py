"""Extract and link the shared Final Fight ROM."""

from __future__ import annotations

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from retro_harness.ladder import entry_for, shared_rom_zip
from retro_harness.env import setup_game_rom

SLUG = "final_fight"


def main() -> None:
    """Wire the shared zip into this game's integration."""
    entry = entry_for(SLUG)
    game_dir = Path(__file__).resolve().parent.parent
    rom = setup_game_rom(
        shared_zip=shared_rom_zip(entry),
        game_dir=game_dir,
        integration_name=entry.integration,
    )
    print(f"ROM ready: {rom}")


if __name__ == "__main__":
    main()
