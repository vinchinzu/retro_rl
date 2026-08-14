"""Extract and link the shared NES ROM for this game."""

from __future__ import annotations

from tmnt_iii.paths import GAME, GAME_DIR, SHARED_ROM_ZIP
from retro_harness.env import NES_EXTENSIONS
from retro_harness.setup_rom_cli import main_setup_rom


def main() -> None:
    """Wire the shared zip into this game's integration."""
    raise SystemExit(
        main_setup_rom(
            shared_zip=SHARED_ROM_ZIP,
            game_dir=GAME_DIR,
            integration_name=GAME,
            extensions=NES_EXTENSIONS,
        )
    )


if __name__ == "__main__":
    main()
