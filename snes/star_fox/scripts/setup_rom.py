"""Extract and link the shared Star Fox ROM."""

from __future__ import annotations

from pathlib import Path

from retro_harness.setup_rom_cli import main_setup_rom

GAME_DIR = Path(__file__).resolve().parent.parent
REPO_ROOT = GAME_DIR.parent


def main() -> None:
    """Wire the shared USA Rev 2 ROM into the custom integration."""
    raise SystemExit(
        main_setup_rom(
            shared_zip=REPO_ROOT / "roms" / "Super Nintendo" / "Star Fox.zip",
            game_dir=GAME_DIR,
            integration_name="StarFox-Snes",
        )
    )


if __name__ == "__main__":
    main()
