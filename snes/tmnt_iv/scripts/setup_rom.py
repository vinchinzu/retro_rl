"""Extract and link the shared ROM for this ladder game."""

from __future__ import annotations

from retro_harness.setup_rom_cli import main_ladder_setup_rom

SLUG = "tmnt_iv"


def main() -> None:
    """Wire the shared zip into this game's integration."""
    raise SystemExit(main_ladder_setup_rom(SLUG))


if __name__ == "__main__":
    main()
