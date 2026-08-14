"""Thin CLI helpers for per-game ``scripts/setup_rom.py``.

``setup_game_rom`` already does extract+link. These wrappers print the
existing ``ROM ready:`` line and resolve ladder slugs.
"""

from __future__ import annotations

from pathlib import Path

from retro_harness.env import setup_game_rom


def setup_and_print(
    *,
    shared_zip: Path,
    game_dir: Path,
    integration_name: str,
    extensions: set[str] | None = None,
) -> Path:
    """Call ``setup_game_rom`` and print the standard ready line."""
    rom = setup_game_rom(
        shared_zip=shared_zip,
        game_dir=game_dir,
        integration_name=integration_name,
        extensions=extensions,
    )
    print(f"ROM ready: {rom}")
    return rom


def main_setup_rom(
    *,
    shared_zip: Path,
    game_dir: Path,
    integration_name: str,
    extensions: set[str] | None = None,
) -> int:
    """CLI-safe wrapper used by NES / one-off ``setup_rom.py`` scripts."""
    setup_and_print(
        shared_zip=shared_zip,
        game_dir=game_dir,
        integration_name=integration_name,
        extensions=extensions,
    )
    return 0


def main_ladder_setup_rom(slug: str) -> int:
    """Resolve a ladder zip + integration and wire the ROM."""
    from retro_harness.ladder import entry_for, shared_rom_zip
    from retro_harness.repo import resolve_game_dir

    entry = entry_for(slug)
    setup_and_print(
        shared_zip=shared_rom_zip(entry),
        game_dir=resolve_game_dir(slug),
        integration_name=entry.integration,
    )
    return 0
