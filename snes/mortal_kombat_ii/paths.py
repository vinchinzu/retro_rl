"""Game-local paths for Mortal Kombat II SNES."""

from __future__ import annotations

from pathlib import Path

from retro_harness.repo import resolve_game_dir

GAME_ID = "MortalKombatII-Snes"
GAME_DIR = resolve_game_dir("mortal_kombat_ii")
INTEGRATION_DIR = GAME_DIR / "custom_integrations" / GAME_ID
MODEL_DIR = GAME_DIR / "models"
FIGHT_LIUKANG = "Fight_LiuKang"


def state_file(name: str) -> Path:
    """Return the custom-integration save-state path for ``name``."""
    filename = name if name.endswith(".state") else f"{name}.state"
    return INTEGRATION_DIR / filename
