"""Filesystem and integration constants for Super Metroid."""

from __future__ import annotations

from pathlib import Path

GAME = "SuperMetroid-Snes"
GAME_DIR = Path(__file__).resolve().parent
REPO_DIR = GAME_DIR.parent
INTEGRATION_DIR = GAME_DIR / "custom_integrations" / GAME
POLICY_DIR = GAME_DIR / "policies" / "start_to_morph"
EARLY_POLICY_DIR = GAME_DIR / "policies" / "early_game"
RECORDINGS_DIR = GAME_DIR / "recordings"
MODELS_DIR = GAME_DIR / "models"
MAPS_DIR = GAME_DIR / "maps"
SHARED_ROM = REPO_DIR / "roms" / "SuperMetroid.sfc"
