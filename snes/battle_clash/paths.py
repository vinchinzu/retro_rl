"""Filesystem and integration constants for Battle Clash."""

from __future__ import annotations

from pathlib import Path

GAME_DIR = Path(__file__).resolve().parent
GAME = "BattleClash-Snes"
RECORDINGS_DIR = GAME_DIR / "recordings"
