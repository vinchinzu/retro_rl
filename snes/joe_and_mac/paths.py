"""Filesystem and integration constants for Joe & Mac."""

from __future__ import annotations

from pathlib import Path

GAME_DIR = Path(__file__).resolve().parent
GAME = "JoeAndMac-Snes"
RECORDINGS_DIR = GAME_DIR / "recordings"
STAGE1_STATE = "Stage1"
