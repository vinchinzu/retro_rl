"""Filesystem and integration constants for Pilotwings."""

from __future__ import annotations

from pathlib import Path

GAME_DIR = Path(__file__).resolve().parent
REPO_ROOT = GAME_DIR.parent
INTEGRATION = "Pilotwings-Snes"
GAME = INTEGRATION
INTEGRATION_DIR = GAME_DIR / "custom_integrations" / INTEGRATION
RECORDINGS_DIR = GAME_DIR / "recordings"
ROMS_DIR = GAME_DIR / "roms"
DOCS_DIR = GAME_DIR / "docs"
LESSON1_PLANE_STATE = "Lesson1Plane"
