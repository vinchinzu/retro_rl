"""Filesystem constants for Super Double Dragon."""

from __future__ import annotations

from pathlib import Path

GAME_DIR = Path(__file__).resolve().parent
REPO_ROOT = GAME_DIR.parent
INTEGRATION = "SuperDoubleDragon-Snes"
GAME = INTEGRATION
INTEGRATION_DIR = GAME_DIR / "custom_integrations" / INTEGRATION
RECORDINGS_DIR = GAME_DIR / "recordings"
ROMS_DIR = GAME_DIR / "roms"
DOCS_DIR = GAME_DIR / "docs"

STAGE1_STATE = "Stage1"
STAGE1_FIRST_CLEAR_STATE = "Stage1_FirstSegment_Clear"
STAGE1_AREA2_STATE = "Stage1_Area2"
STAGE2_STATE = "Stage2"
STAGE3_STATE = "Stage3"
STAGE4_STATE = "Stage4"
STAGE5_STATE = "Stage5"
