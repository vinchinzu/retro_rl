"""Filesystem locations for the Super Mario Bros. 2 TAS scaffold."""

from __future__ import annotations

from pathlib import Path

GAME_DIR = Path(__file__).resolve().parent
REPO_ROOT = GAME_DIR.parent.parent
GAME = "SuperMarioBros2-Nes"

ROMS_DIR = GAME_DIR / "roms"
ROM_FILENAME = "Super Mario Bros. 2.nes"
ROM_PATH = ROMS_DIR / ROM_FILENAME
SHARED_ROM_ZIP = REPO_ROOT / "roms" / "Nintendo" / "NES" / ("Super Mario Bros. 2.zip")

INTEGRATION_DIR = GAME_DIR / "custom_integrations" / GAME
ROM_SHA1_PATH = INTEGRATION_DIR / "rom.sha"

MOVIES_DIR = GAME_DIR / "movies"
ARTIFACTS_DIR = GAME_DIR / "artifacts"
EVIDENCE_DIR = ARTIFACTS_DIR / "evidence"
EVIDENCE_MANIFEST_PATH = EVIDENCE_DIR / "level1_tas_evidence.json"

# State files stay under the integration directory even though the manifest
# stores portable, game-relative paths.
STATE_ARTIFACTS_DIR = INTEGRATION_DIR / "checkpoints"
STATE_ARTIFACTS_RELATIVE_DIR = Path("custom_integrations") / GAME / "checkpoints"
