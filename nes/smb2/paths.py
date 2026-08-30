"""Filesystem locations for the Super Mario Bros. 2 TAS scaffold."""

from __future__ import annotations

from pathlib import Path

from retro_harness.game_layout import game_paths

_paths = game_paths(__file__, "SuperMarioBros2-Nes")
GAME_DIR = _paths.game_dir
REPO_ROOT = _paths.repo_root
INTEGRATION = _paths.integration
GAME = INTEGRATION
INTEGRATION_DIR = _paths.integration_dir
RECORDINGS_DIR = _paths.recordings_dir
ROMS_DIR = _paths.roms_dir
DOCS_DIR = _paths.docs_dir

ROM_FILENAME = "Super Mario Bros. 2.nes"
ROM_PATH = ROMS_DIR / ROM_FILENAME
SHARED_ROM_ZIP = REPO_ROOT / "roms" / "Nintendo" / "NES" / "Super Mario Bros. 2.zip"

ROM_SHA1_PATH = INTEGRATION_DIR / "rom.sha"

MOVIES_DIR = GAME_DIR / "movies"
ARTIFACTS_DIR = GAME_DIR / "artifacts"
EVIDENCE_DIR = ARTIFACTS_DIR / "evidence"
EVIDENCE_MANIFEST_PATH = EVIDENCE_DIR / "level1_tas_evidence.json"
CONTROL_PROOF_PATH = EVIDENCE_DIR / "level1_control_proof.json"
REF_MOVIE_PATH = GAME_DIR / "tas" / "ref" / "tasvideos_1724_warps.fm2"

# State files stay under the integration directory even though the manifest
# stores portable, game-relative paths.
STATE_ARTIFACTS_DIR = INTEGRATION_DIR / "checkpoints"
STATE_ARTIFACTS_RELATIVE_DIR = Path("custom_integrations") / GAME / "checkpoints"
