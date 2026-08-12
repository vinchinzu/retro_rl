"""Filesystem contracts for the SMB2 scaffold."""

from __future__ import annotations

from pathlib import Path

from smb2.paths import (
    ARTIFACTS_DIR,
    EVIDENCE_MANIFEST_PATH,
    GAME,
    INTEGRATION_DIR,
    MOVIES_DIR,
    ROM_PATH,
    SHARED_ROM_ZIP,
    STATE_ARTIFACTS_DIR,
)


def test_paths_are_bounded_to_smb2() -> None:
    game_dir = Path(__file__).resolve().parents[1]

    assert ROM_PATH == game_dir / "roms" / "Super Mario Bros. 2.nes"
    assert SHARED_ROM_ZIP == (
        game_dir.parent.parent / "roms" / "Nintendo" / "NES" / "Super Mario Bros. 2.zip"
    )
    assert INTEGRATION_DIR == game_dir / "custom_integrations" / GAME
    assert MOVIES_DIR == game_dir / "movies"
    assert ARTIFACTS_DIR == game_dir / "artifacts"
    assert STATE_ARTIFACTS_DIR == INTEGRATION_DIR / "checkpoints"
    assert EVIDENCE_MANIFEST_PATH == (
        ARTIFACTS_DIR / "evidence" / "level1_tas_evidence.json"
    )
