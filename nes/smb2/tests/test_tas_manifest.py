"""Offline round-trip and validation tests for the SMB2 evidence manifest."""

from __future__ import annotations

from dataclasses import replace

import pytest

from smb2.tas_manifest import (
    BizHawkValidationStatus,
    CheckpointEvidence,
    CheckpointStatus,
    MovieFormat,
    TASEvidenceManifest,
    make_scaffold_manifest,
    planned_level1_checkpoints,
)


def _manifest() -> TASEvidenceManifest:
    return make_scaffold_manifest(
        source_url="https://example.test/smb2-first-level.fm2",
        movie_hash="a" * 64,
        movie_format=MovieFormat.FM2,
        rom_hash="b" * 64,
        source_emulator="FCEUX",
        source_core="FCEUX 2.6.6",
        movie_path="movies/reference.fm2",
        rom_path="roms/Super Mario Bros. 2.nes",
    )


def test_scaffold_has_named_planned_state_slots() -> None:
    manifest = _manifest()

    assert manifest.bizhawk_validation_status is BizHawkValidationStatus.NOT_RUN
    assert [checkpoint.name for checkpoint in manifest.checkpoints] == [
        "level1_start",
        "level1_control",
        "level1_goal",
    ]
    assert all(
        checkpoint.status is CheckpointStatus.PLANNED
        and checkpoint.state_artifact_path.endswith(".state")
        and checkpoint.state_artifact_path.startswith(
            "custom_integrations/SuperMarioBros2-Nes/checkpoints/"
        )
        for checkpoint in manifest.checkpoints
    )


def test_manifest_dict_and_json_round_trip(tmp_path) -> None:
    manifest = _manifest()

    restored_from_dict = TASEvidenceManifest.from_dict(manifest.to_dict())
    assert restored_from_dict == manifest

    json_path = tmp_path / "manifest.json"
    assert manifest.write_json(json_path) == json_path
    assert TASEvidenceManifest.from_json(json_path) == manifest
    assert TASEvidenceManifest.from_json(manifest.to_json()) == manifest


def test_manifest_rejects_invalid_provenance() -> None:
    manifest = _manifest()

    with pytest.raises(ValueError, match="source_url"):
        replace(manifest, source_url="not-a-url")
    with pytest.raises(ValueError, match="movie_hash"):
        replace(manifest, movie_hash="not-a-digest")
    with pytest.raises(ValueError, match="checkpoint names"):
        replace(
            manifest,
            checkpoints=(
                manifest.checkpoints[0],
                replace(manifest.checkpoints[1], name=manifest.checkpoints[0].name),
            ),
        )


def test_checkpoint_requires_an_explicit_artifact_path() -> None:
    with pytest.raises(ValueError, match="state_artifact_path"):
        CheckpointEvidence(name="level1_start", state_artifact_path="")


def test_planned_slots_do_not_require_state_files() -> None:
    slots = planned_level1_checkpoints()

    assert slots
    assert all(checkpoint.status is CheckpointStatus.PLANNED for checkpoint in slots)
