"""Offline tests for SMB2 TAS checkpoint capture (no emulator)."""

from __future__ import annotations

import json
from dataclasses import replace

from smb2.ram import (
    HEARTS_TWO,
    SPAWN_X,
    SPAWN_Y,
    Smb2Snapshot,
    is_level1_control,
)
from smb2.tas_manifest import (
    BizHawkValidationStatus,
    CheckpointStatus,
    MovieFormat,
    TASEvidenceManifest,
    make_scaffold_manifest,
)
from smb2.paths import CONTROL_PROOF_PATH, EVIDENCE_MANIFEST_PATH
from smb2.tas_oracle import (
    CapturedCheckpoint,
    apply_captures_to_manifest,
    first_matching_frame,
    snapshot_dict,
    write_control_proof,
)


def _snap(**overrides: int) -> Smb2Snapshot:
    values = dict(
        frame=0,
        player_x=SPAWN_X,
        player_y=SPAWN_Y,
        x_page=0,
        y_page=0,
        x_speed=0,
        jump_physics=0,
        character=2,
        hearts=HEARTS_TWO,
        lives=3,
        transition=0,
        area=0,
        subarea=0,
        level=0,
        world=0,
    )
    values.update(overrides)
    return Smb2Snapshot(**values)


def test_first_matching_frame_skips_pre_control_spawn() -> None:
    start = _snap(frame=253, jump_physics=0)
    control = _snap(frame=304, jump_physics=7)
    later = _snap(frame=400, jump_physics=7, player_x=176, y_page=1)
    found = first_matching_frame([start, control, later], is_level1_control)
    assert found is not None
    assert found.frame == 304
    assert first_matching_frame([start], is_level1_control) is None


def test_apply_captures_marks_control_materialized_and_keeps_bizhawk_blocked() -> None:
    manifest = make_scaffold_manifest(
        source_url="https://tasvideos.org/1724M?handler=Download",
        movie_hash="a" * 64,
        movie_format=MovieFormat.FM2,
        rom_hash="b" * 64,
        source_emulator="FCEUX",
        source_core="FCEUX 0.98.28",
        movie_path="tas/ref/tasvideos_1724_warps.fm2",
        rom_path="roms/Super Mario Bros. 2.nes",
    )
    snap = _snap(frame=304, jump_physics=7)
    captured = CapturedCheckpoint(
        name="level1_control",
        frame=304,
        snapshot=snap,
        state_bytes=b"state",
    )
    updated = apply_captures_to_manifest(manifest, {"level1_control": captured})
    by_name = {slot.name: slot for slot in updated.checkpoints}
    assert by_name["level1_control"].status is CheckpointStatus.MATERIALIZED
    assert by_name["level1_control"].frame == 304
    assert by_name["level1_start"].status is CheckpointStatus.PLANNED
    assert updated.bizhawk_validation_status is BizHawkValidationStatus.BLOCKED
    assert "fceumm" in by_name["level1_control"].description


def test_write_control_proof_round_trips_gate_fields(tmp_path) -> None:
    snap = _snap(frame=304, jump_physics=7)
    captured = CapturedCheckpoint(
        name="level1_control",
        frame=304,
        snapshot=snap,
        state_bytes=b"",
    )
    path = write_control_proof(captured, path=tmp_path / "proof.json")
    payload = path.read_text(encoding="utf-8")
    assert '"movie_frame": 304' in payload
    assert '"replay_core": "fceumm"' in payload
    assert snapshot_dict(snap)["player_x"] == SPAWN_X
    restored = replace(snap, obs_mean=None)
    assert is_level1_control(restored) is True


def test_tracked_evidence_parses_and_records_control_frame() -> None:
    manifest = TASEvidenceManifest.from_json(EVIDENCE_MANIFEST_PATH)
    by_name = {slot.name: slot for slot in manifest.checkpoints}
    assert by_name["level1_control"].status is CheckpointStatus.MATERIALIZED
    assert by_name["level1_control"].frame == 304
    assert by_name["level1_start"].status is CheckpointStatus.MATERIALIZED
    assert by_name["level1_start"].frame == 253
    assert by_name["level1_goal"].status is CheckpointStatus.PLANNED
    assert manifest.bizhawk_validation_status is BizHawkValidationStatus.BLOCKED


def test_tracked_control_proof_is_fceumm_not_bizhawk() -> None:
    payload = json.loads(CONTROL_PROOF_PATH.read_text(encoding="utf-8"))
    assert payload["movie_frame"] == 304
    assert payload["replay_core"] == "fceumm"
    assert payload["bizhawk_validation_status"] == "blocked"
    assert payload["snapshot"]["player_x"] == 120
    assert payload["snapshot"]["jump_physics"] == 7
