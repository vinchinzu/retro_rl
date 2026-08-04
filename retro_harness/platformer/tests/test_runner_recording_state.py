"""Tests for verifier recording-state selection."""

import json

from retro_harness.platformer.runner import _recording_start_state


def test_recording_start_state_reads_practice_metadata(tmp_path):
    path = tmp_path / "attempt_000_raw.json"
    path.write_text(
        json.dumps(
            {
                "raw_buttons": [[0] * 12],
                "metadata": {"state": "ret00_0x96BA"},
            }
        )
    )

    assert _recording_start_state(path) == "ret00_0x96BA"


def test_recording_start_state_reads_top_level_state(tmp_path):
    path = tmp_path / "seed.json"
    path.write_text(json.dumps({"raw_buttons": [[0] * 12], "start_state": "Start"}))

    assert _recording_start_state(path) == "Start"


def test_recording_start_state_rejects_conflicting_or_invalid_metadata(tmp_path):
    conflict = tmp_path / "conflict.json"
    conflict.write_text(
        json.dumps(
            {
                "state": "Direct",
                "metadata": {"state": "Split"},
            }
        )
    )
    invalid = tmp_path / "invalid.json"
    invalid.write_text("not json")

    assert _recording_start_state(conflict) is None
    assert _recording_start_state(invalid) is None
