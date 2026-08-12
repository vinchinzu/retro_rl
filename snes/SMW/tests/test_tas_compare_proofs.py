"""Tests for independent BizHawk proof comparison."""

from __future__ import annotations

import json

import pytest

from SMW.tas.compare_proofs import compare_proofs


def _write_proof(path, *, exit_frame: int = 20, status: str = "GREEN") -> None:
    path.write_text(
        json.dumps(
            {
                "status": status,
                "source_sha256": "source",
                "rom_hash": "rom",
                "segments": [
                    {
                        "index": 1,
                        "translevel": 0x2A,
                        "entry_frame": 10,
                        "exit_frame": exit_frame,
                        "max_player_x": 5000,
                        "retry_count": 0,
                        "sublevel_count": 0,
                        "lives_drops": 0,
                        "completion_signal": "end_level_timer",
                        "entry_ram": {"game_mode": 0x14},
                        "exit_ram": {"game_mode": 0x0C},
                    }
                ],
            }
        )
    )


def test_compare_proofs_accepts_identical_independent_runs(tmp_path) -> None:
    first = tmp_path / "first.json"
    second = tmp_path / "second.json"
    _write_proof(first)
    _write_proof(second)

    result = compare_proofs([first, second])

    assert result["status"] == "GREEN"
    assert result["independent_runs"] == 2


def test_compare_proofs_rejects_frame_mismatch_and_red(tmp_path) -> None:
    first = tmp_path / "first.json"
    mismatch = tmp_path / "mismatch.json"
    red = tmp_path / "red.json"
    _write_proof(first)
    _write_proof(mismatch, exit_frame=21)
    _write_proof(red, status="RED")

    with pytest.raises(ValueError, match="fingerprint mismatch"):
        compare_proofs([first, mismatch])
    with pytest.raises(ValueError, match="not GREEN"):
        compare_proofs([first, red])
