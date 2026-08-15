from __future__ import annotations

import json
from pathlib import Path
import zipfile

import pytest

from zelda_i.scripts.compose_honest_route_recording import (
    bk2_frame_count,
    has_live_level8_complete_pin,
    validate_l5_endpoint,
)


def test_bk2_frame_count_counts_controller_rows(tmp_path: Path) -> None:
    movie = tmp_path / "sample.bk2"
    with zipfile.ZipFile(movie, "w") as archive:
        archive.writestr(
            "Input Log.txt",
            "[Input]\nP1 A|P1 B|\n|..|..|\n|..|A.|\n|..|.B|\n",
        )
    assert bk2_frame_count(movie) == 3


def test_level8_pin_requires_route_eligible_provenance(tmp_path: Path) -> None:
    (tmp_path / "Level8Complete.state").write_bytes(b"state")
    assert has_live_level8_complete_pin(tmp_path) is False
    (tmp_path / "Level8Complete.provenance.json").write_text(
        json.dumps({"natural_entry": False, "route_eligible": False})
    )
    assert has_live_level8_complete_pin(tmp_path) is False
    (tmp_path / "Level8Complete.provenance.json").write_text(
        json.dumps({"route_eligible": True})
    )
    assert has_live_level8_complete_pin(tmp_path) is True


def test_validate_l5_endpoint_accepts_assisted_tf() -> None:
    validate_l5_endpoint(
        {
            "ok": True,
            "final": {"level": 5, "screen": 0x14, "triforce": 0x1C},
            "assist": {"progression_writes": 0, "capacity_writes": 0},
        }
    )


def test_validate_l5_endpoint_fails_closed_on_progression_write() -> None:
    with pytest.raises(ValueError, match="progression writes"):
        validate_l5_endpoint(
            {
                "ok": True,
                "final": {"level": 5, "screen": 0x14, "triforce": 0x1C},
                "assist": {"progression_writes": 1, "capacity_writes": 0},
            }
        )
