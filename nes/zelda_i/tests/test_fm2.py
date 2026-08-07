"""Unit tests for FCEUX FM2 import (no emulator required)."""

from __future__ import annotations

from pathlib import Path

from retro_harness.controls import NES_A, NES_B, NES_LEFT, NES_RIGHT, NES_START
from zelda_i.paths import GAME_DIR
from zelda_i.tas.fm2 import frames_to_nes9_rle_payload, parse_fm2

REF_ALL = GAME_DIR / "tas" / "ref" / "chatterbox_allitems_4767M.fm2"
REF_ALL_OLD = GAME_DIR / "tas" / "ref" / "taseditor_allitems_2508M.fm2"


def test_parse_minimal_inline(tmp_path: Path) -> None:
    path = tmp_path / "tiny.fm2"
    path.write_text(
        "version 3\n"
        "romFilename Legend of Zelda.nes\n"
        "|0|........||\n"
        "|0|R.....B.||\n"
        "|0|....T..A||\n",  # T=Start, A=A (FM2: RLDUTSBA)
        encoding="utf-8",
    )
    movie = parse_fm2(path)
    assert movie.num_frames == 3
    assert movie.frames[1][NES_RIGHT] == 1
    assert movie.frames[1][NES_B] == 1
    assert movie.frames[2][NES_START] == 1
    assert movie.frames[2][NES_A] == 1


def test_frames_to_rle_payload() -> None:
    frames = [
        [0, 0, 0, 0, 0, 0, 0, 1, 0],
        [0, 0, 0, 0, 0, 0, 0, 1, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 1],
    ]
    payload = frames_to_nes9_rle_payload(
        frames, route_id="test", source="unit"
    )
    assert payload["format"] == "nes9_rle"
    assert payload["game_name"] == "LegendOfZelda-Nes"
    assert payload["num_frames"] == 3
    assert payload["segments"] == [
        {"b": [0, 0, 0, 0, 0, 0, 0, 1, 0], "n": 2},
        {"b": [0, 0, 0, 0, 0, 0, 0, 0, 1], "n": 1},
    ]


def test_parse_allitems_if_present() -> None:
    """Primary non-glitch movie (no Heavy glitch abuse tag)."""
    if not REF_ALL.exists():
        return
    movie = parse_fm2(REF_ALL)
    assert movie.num_frames == 114_913
    s = movie.summary()
    assert s["first_nonzero_frame"] is not None
    assert any(any(fr) for fr in movie.frames[:200])


def test_parse_taseditor_allitems_if_present() -> None:
    if not REF_ALL_OLD.exists():
        return
    movie = parse_fm2(REF_ALL_OLD)
    assert movie.num_frames > 50_000
    assert movie.num_frames < 200_000
