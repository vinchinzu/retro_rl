"""BizHawk BK2 LogKey parser (no emulator)."""

from __future__ import annotations

import zipfile
from pathlib import Path

from retro_harness.bk2 import parse_bk2, parse_logkey_p1_to_env
from retro_harness.controls import SNES_B, SNES_RIGHT, SNES_START
from retro_harness.platformer.bk2_extract import extract_raw_actions_from_bk2


def _write_bk2(path: Path, log: str) -> Path:
    with zipfile.ZipFile(path, "w") as zf:
        zf.writestr("Header.txt", "GameName Test\nPlatform SNES\n")
        zf.writestr("Input Log.txt", log)
    return path


def test_parse_logkey_maps_mnemonic_order() -> None:
    logkey = (
        "LogKey:#Reset|Power|#P1 Up|P1 Down|P1 Left|P1 Right|"
        "P1 Select|P1 Start|P1 Y|P1 B|P1 X|P1 A|P1 L|P1 R|"
    )
    mapped = parse_logkey_p1_to_env(logkey)
    assert mapped is not None
    assert mapped[5] == SNES_START
    assert mapped[7] == SNES_B


def test_parse_bk2_reads_logkey(tmp_path: Path) -> None:
    log = (
        "[Input]\n"
        "LogKey:#Reset|Power|#P1 Up|P1 Down|P1 Left|P1 Right|"
        "P1 Select|P1 Start|P1 Y|P1 B|P1 X|P1 A|P1 L|P1 R|\n"
        "|..|............|\n"
        "|..|.....S......|\n"
        "|..|...R...B....|\n"
    )
    movie = parse_bk2(_write_bk2(tmp_path / "tiny.bk2", log))
    assert movie.num_frames == 3
    assert movie.frames[1][SNES_START] == 1
    assert movie.frames[2][SNES_RIGHT] == 1
    assert movie.frames[2][SNES_B] == 1
    auto = extract_raw_actions_from_bk2(tmp_path / "tiny.bk2")
    assert auto == movie.frames
