"""NesHawk FM2→BK2 conversion (no emulator)."""

from __future__ import annotations

from pathlib import Path
from zipfile import ZipFile

from retro_harness.controls import NES_A, NES_LEFT, NES_RIGHT, NES_START
from smb.paths import GAME_DIR
from smb.tas.bk2 import (
    KNOWN_HEADERLESS_MD5,
    KNOWN_HEADERLESS_SHA1,
    build_input_log,
    fm2_cmd_to_bk2_reset,
    fm2_p1_to_bk2,
    headerless_rom_hashes,
    parse_bk2,
    write_neshawk_bk2,
)
from smb.tas.fm2 import parse_fm2, parse_movie

WARPS_FM2 = GAME_DIR / "tas" / "ref" / "happylee_warps_1715M.fm2"
WARPS_BK2 = GAME_DIR / "tas" / "ref" / "happylee_warps_1715M.fm2.bk2"


def test_fm2_p1_to_bk2_start_lr_a() -> None:
    assert fm2_p1_to_bk2("........") == "........"
    assert fm2_p1_to_bk2("....T...") == "....S..."
    assert fm2_p1_to_bk2("RL......") == "..LR...."
    assert fm2_p1_to_bk2("R......A") == "...R...A"
    assert fm2_cmd_to_bk2_reset(1) == "r."
    assert fm2_cmd_to_bk2_reset(0) == ".."


def test_headerless_rom_hashes_match_bizhawk() -> None:
    md5, sha1 = headerless_rom_hashes()
    assert md5 == KNOWN_HEADERLESS_MD5
    assert sha1 == KNOWN_HEADERLESS_SHA1


def test_roundtrip_tiny_fm2(tmp_path: Path) -> None:
    fm2 = tmp_path / "tiny.fm2"
    fm2.write_text(
        "version 3\n"
        "emuVersion 9828\n"
        "rerecordCount 1\n"
        "palFlag 0\n"
        "romFilename Super Mario Bros. (JU) [!]\n"
        "comment author TestLee\n"
        "|1|........|........||\n"
        "|0|....T...|........||\n"
        "|0|RL......|........||\n"
        "|0|R......A|........||\n",
        encoding="utf-8",
    )
    bk2 = write_neshawk_bk2(fm2, tmp_path / "tiny.fm2.bk2")
    with ZipFile(bk2) as zf:
        names = zf.namelist()
        header = zf.read("Header.txt").decode()
        log = zf.read("Input Log.txt").decode()
        comments = zf.read("Comments.txt").decode()
    assert names[0] == "BizState 1.0"
    assert "Core NesHawk" in header
    assert "Author TestLee" in header
    assert "MovieOrigin .fm2 version 3" in comments
    assert "|r.|........|........|" in log
    assert "|..|....S...|........|" in log
    assert "|..|..LR....|........|" in log
    assert "|..|...R...A|........|" in log

    movie = parse_bk2(bk2)
    assert movie.num_frames == 4
    assert movie.commands[0] == 1
    assert movie.frames[1][NES_START] == 1
    assert movie.frames[2][NES_LEFT] == 1
    assert movie.frames[2][NES_RIGHT] == 1
    assert movie.frames[3][NES_A] == 1
    assert movie.frames[3][NES_RIGHT] == 1
    via = parse_movie(bk2)
    assert via.frames == movie.frames


def test_warps_conversion_matches_on_disk_input_log(tmp_path: Path) -> None:
    if not WARPS_FM2.exists() or not WARPS_BK2.exists():
        return
    rebuilt = build_input_log(WARPS_FM2)
    with ZipFile(WARPS_BK2) as zf:
        original = zf.read("Input Log.txt").decode("utf-8")
    assert rebuilt == original
    out = write_neshawk_bk2(WARPS_FM2, tmp_path / "warps.fm2.bk2")
    fm2 = parse_fm2(WARPS_FM2)
    bk2 = parse_bk2(out)
    assert bk2.num_frames == fm2.num_frames == 17_868
    assert bk2.frames == fm2.frames
    assert bk2.lr_frames == 85
