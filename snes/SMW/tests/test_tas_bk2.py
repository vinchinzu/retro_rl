"""Native BizHawk BK2 parsing tests."""

from __future__ import annotations

import hashlib
import zipfile

import pytest

from SMW.tas.bk2 import parse_bk2, retarget_bk2
from SMW.tas.smv import word_to_bk2_mnemonic


def _write_bk2(path, rom_data: bytes) -> None:
    header = "\n".join(
        (
            "emuVersion Version 1.11.8",
            "Platform SNES",
            "GameName Super Mario World",
            f"SHA1 {hashlib.sha1(rom_data).hexdigest().upper()}",
        )
    )
    log = "\n".join(
        (
            "[Input]",
            "LogKey:#Reset|Power|#P1 Up|P1 Down|P1 Left|P1 Right|P1 Select|P1 Start|P1 Y|P1 B|P1 X|P1 A|P1 L|P1 R|#P2 Up|P2 Down|P2 Left|P2 Right|P2 Select|P2 Start|P2 Y|P2 B|P2 X|P2 A|P2 L|P2 R|",
            "|..|...R..YB....|............|",
            "|R.|............|............|",
            "[/Input]",
        )
    )
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("Header.txt", header + "\n")
        archive.writestr("Input Log.txt", log + "\n")


def test_parse_native_bk2_and_verify_rom(tmp_path) -> None:
    rom_data = b"owned rom fixture"
    path = tmp_path / "movie.bk2"
    _write_bk2(path, rom_data)
    rom_path = tmp_path / "smw.sfc"
    rom_path.write_bytes(rom_data)

    movie = parse_bk2(path)

    assert movie.num_frames == 2
    assert movie.first_input_frame == 0
    assert word_to_bk2_mnemonic(movie.p1_words[0]) == "...R..YB...."
    assert movie.p1_words[1] == 0xFFFF
    movie.verify_rom(rom_path)

    wrong_rom = tmp_path / "wrong.sfc"
    wrong_rom.write_bytes(b"wrong")
    with pytest.raises(ValueError, match="ROM SHA-1 mismatch"):
        movie.verify_rom(wrong_rom)


def test_retarget_bk2_preserves_inputs_and_sets_explicit_core(tmp_path) -> None:
    rom_data = b"owned rom fixture"
    source = tmp_path / "source.bk2"
    _write_bk2(source, rom_data)

    output = retarget_bk2(source, tmp_path / "retargeted.bk2", core_profile="v115")

    with zipfile.ZipFile(output) as archive:
        header = archive.read("Header.txt").decode()
        sync = archive.read("SyncSettings.json").decode()
        input_log = archive.read("Input Log.txt")
    with zipfile.ZipFile(source) as archive:
        source_input_log = archive.read("Input Log.txt")
    assert "Core BSNESv115+" in header
    assert "BsnesCore+SnesSyncSettings" in sync
    assert input_log == source_input_log
