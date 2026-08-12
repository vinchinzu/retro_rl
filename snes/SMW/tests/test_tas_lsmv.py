"""LSMV parsing and BizHawk conversion tests."""

from __future__ import annotations

import hashlib
import json
import zipfile

import pytest

from SMW.tas.lsmv import parse_lsmv, write_bizhawk_bk2
from SMW.tas.smv import word_to_bk2_mnemonic


def _write_lsmv(path, rom_data: bytes, input_lines: list[str]) -> None:
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("input", "\n".join(input_lines) + "\n")
        archive.writestr("authors", "Test Author\n")
        archive.writestr("coreversion", "bsnes v085 (Compatibility core)\n")
        archive.writestr("gametype", "snes_ntsc\n")
        archive.writestr("rerecords", "17\n")
        archive.writestr("rom.hint", "Super Mario World (U) [!]\n")
        archive.writestr("rom.sha256", hashlib.sha256(rom_data).hexdigest() + "\n")


def test_parse_lsmv_and_convert_to_bsnes_bk2(tmp_path) -> None:
    rom_data = b"owned rom fixture"
    lsmv_path = tmp_path / "source.lsmv"
    _write_lsmv(
        lsmv_path,
        rom_data,
        ["F. 0 0|............", "F. 0 0|BY......AXLR", "FR 0 0|............"],
    )
    rom_path = tmp_path / "smw.sfc"
    rom_path.write_bytes(rom_data)

    movie = parse_lsmv(lsmv_path)

    assert movie.num_frames == 3
    assert movie.first_input_frame == 1
    assert movie.p1_words[2] == 0xFFFF
    assert word_to_bk2_mnemonic(movie.p1_words[1]) == "......YBXAlr"
    output = write_bizhawk_bk2(movie, tmp_path / "converted.bk2", rom_path=rom_path)
    with zipfile.ZipFile(output) as archive:
        header = archive.read("Header.txt").decode()
        log = archive.read("Input Log.txt").decode().splitlines()
        sync = json.loads(archive.read("SyncSettings.json"))
    assert "Core BSNESv115+" in header
    assert log[3] == "|..|......YBXAlr|............|"
    assert log[4].startswith("|R.|")
    assert sync["o"]["Profile"] == "Compatibility"


def test_lsmv_converter_rejects_wrong_rom(tmp_path) -> None:
    lsmv_path = tmp_path / "source.lsmv"
    _write_lsmv(lsmv_path, b"expected", ["F. 0 0|............"])
    wrong_rom = tmp_path / "wrong.sfc"
    wrong_rom.write_bytes(b"wrong")

    with pytest.raises(ValueError, match="ROM SHA-256 mismatch"):
        write_bizhawk_bk2(
            parse_lsmv(lsmv_path), tmp_path / "bad.bk2", rom_path=wrong_rom
        )


def test_lsmv_converter_can_target_legacy_bsnes(tmp_path) -> None:
    rom_data = b"owned rom fixture"
    lsmv_path = tmp_path / "source.lsmv"
    _write_lsmv(lsmv_path, rom_data, ["F. 0 0|............"])
    rom_path = tmp_path / "smw.sfc"
    rom_path.write_bytes(rom_data)

    output = write_bizhawk_bk2(
        parse_lsmv(lsmv_path),
        tmp_path / "legacy.bk2",
        rom_path=rom_path,
        core_profile="legacy",
    )
    with zipfile.ZipFile(output) as archive:
        header = archive.read("Header.txt").decode()
        sync = archive.read("SyncSettings.json").decode()
    assert "Core BSNES\n" in header
    assert "LibsnesCore+SnesSyncSettings" in sync
