"""SMV parsing, BizHawk conversion, and exact-skill extraction tests."""

from __future__ import annotations

import gzip
import json
import struct
import zipfile

import pytest

from SMW.tas.skills import extract_level_skills, rle_words
from SMW.tas.smv import parse_smv, word_to_bk2_mnemonic, write_bizhawk_bk2


def _write_smv(path, words: list[int]) -> None:
    author = "Test Author".encode("utf-16le") + b"\0\0"
    rom_info = b"\0\0\0" + struct.pack("<I", 0x12345678)
    rom_info += b"SUPER MARIOWORLD".ljust(23, b"\0")
    save_ram = gzip.compress(bytes(0x20000), mtime=0) + b"\xcc\xcc"
    state_offset = 32 + len(author) + len(rom_info)
    input_offset = state_offset + len(save_ram)
    header = bytearray(32)
    header[:4] = b"SMV\x1a"
    struct.pack_into("<4I", header, 4, 1, 123, 7, len(words) - 1)
    header[0x14] = 1
    header[0x15] = 1
    header[0x17] = 0x41
    struct.pack_into("<2I", header, 0x18, state_offset, input_offset)
    raw_words = b"".join(
        b"\xff\xff"
        if word == 0xFFFF
        else bytes((((word >> 8) & 0xF) << 4, word & 0xFF))
        for word in words
    )
    path.write_bytes(bytes(header) + author + rom_info + save_ram + raw_words)


def test_parse_smv_and_convert_to_native_bk2(tmp_path) -> None:
    smv_path = tmp_path / "source.smv"
    # Up + Y, repeated, reset, A.
    words = [(1 << 3) | (1 << 6), (1 << 3) | (1 << 6), 0xFFFF, 1 << 11]
    _write_smv(smv_path, words)
    rom_path = tmp_path / "smw.sfc"
    rom_path.write_bytes(b"owned rom fixture")

    movie = parse_smv(smv_path)

    assert movie.author == "Test Author"
    assert movie.rom_crc32 == "12345678"
    assert movie.rom_name == "SUPER MARIOWORLD"
    assert movie.save_ram == bytes(0x20000)
    assert movie.p1_words == tuple(words)
    assert movie.first_input_frame == 0
    assert word_to_bk2_mnemonic(words[0]) == "U.....Y....."

    output = write_bizhawk_bk2(movie, tmp_path / "converted.bk2", rom_path=rom_path)
    with zipfile.ZipFile(output) as zf:
        header = zf.read("Header.txt").decode()
        log = zf.read("Input Log.txt").decode().splitlines()
        comments = json.loads(zf.read("Comments.txt"))
    assert "Core Snes9x" in header
    assert "StartsFromSavestate False" in header
    assert log[2] == "|..|U.....Y.....|"
    assert log[4].startswith("|R.|")
    assert comments["sync_claim"].startswith("unverified")


def test_parse_smv_rejects_truncated_controller_data(tmp_path) -> None:
    smv_path = tmp_path / "bad.smv"
    _write_smv(smv_path, [0, 0])
    smv_path.write_bytes(smv_path.read_bytes()[:-1])

    with pytest.raises(ValueError, match="sample-aligned"):
        parse_smv(smv_path)


def test_rle_and_verified_skill_bounds(tmp_path) -> None:
    smv_path = tmp_path / "source.smv"
    _write_smv(smv_path, [0, 0, 1 << 6, 1 << 6, 0])
    movie = parse_smv(smv_path)

    assert rle_words(movie.p1_words)[0]["frames"] == 2
    paths = extract_level_skills(
        movie,
        [
            {
                "index": 1,
                "translevel": 0x29,
                "entry_frame": 1,
                "exit_frame": 5,
                "entry_ram": {"game_mode": 0x14},
                "exit_ram": {"game_mode": 0x0B},
            }
        ],
        tmp_path / "skills",
    )
    payload = json.loads(paths[0].read_text())
    assert payload["num_frames"] == 4
    assert payload["quality"] == "clean_single_attempt"
    assert payload["runs"][1] == {"frames": 2, "word": 64, "buttons": ["Y"]}

    with pytest.raises(ValueError, match="invalid verified segment bounds"):
        extract_level_skills(
            movie,
            [{"index": 2, "translevel": 1, "entry_frame": 4, "exit_frame": 8}],
            tmp_path / "bad_skills",
        )


def test_skill_marks_retried_replay(tmp_path) -> None:
    smv_path = tmp_path / "source.smv"
    _write_smv(smv_path, [0, 1 << 6, 0])
    movie = parse_smv(smv_path)

    paths = extract_level_skills(
        movie,
        [
            {
                "index": 1,
                "translevel": 0x2A,
                "entry_frame": 0,
                "exit_frame": 3,
                "retry_count": 2,
                "lives_drops": 2,
            }
        ],
        tmp_path / "skills",
    )

    payload = json.loads(paths[0].read_text())
    assert payload["quality"] == "replay_with_retries"
    assert payload["retry_count"] == 2
