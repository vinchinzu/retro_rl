"""Catalog + SMV env conversion for Super Metroid TAS corpus (no emulator)."""

from __future__ import annotations

import gzip
import struct
from pathlib import Path

import pytest

from retro_harness.controls import SNES_A, SNES_B, SNES_START, SNES_UP, SNES_Y
from super_metroid.tas.catalog import (
    MOVIES,
    SKIPPED,
    by_filename,
    catalog_full_slice_ids,
    fetchable,
    vanilla_fetchable,
)
from super_metroid.tas.fetch_refs import unwrap_movie
from super_metroid.tas.slice import SLICE_CATALOG
from super_metroid.tas.smv import parse_smv_env, write_bizhawk_bk2


def test_fetchable_filenames_unique() -> None:
    names = [m.filename for m in MOVIES]
    assert len(names) == len(set(names))
    assert all(m.url.startswith("https://tasvideos.org/") for m in fetchable())


def test_vanilla_corpus_excludes_contest_and_hacks() -> None:
    vanilla = vanilla_fetchable()
    assert by_filename("sniq_any_3653M.lsmv") in vanilla
    assert by_filename("sniq_low_3273M.lsmv") in vanilla
    assert by_filename("hero_bubbleroom.smv") in vanilla
    contest = by_filename("moozooh_smtc4.bk2")
    assert contest.fetch and not contest.vanilla
    assert contest not in vanilla
    reasons = " ".join(s.skip_reason or "" for s in SKIPPED)
    assert "Project Base" in reasons
    assert "RAM watch" in reasons


def test_catalog_full_slices_are_wired() -> None:
    ids = catalog_full_slice_ids()
    assert "sniq_low_full" in ids
    assert "saturn_rbo_full" in ids
    assert "hero_kraid_entry_full" in ids
    for sid in ids:
        assert sid in SLICE_CATALOG
        spec = SLICE_CATALOG[sid]
        assert spec.start == 0
        assert spec.end is None
        assert spec.kind in {"lsmv", "bk2", "smv"}


def test_existing_sniq_slices_unchanged() -> None:
    assert "sniq_any_full" in SLICE_CATALOG
    assert "sniq_100_full" in SLICE_CATALOG
    assert SLICE_CATALOG["sniq_any_full"].kind == "lsmv"
    assert SLICE_CATALOG["sniq_100_full"].kind == "bk2"


def _write_smv(path: Path, words: list[int]) -> None:
    author = "Test Author".encode("utf-16le") + b"\0\0"
    rom_info = b"\0\0\0" + struct.pack("<I", 0x12345678)
    rom_info += b"SUPER METROID".ljust(23, b"\0")
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


def test_parse_smv_to_env_and_bk2(tmp_path: Path) -> None:
    smv_path = tmp_path / "room.smv"
    # Up+Y, then A, then reset.
    words = [(1 << 3) | (1 << 6), 1 << 11, 0xFFFF]
    _write_smv(smv_path, words)
    movie = parse_smv_env(smv_path)
    assert movie.num_frames == 3
    assert movie.frames[0][SNES_UP] == 1
    assert movie.frames[0][SNES_Y] == 1
    assert movie.frames[1][SNES_A] == 1
    assert not any(movie.frames[2])

    bk2 = write_bizhawk_bk2(movie, tmp_path / "room.bk2")
    assert bk2.exists()
    from super_metroid.tas.bk2 import parse_bk2

    parsed = parse_bk2(bk2)
    assert parsed.num_frames == 3
    assert parsed.frames[0][SNES_UP] == 1
    assert parsed.frames[1][SNES_A] == 1


def test_parse_bkm_zip_as_bk2(tmp_path: Path) -> None:
    import zipfile

    from super_metroid.tas.bk2 import parse_bk2

    path = tmp_path / "map.bk2"
    with zipfile.ZipFile(path, "w") as zf:
        zf.writestr(
            "run.bkm",
            "|.|............|\n|.|.....S......|\n|.|.......B....|\n",
        )
    movie = parse_bk2(path)
    assert movie.num_frames == 3
    assert movie.frames[1][SNES_START] == 1
    assert movie.frames[2][SNES_B] == 1


def test_unwrap_plain_smv_bytes() -> None:
    ref = by_filename("hero_bubbleroom.smv")
    payload = b"SMV\x1a" + b"\0" * 40
    assert unwrap_movie(payload, ref).startswith(b"SMV\x1a")


@pytest.mark.parametrize("filename", [m.filename for m in fetchable() if m.expected_frames])
def test_expected_frame_counts_when_present(filename: str) -> None:
    ref = by_filename(filename)
    if not ref.path.exists():
        pytest.skip(f"missing {filename}")
    from super_metroid.tas.slice import load_movie_frames

    frames = load_movie_frames(ref.path, ref.kind)
    assert len(frames) == ref.expected_frames
