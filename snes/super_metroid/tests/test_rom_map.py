"""Practice ROM preset pointer search (no emulator)."""

from __future__ import annotations

import struct

import pytest

from super_metroid.paths import SHARED_PRACTICE_ROM
from super_metroid.practice_repertoire.catalog import load_catalog
from super_metroid.practice_repertoire.rom_map import (
    _payload,
    file_offset_to_snes,
    join_blobs_by_label,
    map_preset_addresses,
    snes_to_file_offset,
    word_hash,
)


def test_lorom_offset_roundtrip() -> None:
    snes = file_offset_to_snes(0x44B5)
    assert snes_to_file_offset(snes) == 0x44B5
    assert (snes >> 16) == 0x80
    assert (snes & 0xFFFF) == 0xC4B5


def test_payload_layout() -> None:
    blob = _payload(0, [(0x079B, 0x9E9F), (0x0AF6, 0x0080)])
    words = struct.unpack("<6H", blob)
    assert words[0] == 0
    assert words[1] == 0x079B
    assert words[2] == 0x9E9F
    assert words[-1] == 0xFFFF


def test_word_hash_is_order_independent() -> None:
    a = {0x079B: 0x9E9F, 0x0AF6: 0x0080}
    b = {0x0AF6: 0x0080, 0x079B: 0x9E9F}
    assert word_hash(a) == word_hash(b)
    assert word_hash(a) != word_hash({0x079B: 0x9E9F, 0x0AF6: 0x0081})


def test_join_blobs_by_data_label_not_five_tuple() -> None:
    """Same room/xy/items/pose still maps each label to its own blob."""

    five = {
        0x078D: 0xAB58,
        0x079B: 0x9E9F,
        0x0AF6: 0x0580,
        0x0AFA: 0x02A8,
        0x09A4: 0x0000,
        0x0A1C: 0x0000,
    }
    words_a = {**five, 0x09C2: 99}
    words_b = {**five, 0x09C2: 150}
    blobs = [
        {
            "offset": 100,
            "snes": 0xE8E000,
            "snes_hex": "0xE8E000",
            "parent_lo": 0,
            "words": words_a,
        },
        {
            "offset": 200,
            "snes": 0xE8E100,
            "snes_hex": "0xE8E100",
            "parent_lo": 0xE000,
            "words": words_b,
        },
    ]
    sessions = [
        {
            "id": "kpdr20/crateria/morph",
            "data_label": "preset_kpdr20_crateria_morph",
            "effective_state_sha256": word_hash(words_a),
        },
        {
            "id": "kpdr25/crateria/morph",
            "data_label": "preset_kpdr25_crateria_morph",
            "effective_state_sha256": word_hash(words_b),
        },
    ]
    found, missing = join_blobs_by_label(sessions, blobs)
    assert missing == []
    assert found["preset_kpdr20_crateria_morph"]["snes"] == 0xE8E000
    assert found["preset_kpdr25_crateria_morph"]["snes"] == 0xE8E100


def test_join_prefers_catalog_snes() -> None:
    blobs = [
        {
            "offset": 1,
            "snes": 0xE8E354,
            "snes_hex": "0xE8E354",
            "parent_lo": 0,
            "words": {0x079B: 1},
        }
    ]
    sessions = [
        {
            "id": "kpdr25/crateria/morph",
            "data_label": "preset_kpdr25_crateria_morph",
            "snes": 0xE8E354,
            "effective_state_sha256": "deadbeef",
        }
    ]
    found, missing = join_blobs_by_label(sessions, blobs)
    assert missing == []
    assert found["preset_kpdr25_crateria_morph"]["snes"] == 0xE8E354


def test_map_preset_addresses_has_no_cache_fetch() -> None:
    import inspect

    params = inspect.signature(map_preset_addresses).parameters
    assert "cache" not in params
    assert "fetch" not in params


@pytest.mark.skipif(
    not SHARED_PRACTICE_ROM.is_file(),
    reason="practice ROM not built (run setup_practice_rom.py)",
)
def test_live_map_finds_kpdr25_morph() -> None:
    report = map_preset_addresses()
    assert report["blobs"] > 100
    sessions = load_catalog()["sessions"]
    if not any(
        rec.get("snes") is not None or rec.get("effective_state_sha256")
        for rec in sessions
    ):
        pytest.skip("catalog has no ROM pointers; re-export practice_repertoire.json")
    morph = report["presets"].get("preset_kpdr25_crateria_morph")
    assert morph is not None
    assert morph["snes"] > 0
    assert "preset_kpdr25_crateria_morph" not in report["missing"]
