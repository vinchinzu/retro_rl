"""Offline loader for ``pin_probe --fixture`` dumps. No ROM."""

from __future__ import annotations

from pathlib import Path

import pytest

from alttp.scripts.pin_probe import load_room_fixture

# CastleMain, written by pin_probe --fixture. South door under Link is 0x8E.
_ROOM_61 = Path(__file__).parent / "fixtures" / "room_61.json"


def test_room_61_fixture_has_link_tiles_and_sprites() -> None:
    room = load_room_fixture(_ROOM_61)
    assert room["room"] == 0x61
    assert room["room_hex"] == "0x61"
    assert room["layer"] == 0
    assert room["link"] == {
        "x": 760,
        "y": 3520,
        "layer": 0,
        "dir": 0,
        "state": 0,
        "hp": 0x18,
    }
    assert room["tiles"][56][31] == 0x8E
    assert room["tiles"][53][21] == 0x04
    assert [sprite["name"] for sprite in room["sprites"]] == [
        "green soldier",
        "knight",
        "knight",
    ]
    assert room["sprites"][0]["hp"] == 4
    assert room["sprites"][0]["x"] == 704
    assert room["sprites"][0]["y"] == 3296


def test_load_room_fixture_rejects_a_short_grid(tmp_path: Path) -> None:
    bad = tmp_path / "room_00.json"
    bad.write_text(
        '{"room": 0, "link": {"x": 0, "y": 0, "layer": 0, "dir": 0, '
        '"state": 0, "hp": 0}, "sprites": [], "tiles": [[0]]}\n'
    )
    with pytest.raises(ValueError, match="64x64"):
        load_room_fixture(bad)
