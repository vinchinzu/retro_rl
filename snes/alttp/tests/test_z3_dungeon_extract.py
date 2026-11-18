"""Load the 13 z3 dungeon route extracts. No ROM and no network."""

from __future__ import annotations

from alttp.scripts.z3_dungeon_extract import (
    DUNGEON_FILES,
    ROUTE_DATA_DIR,
    build_dungeon,
    load_route,
)


def test_each_dungeon_route_loads() -> None:
    files = sorted(path.stem for path in ROUTE_DATA_DIR.glob("*.yaml"))
    assert files == sorted(DUNGEON_FILES)
    assert len(files) == 13
    for slug in DUNGEON_FILES:
        doc = load_route(slug)
        assert doc == build_dungeon(slug)
        assert doc["slug"] == slug
        assert doc["name"]
        assert doc["regions"]
        assert doc["locations"]
        assert doc["connections"]
        assert any(not row["internal"] for row in doc["connections"])
        assert doc["requirements"]["keys"] or doc["requirements"]["items"]
        # Agahnim 2 is nodeType item with no vanilla nodeItem.
        assert all(row["node_type"] == "item" and row["name"] for row in doc["locations"])


def test_eastern_bow_castle_lamp_and_skull_key_token() -> None:
    eastern = load_route("eastern")
    chest = next(
        row
        for row in eastern["locations"]
        if row["name"] == "Eastern Palace - Big Chest"
    )
    assert chest["item"] == "Bow"
    assert chest["address"] == "0xE97D"
    assert chest["requires"] == ["BigKeyP1"]
    assert any(key["name"] == "BigKeyP1" for key in eastern["requirements"]["keys"])

    castle = load_route("castle")
    assert castle["name"] == "Hyrule Castle"
    lamp = next(row for row in castle["locations"] if row["item"] == "Lamp")
    assert lamp["name"] == "Hyrule Castle - Zelda's Chest"
    assert "Sanctuary" in [region["name"] for region in castle["regions"]]

    skull = load_route("skull")
    assert any("BigKeyD3" in note for note in skull["notes"])
    big_key = next(
        row for row in skull["locations"] if row["name"] == "Skull Woods - Big Chest"
    )
    assert big_key["requires"] == ["BigKeyD3 "]
    assert big_key["keys"] == [{"name": "BigKeyD3", "count": 1, "raw": "BigKeyD3 "}]
