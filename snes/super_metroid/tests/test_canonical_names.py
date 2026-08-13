"""Map Rando / sm-json-data canonical room names (no ROM)."""

from __future__ import annotations

from super_metroid.rooms.canonical_names import (
    MAPRANDO_LOGIC_URL,
    build_catalog_payload,
    load_canonical_names,
    load_canonical_rooms,
    parse_rooms_from_sm_json_data,
    room_name,
)


def test_parse_sm_json_data_has_maprando_count() -> None:
    rooms = parse_rooms_from_sm_json_data()
    assert len(rooms) >= 250
    by_name = {r.name: r for r in rooms}
    ls = by_name["Landing Site"]
    assert ls.maprando_id == 8
    assert ls.room_id == 0x91F8
    assert ls.area == "Crateria"
    assert "room/8" in ls.logic_url
    assert MAPRANDO_LOGIC_URL in ls.logic_url


def test_load_canonical_names_includes_bosses() -> None:
    names = load_canonical_names()
    assert names[0x91F8] == "Landing Site"
    assert names[0xA59F] == "Kraid Room"
    assert names[0xDD58] == "Mother Brain Room"
    assert names[0x95FF] == "The Moat"


def test_room_name_fallback_hex() -> None:
    assert room_name(0x91F8).startswith("Landing") or room_name(0x91F8) == "Landing Site"
    assert room_name(0xDEAD, names={}) == "0xDEAD"


def test_catalog_payload_summary() -> None:
    payload = build_catalog_payload()
    assert payload["summary"]["roomCount"] == len(payload["rooms"])
    assert "Crateria" in payload["summary"]["areaCounts"]
    assert payload["upgradeItems"]


def test_load_canonical_rooms_roundtrip_fields() -> None:
    rooms = load_canonical_rooms()
    morph = next(r for r in rooms if r.name == "Morph Ball Room")
    assert morph.maprando_id == 38
    assert morph.room_id_hex == "0x9E9F"
