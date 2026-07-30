"""Tests for z3-json-data loader (synthetic fixtures; no network/ROM)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from alttp.z3_json_data import (
    OPENING_ROUTE_ROOM_NAMES,
    Z3JsonData,
    Z3JsonDataNotFoundError,
    Z3JsonDataShapeError,
    discover_shape_issues,
    main,
    source_status,
    validate_source_shape,
)


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def _minimal_tree(root: Path) -> Path:
    """Build a tiny opening-route-shaped z3-json-data tree under *root*."""
    _write_json(
        root / "items.json",
        {
            "base": ["GreenMail"],
            "inventory": {"Lamp": {"data": "0x12"}},
            "progressives": {"L1Sword": {"data": "0x49"}},
        },
    )
    _write_json(
        root / "connections" / "main.json",
        {
            "connections": [
                {
                    "connectionType": "Door",
                    "description": "Door connection between Links House Exit and Light World",
                    "nodes": [
                        {"name": "Links House Exit", "position": "origin"},
                        {"name": "Light World", "position": "destination"},
                    ],
                },
                {
                    "connectionType": "Door",
                    "description": "Door connection between Hyrule Castle Main Gate and Hyrule Castle Courtyard",
                    "nodes": [
                        {"name": "Hyrule Castle Main Gate", "position": "origin"},
                        {"name": "Hyrule Castle Courtyard", "position": "destination"},
                    ],
                },
            ]
        },
    )
    _write_json(
        root / "enemies" / "main.json",
        {
            "enemies": [
                {"id": 0, "names": ["Crow"], "hp": 4},
                {"id": 1, "names": ["Octorok"], "hp": 2},
            ]
        },
    )
    _write_json(root / "schema" / "z3-region.schema.json", {"title": "stub"})
    _write_json(root / "schema" / "z3-connection.schema.json", {"title": "stub"})
    _write_json(
        root / "regions" / "lightworld" / "south" / "caves.json",
        {
            "rooms": [
                {
                    "id": 11,
                    "name": "Links House",
                    "roomType": "Cave",
                    "nodes": [
                        {
                            "id": 1,
                            "name": "Links House Exit",
                            "area": "Links House",
                            "nodeType": "door",
                        },
                        {
                            "id": 2,
                            "name": "Link's House",
                            "area": "Links House",
                            "nodeType": "item",
                            "nodeItem": "Lamp",
                            "nodeAddress": "0xE9BC",
                        },
                    ],
                }
            ]
        },
    )
    _write_json(
        root / "regions" / "lightworld" / "northeast" / "regions.json",
        {
            "rooms": [
                {
                    "id": 81,
                    "name": "Hyrule Castle Courtyard",
                    "roomType": "LightWorld",
                    "nodes": [
                        {
                            "id": 1,
                            "name": "Hyrule Castle Entrance (South)",
                            "area": "Hyrule Castle Courtyard",
                            "nodeType": "door",
                        }
                    ],
                }
            ]
        },
    )
    _write_json(
        root / "regions" / "dungeons" / "escape" / "main.json",
        {
            "id": 83,
            "name": "Hyrule Castle",
            "roomType": "Dungeon",
            "nodes": [
                {
                    "id": 3,
                    "name": "Hyrule Castle Exit (South)",
                    "area": "Hyrule Castle",
                    "nodeType": "door",
                }
            ],
        },
    )
    return root


def test_absent_data_raises_actionable_error(tmp_path: Path) -> None:
    missing = tmp_path / "nope"
    with pytest.raises(Z3JsonDataNotFoundError) as excinfo:
        Z3JsonData.load(missing)
    message = str(excinfo.value)
    assert "setup_z3_json_data" in message
    assert str(missing) in message


def test_source_status_absent(tmp_path: Path) -> None:
    status = source_status(tmp_path / "missing")
    assert status.present is False
    assert status.shape_ok is False
    assert status.issues


def test_shape_rejects_incomplete_tree(tmp_path: Path) -> None:
    root = tmp_path / "partial"
    root.mkdir()
    (root / "items.json").write_text("{}", encoding="utf-8")
    issues = discover_shape_issues(root)
    assert any("missing directory" in i for i in issues)
    with pytest.raises(Z3JsonDataShapeError):
        validate_source_shape(root)


def test_shape_rejects_bad_connections_payload(tmp_path: Path) -> None:
    root = _minimal_tree(tmp_path / "bad_conn")
    _write_json(root / "connections" / "main.json", {"not_connections": []})
    issues = discover_shape_issues(root)
    assert any("connections" in i for i in issues)


def test_load_and_lookup_synthetic(tmp_path: Path) -> None:
    root = _minimal_tree(tmp_path / "z3")
    data = Z3JsonData.load(root)
    assert len(data.rooms) == 3
    house = data.room("Links House")
    assert house.room_type == "Cave"
    assert house.nodes[0].name == "Links House Exit"
    assert house.nodes[1].node_item == "Lamp"

    conns = data.find_connections("Links House")
    assert len(conns) == 1
    assert conns[0].origin == "Links House Exit"
    assert conns[0].destination == "Light World"

    swords = data.find_items("L1Sword")
    assert len(swords) == 1
    assert swords[0].category == "progressives"
    assert swords[0].data == "0x49"

    crows = data.find_enemies("crow")
    assert len(crows) == 1
    assert crows[0].hp == 4

    opening = data.opening_route_rooms()
    names = {r.name for r in opening}
    assert "Links House" in names
    assert "Hyrule Castle" in names
    assert names <= OPENING_ROUTE_ROOM_NAMES

    room_conns = data.connections_for_room(house)
    assert any(c.origin == "Links House Exit" for c in room_conns)

    opening_conns = data.opening_route_connections()
    assert any(c.origin == "Links House Exit" for c in opening_conns)
    assert any("Hyrule Castle" in c.origin for c in opening_conns)


def test_cli_status_and_list_regions(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    root = _minimal_tree(tmp_path / "z3")
    assert main(["--root", str(root), "status"]) == 0
    out = capsys.readouterr().out
    assert "shape_ok:  True" in out

    assert main(["--root", str(root), "list-regions", "--opening"]) == 0
    out = capsys.readouterr().out
    assert "Links House" in out
    assert "Hyrule Castle Courtyard" in out

    assert main(["--root", str(root), "list-items", "-q", "Lamp"]) == 0
    out = capsys.readouterr().out
    assert "Lamp" in out

    assert main(["--root", str(root), "show-room", "Links House"]) == 0
    out = capsys.readouterr().out
    assert "Links House Exit" in out


def test_cli_status_missing(tmp_path: Path) -> None:
    assert main(["--root", str(tmp_path / "absent"), "status"]) == 1


def test_cli_validate_missing(tmp_path: Path) -> None:
    assert main(["--root", str(tmp_path / "absent"), "validate"]) == 1
