"""Tests for the data-driven Link's House → castle opening-route catalog."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from alttp.opening_route_catalog import (
    CATALOG_KIND,
    DISCLAIMER,
    build_catalog_artifact,
    correlate_boot_report,
    main,
    opening_checkpoints,
    validate_against_z3,
)
from alttp.ram import HYRULE_CASTLE_SCREEN, LINKS_HOUSE_SCREEN
from alttp.z3_json_data import Z3JsonData, Z3JsonDataNotFoundError


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def _opening_tree(root: Path, *, include_house_exit_conn: bool = False) -> Path:
    """Synthetic tree covering catalog-required rooms/nodes/connections."""
    connections = [
        {
            "connectionType": "Door",
            "description": "Door connection between Hyrule Castle Main Gate and Hyrule Castle Courtyard",
            "nodes": [
                {"name": "Hyrule Castle Main Gate", "position": "origin"},
                {"name": "Hyrule Castle Courtyard", "position": "destination"},
            ],
        },
        {
            "connectionType": "Door",
            "description": "Door connection between Hyrule Castle Entrance (South) and Hyrule Castle",
            "nodes": [
                {
                    "name": "Hyrule Castle Entrance (South)",
                    "position": "origin",
                },
                {"name": "Hyrule Castle", "position": "destination"},
            ],
        },
        {
            "connectionType": "Door",
            "description": "Door connection between Hyrule Castle Exit (South) and Light World",
            "nodes": [
                {"name": "Hyrule Castle Exit (South)", "position": "origin"},
                {"name": "Light World", "position": "destination"},
            ],
        },
        {
            "connectionType": "Door",
            "description": (
                "Door connection between Hyrule Castle Secret Entrance "
                "Stairs and Hyrule Castle Secret Entrance"
            ),
            "nodes": [
                {
                    "name": "Hyrule Castle Secret Entrance Stairs",
                    "position": "origin",
                },
                {
                    "name": "Hyrule Castle Secret Entrance",
                    "position": "destination",
                },
            ],
        },
    ]
    if include_house_exit_conn:
        connections.append(
            {
                "connectionType": "Door",
                "description": "Door connection between Links House Exit and Light World",
                "nodes": [
                    {"name": "Links House Exit", "position": "origin"},
                    {"name": "Light World", "position": "destination"},
                ],
            }
        )

    _write_json(
        root / "items.json",
        {
            "base": ["GreenMail"],
            "inventory": {},
            "progressives": {},
        },
    )
    _write_json(root / "connections" / "main.json", {"connections": connections})
    _write_json(
        root / "enemies" / "main.json",
        {"enemies": [{"id": 0, "names": ["Crow"], "hp": 4}]},
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
                        },
                    ],
                }
            ]
        },
    )
    _write_json(
        root / "regions" / "lightworld" / "main.json",
        {
            "rooms": [
                {
                    "id": 0,
                    "name": "Light World",
                    "roomType": "LightWorld",
                    "nodes": [
                        {
                            "id": 6,
                            "name": "Links House",
                            "area": "Light World",
                            "nodeType": "door",
                        },
                        {
                            "id": 56,
                            "name": "Hyrule Castle Main Gate",
                            "area": "Light World",
                            "nodeType": "door",
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
                        },
                        {
                            "id": 2,
                            "name": "Hyrule Castle Secret Entrance Stairs",
                            "area": "Hyrule Castle Courtyard",
                            "nodeType": "door",
                        },
                    ],
                },
                {
                    "id": 82,
                    "name": "Hyrule Castle Ledge",
                    "roomType": "LightWorld",
                    "nodes": [],
                },
            ]
        },
    )
    _write_json(
        root / "regions" / "lightworld" / "northeast" / "caves.json",
        {
            "rooms": [
                {
                    "id": 4,
                    "name": "Hyrule Castle Secret Entrance",
                    "roomType": "Cave",
                    "nodes": [
                        {
                            "id": 1,
                            "name": "Hyrule Castle Secret Entrance Exit",
                            "area": "Hyrule Castle Secret Entrance",
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


def test_checkpoints_cover_house_to_castle_goal() -> None:
    cps = {cp.id: cp for cp in opening_checkpoints()}
    assert "links_house_interior" in cps
    assert "hyrule_castle_grounds" in cps
    goal = cps["hyrule_castle_grounds"]
    assert goal.role == "goal"
    assert goal.gameplay["screen_id"] == HYRULE_CASTLE_SCREEN
    assert goal.gameplay["on_castle_grounds"] is True
    porch = cps["links_house_overworld"]
    assert porch.gameplay["screen_id"] == LINKS_HOUSE_SCREEN
    # Explicit non-claim: every checkpoint carries the association disclaimer path.
    assert "logic" in DISCLAIMER.lower() or "NOT" in DISCLAIMER


def test_validate_ok_on_synthetic_tree(tmp_path: Path) -> None:
    root = _opening_tree(tmp_path / "z3")
    data = Z3JsonData.load(root)
    validation = validate_against_z3(data)
    assert validation.required_ok is True
    assert validation.ok is True
    assert "Links House" in validation.rooms_present
    assert "Hyrule Castle Courtyard" in validation.rooms_present
    assert any(
        "Hyrule Castle Main Gate" in c for c in validation.connections_present
    )
    # Optional house-exit edge is absent in this fixture → optional_missing.
    assert any(
        "Links House Exit" in c for c in validation.connections_optional_missing
    )


def test_validate_detects_missing_required_connection(tmp_path: Path) -> None:
    root = _opening_tree(tmp_path / "z3")
    # Drop the required main-gate connection.
    payload = json.loads((root / "connections" / "main.json").read_text())
    payload["connections"] = [
        c
        for c in payload["connections"]
        if not (
            c["nodes"][0]["name"] == "Hyrule Castle Main Gate"
            and c["nodes"][1]["name"] == "Hyrule Castle Courtyard"
        )
    ]
    _write_json(root / "connections" / "main.json", payload)
    data = Z3JsonData.load(root)
    validation = validate_against_z3(data)
    assert validation.required_ok is False
    assert any(
        "Hyrule Castle Main Gate" in c for c in validation.connections_missing
    )


def test_validate_detects_missing_room(tmp_path: Path) -> None:
    root = _opening_tree(tmp_path / "z3")
    (root / "regions" / "lightworld" / "south" / "caves.json").unlink()
    data = Z3JsonData.load(root)
    validation = validate_against_z3(data)
    assert validation.required_ok is False
    assert "Links House" in validation.rooms_missing


def test_missing_data_raises_actionable_error(tmp_path: Path) -> None:
    missing = tmp_path / "absent"
    with pytest.raises(Z3JsonDataNotFoundError) as excinfo:
        Z3JsonData.load(missing)
    message = str(excinfo.value)
    assert "setup_z3_json_data" in message
    assert str(missing) in message


def test_build_artifact_structure_and_disclaimer(tmp_path: Path) -> None:
    root = _opening_tree(tmp_path / "z3")
    data = Z3JsonData.load(root)
    artifact = build_catalog_artifact(data)
    assert artifact["kind"] == CATALOG_KIND
    assert artifact["disclaimer"] == DISCLAIMER
    assert "NOT exact stable-retro" in artifact["disclaimer"]
    assert artifact["goal"]["gameplay_acceptance"]["screen_id"] == (
        HYRULE_CASTLE_SCREEN
    )
    assert artifact["validation"]["required_ok"] is True
    assert artifact["metrics"]["checkpoint_count"] == len(opening_checkpoints())
    for cp in artifact["checkpoints"]:
        assert "coordinate_claim" in cp
        assert "logic associations" in cp["coordinate_claim"]
    assert artifact["observed"] is None


def test_correlate_boot_report_proven_goal_only() -> None:
    good = {
        "phase": "castle_grounds",
        "frames": 500,
        "game_mode": 9,
        "submodule": 0,
        "screen_id": HYRULE_CASTLE_SCREEN,
        "screen_hex": f"0x{HYRULE_CASTLE_SCREEN:02X}",
        "indoors": False,
        "dark_world": False,
        "has_control": True,
        "on_castle_grounds": True,
        "link_x": 2386,
        "link_y": 2528,
    }
    corr = correlate_boot_report(good)
    assert corr["proven_gameplay"] is True
    assert len(corr["observed_milestones"]) == 1
    assert corr["observed_milestones"][0]["checkpoint_id"] == (
        "hyrule_castle_grounds"
    )
    assert corr["observed_milestones"][0]["status"] == "observed_gameplay"
    # Does not invent intermediate screens.
    ids = {m["checkpoint_id"] for m in corr["observed_milestones"]}
    assert "links_house_overworld" not in ids
    assert "overworld_to_castle" not in ids

    bad = {**good, "on_castle_grounds": False, "screen_id": LINKS_HOUSE_SCREEN}
    corr_bad = correlate_boot_report(bad)
    assert corr_bad["proven_gameplay"] is False
    assert corr_bad["observed_milestones"][0]["status"] == "not_confirmed"


def test_build_artifact_with_boot_report(tmp_path: Path) -> None:
    root = _opening_tree(tmp_path / "z3")
    data = Z3JsonData.load(root)
    report = {
        "phase": "castle_grounds",
        "screen_id": HYRULE_CASTLE_SCREEN,
        "indoors": False,
        "dark_world": False,
        "has_control": True,
        "on_castle_grounds": True,
    }
    artifact = build_catalog_artifact(data, boot_report=report)
    assert artifact["observed"]["proven_gameplay"] is True
    assert artifact["metrics"]["proven_gameplay_from_boot_report"] is True


def test_cli_missing_data(tmp_path: Path) -> None:
    assert main(["--root", str(tmp_path / "nope"), "status"]) == 1
    assert main(["--root", str(tmp_path / "nope"), "validate"]) == 1
    assert main(["--root", str(tmp_path / "nope"), "emit"]) == 1


def test_cli_validate_and_emit(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    root = _opening_tree(tmp_path / "z3")
    assert main(["--root", str(root), "validate"]) == 0
    out = capsys.readouterr().out
    assert "required_ok: True" in out

    out_path = tmp_path / "artifact.json"
    boot_path = tmp_path / "boot.json"
    boot_path.write_text(
        json.dumps(
            {
                "phase": "castle_grounds",
                "screen_id": HYRULE_CASTLE_SCREEN,
                "indoors": False,
                "dark_world": False,
                "has_control": True,
                "on_castle_grounds": True,
            }
        ),
        encoding="utf-8",
    )
    assert (
        main(
            [
                "--root",
                str(root),
                "emit",
                "--out",
                str(out_path),
                "--from-boot-report",
                str(boot_path),
                "--require-ok",
            ]
        )
        == 0
    )
    payload = json.loads(out_path.read_text(encoding="utf-8"))
    assert payload["kind"] == CATALOG_KIND
    assert payload["observed"]["proven_gameplay"] is True


def test_cli_list_checkpoints(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["list-checkpoints"]) == 0
    out = capsys.readouterr().out
    assert "hyrule_castle_grounds" in out
    assert "links_house_interior" in out


def test_optional_connection_does_not_fail_required(tmp_path: Path) -> None:
    """Links House Exit → Light World is optional (missing in real pin)."""
    root = _opening_tree(tmp_path / "z3", include_house_exit_conn=False)
    data = Z3JsonData.load(root)
    validation = validate_against_z3(data)
    assert validation.required_ok is True
    assert validation.connections_optional_missing


def test_opening_overworld_route_graph_structure() -> None:
    """Catalog-only graph/legs data — not a boot executor."""
    from alttp.opening_route_data import (
        OVERWORLD_SCREEN_PATH,
        opening_overworld_route_graph,
        opening_overworld_route_legs,
    )

    graph = opening_overworld_route_graph()
    assert len(graph.nodes) == len(OVERWORLD_SCREEN_PATH)
    assert len(graph.edges) == len(OVERWORLD_SCREEN_PATH) - 1
    # Screen path is linear: each edge is consecutive path steps.
    for i, edge in enumerate(graph.edges):
        src = int(OVERWORLD_SCREEN_PATH[i]["screen_id"])
        dst = int(OVERWORLD_SCREEN_PATH[i + 1]["screen_id"])
        assert edge.source_id == f"ow_{src:02x}"
        assert edge.target_id == f"ow_{dst:02x}"
        assert edge.meta["from_screen"] == src
        assert edge.meta["to_screen"] == dst

    legs = opening_overworld_route_legs()
    assert len(legs) == len(OVERWORLD_SCREEN_PATH) - 1
    assert legs[-1].target_id == "ow_1b"
    assert legs[-1].goal == "reach_screen_1B"
    # Nodes carry RAM authority, not z3 labels.
    assert len(graph.nodes) == len(OVERWORLD_SCREEN_PATH)
    for node in graph.nodes.values():
        assert node.meta["authority"] == "stable_retro_ram"
        assert "screen_id" in node.meta
