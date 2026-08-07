"""Light tests for L3–L9 NamedRoute stubs (planning catalog only)."""

from __future__ import annotations

from zelda_i.later_nodes import (
    DOOR_SCREEN_BY_LEVEL,
    TF_BIT_L3,
    TF_BIT_L4,
    TF_BIT_L5,
    TF_BIT_L6,
    TF_BIT_L7,
    TF_BIT_L8,
    TF_BITS_ALL,
    TRIFORCE_BITS_BY_LEVEL,
)
from zelda_i.routes_later import (
    ROUTE_LEVEL3_COMPLETE,
    ROUTE_LEVEL4_COMPLETE,
    ROUTE_LEVEL5_COMPLETE,
    ROUTE_LEVEL6_COMPLETE,
    ROUTE_LEVEL7_COMPLETE,
    ROUTE_LEVEL8_COMPLETE,
    ROUTE_LEVEL9_GANON,
    ROUTE_REGISTRY_LATER,
    get_later_route,
    list_later_routes,
)


# Matches docs/research/DUNGEON_WALKTHROUGHS.md triforce bit map
_WALKTHROUGH_TF_BITS = {
    1: 0x01,
    2: 0x02,
    3: 0x04,
    4: 0x08,
    5: 0x10,
    6: 0x20,
    7: 0x40,
    8: 0x80,
}


def test_triforce_bits_match_walkthrough_doc() -> None:
    assert TRIFORCE_BITS_BY_LEVEL == _WALKTHROUGH_TF_BITS
    assert TF_BIT_L3 == 0x04
    assert TF_BIT_L4 == 0x08
    assert TF_BIT_L5 == 0x10
    assert TF_BIT_L6 == 0x20
    assert TF_BIT_L7 == 0x40
    assert TF_BIT_L8 == 0x80
    assert TF_BITS_ALL == 0xFF
    # All shard bits exclusive and cover 0xFF
    combined = 0
    for bit in TRIFORCE_BITS_BY_LEVEL.values():
        assert combined & bit == 0
        combined |= bit
    assert combined == TF_BITS_ALL


def test_later_routes_importable_and_registered() -> None:
    routes = list_later_routes()
    ids = {r.route_id for r in routes}
    assert ids == {
        "zelda_level3_complete",
        "zelda_level4_complete",
        "zelda_level5_complete",
        "zelda_level6_complete",
        "zelda_level7_complete",
        "zelda_level8_complete",
        "zelda_level9_ganon",
    }
    assert get_later_route("level3") is ROUTE_LEVEL3_COMPLETE
    assert get_later_route("level9_ganon") is ROUTE_LEVEL9_GANON
    assert get_later_route("triforce_5") is ROUTE_LEVEL5_COMPLETE
    # Registry aliases point at same objects
    assert ROUTE_REGISTRY_LATER["level4"] is ROUTE_LEVEL4_COMPLETE
    assert ROUTE_REGISTRY_LATER["ganon"] is ROUTE_LEVEL9_GANON


def test_stub_milestones_include_tf_stop_predicates() -> None:
    cases = (
        (ROUTE_LEVEL3_COMPLETE, 0x04),
        (ROUTE_LEVEL4_COMPLETE, 0x08),
        (ROUTE_LEVEL5_COMPLETE, 0x10),
        (ROUTE_LEVEL6_COMPLETE, 0x20),
        (ROUTE_LEVEL7_COMPLETE, 0x40),
        (ROUTE_LEVEL8_COMPLETE, 0x80),
    )
    for route, bit in cases:
        preds = [m.stop_predicate for m in route.milestones]
        assert f"triforce & 0x{bit:02x}" in preds
        assert any(m.milestone_id.endswith("_entrance") for m in route.milestones)
        assert "PLANNING STUB" in route.description or "not route-ready" in route.description.lower()

    l9_preds = [m.stop_predicate for m in ROUTE_LEVEL9_GANON.milestones]
    assert f"triforce == 0x{TF_BITS_ALL:02x}" in l9_preds
    assert "level9_ganon_defeated" in l9_preds


def test_door_screen_candidates_present() -> None:
    # Source-derived table; live probes may supersede OVERWORLD_DOORS.md
    assert DOOR_SCREEN_BY_LEVEL[3] == 0x74
    assert DOOR_SCREEN_BY_LEVEL[4] == 0x45  # island hyp (dock 0x55)
    assert DOOR_SCREEN_BY_LEVEL[5] == 0x0B
    assert DOOR_SCREEN_BY_LEVEL[6] == 0x22
    assert DOOR_SCREEN_BY_LEVEL[7] == 0x42
    assert DOOR_SCREEN_BY_LEVEL[8] == 0x6D
    assert DOOR_SCREEN_BY_LEVEL[9] == 0x05
    assert set(DOOR_SCREEN_BY_LEVEL) == {3, 4, 5, 6, 7, 8, 9}


def test_stub_routes_do_not_collide_with_l1_l2_ids() -> None:
    from zelda_i.routes import ROUTE_REGISTRY

    later_ids = {r.route_id for r in list_later_routes()}
    early_ids = {r.route_id for r in ROUTE_REGISTRY.values()}
    assert later_ids.isdisjoint(early_ids)
