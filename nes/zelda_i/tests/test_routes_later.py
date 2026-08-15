"""Tests for L3–L9 NamedRoutes (first-class L3–L5 + L9 fixture; L6–L8 stubs)."""

from __future__ import annotations

from zelda_i.later_nodes import (
    DOOR_SCREEN_BY_LEVEL,
    NODE_LEVEL3_WEST_KEY,
    NODE_LEVEL4_STEPLADDER,
    NODE_LEVEL5_WHISTLE,
    NODE_LEVEL9_PATRA,
    NODE_LEVEL9_ROOM_41,
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
    level3_complete_route_plan,
    level4_complete_route_plan,
    level5_complete_route_plan,
    level9_fixture_route_plan,
    list_later_routes,
)


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
    assert ROUTE_REGISTRY_LATER["level4"] is ROUTE_LEVEL4_COMPLETE
    assert ROUTE_REGISTRY_LATER["ganon"] is ROUTE_LEVEL9_GANON


def test_l3_l5_milestones_are_first_class() -> None:
    l3_ids = [m.milestone_id for m in ROUTE_LEVEL3_COMPLETE.milestones]
    l3_preds = [m.stop_predicate for m in ROUTE_LEVEL3_COMPLETE.milestones]
    assert "level3_west_key" in l3_ids
    assert "level3_raft" in l3_ids
    assert "triforce & 0x04" in l3_preds
    assert any(m.node_id == NODE_LEVEL3_WEST_KEY for m in ROUTE_LEVEL3_COMPLETE.milestones)
    assert "PLANNING STUB" not in ROUTE_LEVEL3_COMPLETE.description
    assert "assisted" in ROUTE_LEVEL3_COMPLETE.description.lower()

    l4_ids = [m.milestone_id for m in ROUTE_LEVEL4_COMPLETE.milestones]
    l4_preds = [m.stop_predicate for m in ROUTE_LEVEL4_COMPLETE.milestones]
    assert "level4_dock" in l4_ids
    assert "level4_stepladder" in l4_ids
    assert "triforce & 0x08" in l4_preds
    assert any(m.node_id == NODE_LEVEL4_STEPLADDER for m in ROUTE_LEVEL4_COMPLETE.milestones)
    assert "PLANNING STUB" not in ROUTE_LEVEL4_COMPLETE.description
    assert "raft" in ROUTE_LEVEL4_COMPLETE.description.lower()

    l5_ids = [m.milestone_id for m in ROUTE_LEVEL5_COMPLETE.milestones]
    l5_preds = [m.stop_predicate for m in ROUTE_LEVEL5_COMPLETE.milestones]
    assert "level5_lost_hills" in l5_ids
    assert "level5_key_66" in l5_ids
    assert "level5_whistle" in l5_ids
    assert "triforce & 0x10" in l5_preds
    assert any(m.node_id == NODE_LEVEL5_WHISTLE for m in ROUTE_LEVEL5_COMPLETE.milestones)
    assert "PLANNING STUB" not in ROUTE_LEVEL5_COMPLETE.description


def test_l6_l8_remain_stubs() -> None:
    for route, bit in (
        (ROUTE_LEVEL6_COMPLETE, 0x20),
        (ROUTE_LEVEL7_COMPLETE, 0x40),
        (ROUTE_LEVEL8_COMPLETE, 0x80),
    ):
        preds = [m.stop_predicate for m in route.milestones]
        assert f"triforce & 0x{bit:02x}" in preds
        assert "PLANNING STUB" in route.description


def test_l9_fixture_not_route_ready() -> None:
    desc = ROUTE_LEVEL9_GANON.description.lower()
    assert "route_eligible=false" in desc
    assert "fixture" in desc
    ids = [m.milestone_id for m in ROUTE_LEVEL9_GANON.milestones]
    assert "level9_room_41" in ids
    assert "level9_patra" in ids
    assert "level9_ganon" in ids
    assert any(m.node_id == NODE_LEVEL9_PATRA for m in ROUTE_LEVEL9_GANON.milestones)
    preds = [m.stop_predicate for m in ROUTE_LEVEL9_GANON.milestones]
    assert "level9_ganon_defeated" in preds
    assert "level9_ending" in preds


def test_door_screen_candidates_present() -> None:
    assert DOOR_SCREEN_BY_LEVEL[3] == 0x74
    assert DOOR_SCREEN_BY_LEVEL[4] == 0x45
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


def test_plan_legs_l3_l5_and_l9_fixture() -> None:
    l3 = level3_complete_route_plan()
    assert l3[-1].leg.target_id == ROUTE_LEVEL3_COMPLETE.milestones[-1].node_id
    assert "triforce_shard_3" in l3[-1].capabilities_after
    assert "raft" in l3[-1].capabilities_after

    l4 = level4_complete_route_plan()
    assert l4[-1].leg.target_id == ROUTE_LEVEL4_COMPLETE.milestones[-1].node_id
    assert "triforce_shard_4" in l4[-1].capabilities_after
    assert "stepladder" in l4[-1].capabilities_after
    assert "raft" in l4[0].capabilities_before

    l5 = level5_complete_route_plan()
    assert l5[-1].leg.target_id == ROUTE_LEVEL5_COMPLETE.milestones[-1].node_id
    assert "triforce_shard_5" in l5[-1].capabilities_after
    assert "whistle" in l5[-1].capabilities_after

    l9 = level9_fixture_route_plan()
    assert l9[0].leg.source_id == NODE_LEVEL9_ROOM_41
    assert l9[-1].leg.target_id == ROUTE_LEVEL9_GANON.milestones[-1].node_id
    assert "fixture_only" in l9[0].leg.constraints
    assert "route_eligible=false" in l9[0].leg.constraints
