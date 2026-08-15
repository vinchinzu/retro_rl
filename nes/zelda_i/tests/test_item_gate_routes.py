"""NamedRoute / RouteLeg catalog for early OW item gates."""

from __future__ import annotations

import pytest

from zelda_i.item_gate_hops import (
    CANDLE_SHOP_NEAR_HOPS,
    CANDLE_SHOP_NEAR_SCREENS,
    CANDLE_SHOP_PRICE_SOURCE,
    WHITE_SWORD_MIN_CONTAINERS,
)
from zelda_i.item_gate_routes import (
    CAP_BLUE_CANDLE,
    CAP_BOMBS,
    CAP_WHITE_SWORD,
    NODE_CANDLE_SHOP_CAVE,
    NODE_CANDLE_SHOP_NEAR,
    NODE_WHITE_SWORD_CAVE,
    NODE_WHITE_SWORD_REGION,
    ROUTE_BOMB_SHOP,
    ROUTE_CANDLE_SHOP,
    ROUTE_WHITE_SWORD,
    WHITE_SWORD_CONTAINER_GATE,
    bomb_shop_route_plan,
    build_item_gate_route_graph,
    candle_shop_route_legs,
    candle_shop_route_plan,
    get_item_gate_route,
    list_item_gate_routes,
    white_sword_route_legs,
    white_sword_route_plan,
)
from zelda_i.overworld import NODE_START, SCREEN_START, node_id_for_screen


def test_registry_lookup_and_aliases() -> None:
    routes = list_item_gate_routes()
    assert {r.route_id for r in routes} == {
        "zelda_candle_shop",
        "zelda_white_sword",
        "zelda_bomb_shop",
    }
    assert get_item_gate_route("zelda_candle_shop") is ROUTE_CANDLE_SHOP
    assert get_item_gate_route("candle") is ROUTE_CANDLE_SHOP
    assert get_item_gate_route("candle_shop_near") is ROUTE_CANDLE_SHOP
    assert get_item_gate_route("white_sword") is ROUTE_WHITE_SWORD
    assert get_item_gate_route("white") is ROUTE_WHITE_SWORD
    assert get_item_gate_route("bomb_shop") is ROUTE_BOMB_SHOP
    assert get_item_gate_route("bombs") is ROUTE_BOMB_SHOP


def test_item_gate_ids_do_not_collide_with_dungeon_catalogs() -> None:
    from zelda_i.routes import list_routes
    from zelda_i.routes_later import list_later_routes

    ours = {r.route_id for r in list_item_gate_routes()}
    early = {r.route_id for r in list_routes()}
    later = {r.route_id for r in list_later_routes()}
    assert ours.isdisjoint(early)
    assert ours.isdisjoint(later)


def test_no_clean_claim_in_descriptions() -> None:
    for route in list_item_gate_routes():
        assert "clean" not in route.description.lower()


def test_candle_hops_match_near_table() -> None:
    legs = candle_shop_route_legs()
    hop_legs = [leg for leg in legs if leg.leg_id.startswith("ow_")]
    assert CANDLE_SHOP_NEAR_SCREENS[0] == SCREEN_START == 0x77
    assert CANDLE_SHOP_NEAR_SCREENS[-1] == 0x66
    assert len(hop_legs) == len(CANDLE_SHOP_NEAR_HOPS)
    for leg, src, hop in zip(
        hop_legs, CANDLE_SHOP_NEAR_SCREENS, CANDLE_SHOP_NEAR_HOPS
    ):
        assert hop.target == CANDLE_SHOP_NEAR_SCREENS[
            CANDLE_SHOP_NEAR_SCREENS.index(src) + 1
        ]
        assert leg.source_id == node_id_for_screen(src)
        assert leg.target_id == node_id_for_screen(hop.target)
    assert legs[-1].target_id == NODE_CANDLE_SHOP_CAVE
    assert CAP_BLUE_CANDLE in legs[-1].acquires
    assert f"costs_{CANDLE_SHOP_PRICE_SOURCE}_rupees" in legs[-1].constraints


def test_candle_plan_acquires_blue_candle() -> None:
    planned = candle_shop_route_plan()
    assert planned[0].leg.source_id == NODE_START
    assert planned[-1].leg.target_id == NODE_CANDLE_SHOP_CAVE
    assert CAP_BLUE_CANDLE in planned[-1].capabilities_after
    assert NODE_CANDLE_SHOP_NEAR in {leg.leg.target_id for leg in planned}


def test_bomb_plan_acquires_inventory_bombs_not_capacity() -> None:
    planned = bomb_shop_route_plan()
    assert CAP_BOMBS in planned[-1].capabilities_after
    assert "bomb_capacity" not in planned[-1].capabilities_after
    assert "level5" not in ROUTE_BOMB_SHOP.route_id
    assert "level7" not in ROUTE_BOMB_SHOP.route_id
    assert "capacity" in ROUTE_BOMB_SHOP.description.lower()
    assert "level5" in ROUTE_BOMB_SHOP.description.lower()
    assert "level7" in ROUTE_BOMB_SHOP.description.lower()


def test_white_sword_container_gate_is_explicit() -> None:
    assert WHITE_SWORD_CONTAINER_GATE == f"heart_containers>={WHITE_SWORD_MIN_CONTAINERS}"
    assert WHITE_SWORD_MIN_CONTAINERS == 5

    legs = white_sword_route_legs()
    acquire = next(leg for leg in legs if CAP_WHITE_SWORD in leg.acquires)
    assert acquire.target_id == NODE_WHITE_SWORD_CAVE
    assert WHITE_SWORD_CONTAINER_GATE in acquire.requires

    preds = [m.stop_predicate for m in ROUTE_WHITE_SWORD.milestones]
    assert WHITE_SWORD_CONTAINER_GATE in preds
    assert WHITE_SWORD_CONTAINER_GATE in ROUTE_WHITE_SWORD.description
    assert "nibble" in ROUTE_WHITE_SWORD.description.lower()

    graph = build_item_gate_route_graph()
    with pytest.raises(ValueError, match="heart_containers"):
        graph.plan_legs(legs, initial_capabilities=frozenset())
    # Assist fill / infinite-life is not the container gate.
    for fake in (
        frozenset({"infinite_life"}),
        frozenset({"filled_hearts"}),
        frozenset({"heart_containers>=3"}),
        frozenset({"heart_containers"}),
    ):
        with pytest.raises(ValueError, match="heart_containers"):
            graph.plan_legs(legs, initial_capabilities=fake)

    planned = graph.plan_legs(
        legs,
        initial_capabilities=frozenset({WHITE_SWORD_CONTAINER_GATE}),
    )
    assert planned[-1].leg.target_id == NODE_WHITE_SWORD_CAVE
    assert CAP_WHITE_SWORD in planned[-1].capabilities_after
    assert WHITE_SWORD_CONTAINER_GATE in planned[-1].capabilities_before
    assert NODE_WHITE_SWORD_REGION in {leg.leg.target_id for leg in planned}


def test_white_sword_route_plan_defaults_to_blocked() -> None:
    with pytest.raises(ValueError, match="heart_containers"):
        white_sword_route_plan()
    planned = white_sword_route_plan(
        initial_capabilities=frozenset({WHITE_SWORD_CONTAINER_GATE}),
    )
    assert CAP_WHITE_SWORD in planned[-1].capabilities_after
