"""Unit tests for early OW item-gate hop tables (structure only)."""

from __future__ import annotations

import numpy as np

from zelda_i.item_gate_hops import (
    BOMB_SHOP_HOPS,
    BOMB_SHOP_SCREENS,
    CANDLE_SHOP_MOUNTAIN_HOPS,
    CANDLE_SHOP_NEAR_HOPS,
    CANDLE_SHOP_NEAR_SCREENS,
    ITEM_GATE_ROUTES,
    SCREEN_BOMB_SHOP,
    SCREEN_CANDLE_SHOP_MOUNTAIN,
    SCREEN_CANDLE_SHOP_NEAR,
    SCREEN_WHITE_SWORD_CAVE,
    WHITE_SWORD_HOPS,
    WHITE_SWORD_MIN_CONTAINERS,
    WHITE_SWORD_SCREENS,
    ItemGateHopController,
    gate_report_snapshot,
    hops_are_neighbors,
    route_for,
    screen_reached,
    white_sword_containers_ok,
    white_sword_heart_gate_blocks,
)
from zelda_i.overworld import SCREEN_START, neighbor_screens
from zelda_i.ram import (
    ADDR_HEALTH,
    ADDR_LEVEL,
    ADDR_LINK_X,
    ADDR_LINK_Y,
    ADDR_MODE,
    ADDR_SCREEN,
    ADDR_SWORD,
    PLAY_MODE,
    read_snapshot,
)


def _ram(**fields: int) -> np.ndarray:
    ram = np.zeros(0x800, dtype=np.uint8)
    ram[ADDR_MODE] = fields.get("mode", PLAY_MODE)
    ram[ADDR_LEVEL] = fields.get("level", 0)
    ram[ADDR_SCREEN] = fields.get("screen", SCREEN_START)
    ram[ADDR_LINK_X] = fields.get("x", 120)
    ram[ADDR_LINK_Y] = fields.get("y", 140)
    ram[ADDR_SWORD] = fields.get("sword", 1)
    # health: high nibble = containers-1, low = filled
    # default 3 containers full → 0x2F
    ram[ADDR_HEALTH] = fields.get("health", 0x2F)
    return ram


def test_all_routes_registered() -> None:
    assert set(ITEM_GATE_ROUTES) == {
        "candle_shop_near",
        "candle_shop_mountain",
        "white_sword",
        "bomb_shop",
    }


def test_candle_near_screens_chain() -> None:
    assert CANDLE_SHOP_NEAR_SCREENS[0] == SCREEN_START == 0x77
    assert CANDLE_SHOP_NEAR_SCREENS[-1] == SCREEN_CANDLE_SHOP_NEAR == 0x66
    assert len(CANDLE_SHOP_NEAR_HOPS) == len(CANDLE_SHOP_NEAR_SCREENS) - 1
    assert hops_are_neighbors(CANDLE_SHOP_NEAR_HOPS, SCREEN_START)
    # Trap: must not go via 0x67 west (dead-end north of start).
    assert 0x67 not in CANDLE_SHOP_NEAR_SCREENS
    # Live: 0x56 has no south exit — approach 0x66 via 0x65 east.
    assert 0x65 in CANDLE_SHOP_NEAR_SCREENS
    assert 0x55 in CANDLE_SHOP_NEAR_SCREENS
    for a, b in zip(CANDLE_SHOP_NEAR_SCREENS, CANDLE_SHOP_NEAR_SCREENS[1:]):
        assert b in neighbor_screens(a).values(), f"{a:02x}->{b:02x}"


def test_white_sword_screens_chain() -> None:
    assert WHITE_SWORD_SCREENS[0] == 0x77
    # Hop table ends on Lost Hills; controller finishes on region 0x0B.
    assert WHITE_SWORD_SCREENS[-1] == 0x1B
    assert SCREEN_WHITE_SWORD_CAVE == 0x0A  # source cave residual
    assert hops_are_neighbors(WHITE_SWORD_HOPS, SCREEN_START)
    assert 0x79 not in WHITE_SWORD_SCREENS
    assert 0x4A in WHITE_SWORD_SCREENS  # L2-style join then L5 north
    assert 0x3A in WHITE_SWORD_SCREENS
    assert route_for("white_sword").end == 0x0B
    assert 0x48 not in WHITE_SWORD_SCREENS


def test_bomb_shop_screens_chain() -> None:
    assert BOMB_SHOP_SCREENS[0] == 0x77
    assert BOMB_SHOP_SCREENS[-1] == SCREEN_BOMB_SHOP == 0x4A
    assert hops_are_neighbors(BOMB_SHOP_HOPS, SCREEN_START)
    assert 0x79 not in BOMB_SHOP_SCREENS
    # L2-style early corridor (no 0x5C maze on primary bomb shop).
    assert 0x59 in BOMB_SHOP_SCREENS


def test_mountain_candle_placeholder_uses_near_path() -> None:
    """M-1 0x0C residual; registered route reuses near-start candle hops."""
    route = route_for("candle_shop_mountain")
    assert route.end == SCREEN_CANDLE_SHOP_NEAR == 0x66
    assert hops_are_neighbors(CANDLE_SHOP_MOUNTAIN_HOPS, SCREEN_START)
    assert SCREEN_CANDLE_SHOP_MOUNTAIN == 0x0C  # source target remains


def test_routes_mark_verification_and_structure() -> None:
    for name, route in ITEM_GATE_ROUTES.items():
        assert route.verification in ("planned", "assisted", "observed")
        assert route.screens[0] == route.start
        assert len(route.hops) == len(route.screens) - 1
        if name == "white_sword":
            # Hop table ends on Lost Hills; route.end is live region 0x0B.
            assert route.screens[-1] == 0x1B
            assert route.end == 0x0B
        elif name == "candle_shop_mountain":
            # Placeholder reuses near-start candle (0x66) until M-1 is live.
            assert route.end == SCREEN_CANDLE_SHOP_NEAR
            assert route.screens[-1] == route.end
        else:
            assert route.screens[-1] == route.end
    # Candle near-start promoted after live assisted settle (ig3).
    assert ITEM_GATE_ROUTES["candle_shop_near"].verification == "assisted"


def test_white_sword_heart_gate_uses_containers_not_fill() -> None:
    # 3 containers (high nibble 2), full fill 0xF → still blocked.
    snap3 = read_snapshot(_ram(health=0x2F))
    assert snap3.heart_containers == 3
    assert white_sword_heart_gate_blocks(snap3)
    assert not white_sword_containers_ok(snap3)

    # 5 containers (high nibble 4), even low fill → gate open.
    snap5 = read_snapshot(_ram(health=0x40))  # 5 containers, 0 filled nibble
    assert snap5.heart_containers == 5
    assert white_sword_containers_ok(snap5)
    assert not white_sword_heart_gate_blocks(snap5)

    # Assist-style full fill with 3 containers still blocked.
    snap_assist = read_snapshot(_ram(health=0x2F))
    assert white_sword_heart_gate_blocks(snap_assist)
    assert WHITE_SWORD_MIN_CONTAINERS == 5


def test_screen_reached_predicate() -> None:
    assert screen_reached(_ram(screen=0x66, sword=1, y=140), 0x66)
    assert not screen_reached(_ram(screen=0x66, sword=0), 0x66)
    assert not screen_reached(_ram(screen=0x77, sword=1), 0x66)


def test_controller_defaults_candle() -> None:
    nav = ItemGateHopController(route_name="candle_shop_near")
    assert nav.hops[-1].target == 0x66
    assert nav.end_screen() == 0x66
    report = nav.report()
    assert report["route_name"] == "candle_shop_near"
    assert report["verification"] == "assisted"


def test_controller_white_sword_ends_on_region() -> None:
    nav = ItemGateHopController(route_name="white_sword")
    assert nav.hops[-1].target == 0x1B
    assert nav.end_screen() == 0x0B
    assert nav._wants_post_hop()


def test_controller_bomb_shop_defaults() -> None:
    nav = ItemGateHopController(route_name="bomb_shop")
    assert nav.hops[-1].target == 0x4A
    assert not nav.maze_waypoints


def test_gate_report_snapshot_fields() -> None:
    ram = _ram(screen=0x0A, health=0x2F, sword=1)
    snap = read_snapshot(ram)
    d = gate_report_snapshot(snap, ram)
    assert d["heart_containers"] == 3
    assert d["white_sword_heart_gate_blocks"] is True
    assert d["screen_hex"] == "0x0a"
