"""ZD Map-1.png cyan overlay is the pre-L1 bomb-shop route. No emulator."""

from __future__ import annotations

from zelda_i.overworld.graph import SCREEN_START, ScreenHop
from zelda_i.overworld.zd_map import MAP1_PATH, MAP1_URL, map1_route


def test_map1_route_does_not_require_the_ignored_reference_png() -> None:
    # MAP1_PATH is only an optional input for manually re-deriving the route;
    # fresh checkouts intentionally do not contain ignored PNG files.
    assert MAP1_PATH.name == "Map-1.png"
    assert map1_route().screens
    assert MAP1_URL.endswith("/Zelda01/Walkthrough/01/Map-1.png")


def test_map1_cyan_path_is_start_along_the_south_coast_to_6f() -> None:
    route = map1_route()
    assert route.source == MAP1_URL
    assert route.screens[0] == SCREEN_START == 0x77
    assert route.dest == 0x6F
    assert route.screens == (
        0x77,
        0x78,
        0x79,
        0x7A,
        0x7B,
        0x7C,
        0x7D,
        0x7E,
        0x7F,
        0x6F,
    )
    assert 0x4A not in route.screens
    assert 0x68 not in route.screens


def test_map1_hops_are_right_eight_then_up_on_the_cyan_lane() -> None:
    hops = map1_route().hops
    assert tuple(h.target for h in hops) == (
        0x78,
        0x79,
        0x7A,
        0x7B,
        0x7C,
        0x7D,
        0x7E,
        0x7F,
        0x6F,
    )
    assert all(h.direction == "RIGHT" for h in hops[:-1])
    assert hops[-1] == ScreenHop(0x6F, "UP", align_x=82)
    # Overlay is playfield centre (Link y≈130), not the y≈180 rocky pocket.
    assert all(120 <= int(h.align_y) <= 145 for h in hops[:-1])


def test_map1_hops_from_7a_is_the_coast_suffix() -> None:
    route = map1_route()
    suffix = route.hops_from(0x7A)
    assert suffix[0].target == 0x7B
    assert suffix[-1].target == 0x6F


def test_shop_p7_walk_uses_map1_screens_not_overlay_79_lane() -> None:
    from zelda_i.overworld.shop_p7 import (
        PRE_L1_BOMB_HOPS,
        SCREEN_79_BEACH_Y,
        SHOP_P7_SCREEN,
        shop_p7_screens,
    )

    route = map1_route()
    assert SHOP_P7_SCREEN == route.dest == 0x6F
    # Overlay names the screens; it does not generate the hop table.
    assert shop_p7_screens() == route.screens
    assert tuple(h.target for h in PRE_L1_BOMB_HOPS) == tuple(
        h.target for h in route.hops
    )
    assert tuple(h.direction for h in PRE_L1_BOMB_HOPS) == tuple(
        h.direction for h in route.hops
    )
    assert PRE_L1_BOMB_HOPS[-1].target == route.dest
    assert PRE_L1_BOMB_HOPS[2].align_y == SCREEN_79_BEACH_Y
    assert route.hops[2].align_y != SCREEN_79_BEACH_Y
    assert any(
        a.target == 0x79 and b.target == 0x7A
        for a, b in zip(PRE_L1_BOMB_HOPS, PRE_L1_BOMB_HOPS[1:])
    )
