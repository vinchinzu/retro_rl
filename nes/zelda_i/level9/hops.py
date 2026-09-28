"""Level 9 chapter factories and Survival ``SpineHop`` rows.

Prefix rows stay one-frame fail-closed: Magical Key topology is hypothesized,
but live rooms, post-L8 leftover, and the 0x51 dest walk are unverified.
Fixture-live dest hops 0x76 UP → 0x66 and 0x66 LEFT → 0x65 live in
``level9.prefix``; they do not green these factories.
The credits row is a write-free adapter; it must not load a fixture.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from zelda_i.dungeon.shot_guard import GuardedController
from zelda_i.level9.dungeon import (
    L9_CREDITS_ENDPOINT,
    L9_ENTRY_ENDPOINT,
    L9_PATRA_ENDPOINT,
    L9_SELECTED_JOIN_ROOMS,
    L9_SELECTED_PREFIX_ROOMS,
    L9_SILVER_ARROWS_ENDPOINT,
    PostLevel8Handoff,
    ROOM_SILVER_ARROWS_HYP,
    ROOM_SUFFIX_JOIN,
    UNMEASURED_POST_L8_HANDOFF,
    level9_credits_stop,
    level9_entry_stop,
    level9_live_patra_stop,
    level9_silver_arrows_stop,
)
from zelda_i.overworld.bomb_shop import BOMB_SHOP_SCREEN, bomb_restock_stages
from zelda_i.level9.overworld import (
    POST_L8_TO_LEVEL9_HOPS,
    POST_L8_VIA_BOMB_SHOP_HOPS,
    SCREEN_LEVEL9_ROCK_HYP,
)
from zelda_i.overworld.white_sword import make_white_sword_detour_controller
from zelda_i.level9.natural_path import (
    NaturalCreditsController,
    NaturalEnterZeldaController,
    NaturalFinalPatraController,
    NaturalGanonController,
    NaturalPatraToGanonController,
    NaturalPowerTriforceController,
    NaturalRescueZeldaController,
    NaturalSelectSilverArrowsController,
    NaturalSilverArrowsController,
    make_natural_patra_join_controller,
    make_natural_silver_arrows_controller,
    make_patra_join_unavailable_controller,
    make_post_l8_overworld_controller,
    make_silver_arrows_unavailable_controller,
    make_spectacle_rock_bomb_controller,
)
from zelda_i.ram import ADDR_MAGIC_KEY, read_u8
from zelda_i.rollout import PolicyGuard
from zelda_i.spine.hops import SpineHop


@dataclass(frozen=True)
class Level9NaturalRouteSelection:
    """Decoded Magical Key route. Live rooms still missing; not route-eligible."""

    topology_decoded: bool = False
    silver_arrow_room: int | None = None
    suffix_join_room: int | None = None
    requires_51_to_41: bool | None = None
    prefix_rooms: tuple[int, ...] = ()
    join_rooms: tuple[int, ...] = ()
    red_ring_included: bool = False
    evidence: str = "hypothesis"
    route_eligible: bool = False


UNSELECTED_NATURAL_ROUTE = Level9NaturalRouteSelection()

# Hypothesis graph only. 0x51 dest walk is unverified; 0x62 is Keese, not Patra south.
SELECTED_NATURAL_ROUTE = Level9NaturalRouteSelection(
    topology_decoded=True,
    silver_arrow_room=ROOM_SILVER_ARROWS_HYP,
    suffix_join_room=ROOM_SUFFIX_JOIN,
    requires_51_to_41=True,
    prefix_rooms=L9_SELECTED_PREFIX_ROOMS,
    join_rooms=L9_SELECTED_JOIN_ROOMS,
    red_ring_included=False,
    evidence="hypothesis",
    route_eligible=False,
)


def _stage(name: str, controller) -> tuple[str, Any, int]:
    return (name, controller, controller.max_frames)


from zelda_i.overworld.graph import ScreenHop

# Level 9's walls: Spectacle Rock and 0x65 north before room 0x16, then
# 0x06 west, 0x20 north, 0x31 west and 0x04 west. Clean Link leaves Level 8
# with 4 bombs and 2R, and 0x16's Patra item (+4, ``level9_patra_16``) pays
# for the four walls after it. The two 0x4A packs are headroom: bought when
# the wallet can pay, skipped when it cannot, and skipped when a natural
# drop on 0x5D already filled the bag to this count.
LEVEL9_BOMBS_WANTED = 8

# Legacy 0x67 hops exported for test compatibility
L9_RUPEES_67_HOPS: tuple[ScreenHop, ...] = (
    ScreenHop(0x49, "LEFT", align_y=141),
    ScreenHop(0x59, "DOWN", align_x=112),
    ScreenHop(0x58, "LEFT", y_band_lo=148, y_band_hi=162),
    ScreenHop(0x68, "DOWN", align_x=48),
    ScreenHop(0x78, "DOWN", align_x=48),
    ScreenHop(0x77, "LEFT", align_y=141),
    ScreenHop(0x67, "UP", align_x=112),
)
L9_RUPEES_67_RETURN_HOPS: tuple[ScreenHop, ...] = (
    ScreenHop(0x77, "DOWN", align_x=112),
    ScreenHop(0x78, "RIGHT", align_y=141),
    ScreenHop(0x68, "UP", align_x=48),
    ScreenHop(0x58, "UP", align_x=48),
    ScreenHop(0x59, "RIGHT", y_band_lo=148, y_band_hi=162),
    ScreenHop(0x49, "UP", align_x=112),
    ScreenHop(0x4A, "RIGHT", align_y=141),
)


def level9_entry_chapter(
    route: Level9NaturalRouteSelection = SELECTED_NATURAL_ROUTE,
    *,
    handoff: PostLevel8Handoff = UNMEASURED_POST_L8_HANDOFF,
    post_l8_hops: tuple[Any, ...] = POST_L8_VIA_BOMB_SHOP_HOPS,
) -> tuple[tuple[str, Any, int], ...]:
    """Post-L8 OW → two bomb packs at 0x4A → White Sword detour → Spectacle Rock → L9.

    The packs are skipped when the wallet is short (``LEVEL9_BOMBS_WANTED``).

    The detour slots in here because the post-L8 overworld leg already ends on
    0x05, the screen it departs from and returns to, and because Level 9's
    ending contracts need a sword upgrade Link does not otherwise have: the
    power-on run arrives with the wooden sword and 10 heart containers.
    """
    del route
    targets = [hop.target for hop in post_l8_hops]
    if BOMB_SHOP_SCREEN in targets:
        shop_idx = targets.index(BOMB_SHOP_SCREEN)
        to_shop = post_l8_hops[: shop_idx + 1]
        post_l8_to_shop = make_post_l8_overworld_controller(
            handoff=handoff,
            hops=to_shop,
            stop_screen=BOMB_SHOP_SCREEN,
            bomb_goal=LEVEL9_BOMBS_WANTED,
        )
        from_shop = post_l8_hops[shop_idx + 1 :]
        post_l8_to_rock = make_post_l8_overworld_controller(
            handoff=handoff,
            hops=from_shop,
            stop_screen=SCREEN_LEVEL9_ROCK_HYP,
            resumed=True,
        )
        walk = (
            _stage("level9_post_l8_overworld", post_l8_to_shop),
            *bomb_restock_stages(
                to_shop, "l8", want=LEVEL9_BOMBS_WANTED,
                shop_screen=BOMB_SHOP_SCREEN, skip_unaffordable=True,
            ),
            *bomb_restock_stages(
                to_shop, "l8_second", want=LEVEL9_BOMBS_WANTED,
                shop_screen=BOMB_SHOP_SCREEN, skip_unaffordable=True,
            ),
            # 0x58-0x06 bleed 1.75-4 hearts to Lynels, peahats and Zoras:
            # the walk is checked on the ROM like the rock bomb below.
            _stage("level9_post_l8_to_rock", PolicyGuard(post_l8_to_rock)),
        )
    else:
        walk = (
            _stage(
                "level9_post_l8_overworld",
                make_post_l8_overworld_controller(handoff=handoff, hops=post_l8_hops),
            ),
        )
    return (
        *walk,
        _stage("level9_white_sword", make_white_sword_detour_controller()),
        # The rock bomb is a hand phase machine among Lynels and Leevers
        # (3.5-5.5 hearts a run on 0x05, all body contact): the ROM checks
        # its own next frames and detours when they meet a hit.
        _stage(
            "level9_spectacle_rock_bomb",
            PolicyGuard(make_spectacle_rock_bomb_controller(handoff)),
        ),
    )


def level9_silver_arrows_chapter(
    route: Level9NaturalRouteSelection = SELECTED_NATURAL_ROUTE,
    *,
    handoff: PostLevel8Handoff = UNMEASURED_POST_L8_HANDOFF,
) -> tuple[tuple[str, Any, int], ...]:
    if route.silver_arrow_room is None:
        controller = make_silver_arrows_unavailable_controller()
        controller.reason = "silver_arrow_room_not_selected"
        return (_stage("level9_natural_silver_arrows", controller),)
    controller = GuardedController(
        make_natural_silver_arrows_controller(
            handoff=handoff, red_ring=route.red_ring_included
        )
    )
    return (_stage("level9_natural_silver_arrows", controller),)


def level9_patra_chapter(
    route: Level9NaturalRouteSelection = SELECTED_NATURAL_ROUTE,
) -> tuple[tuple[str, Any, int], ...]:
    if route.suffix_join_room is None:
        controller = make_patra_join_unavailable_controller()
        controller.reason = "natural_suffix_join_not_selected"
        return (_stage("level9_natural_patra_join", controller),)
    controller = GuardedController(make_natural_patra_join_controller())
    return (_stage("level9_natural_patra_join", controller),)


def level9_credits_chapter() -> tuple[tuple[str, Any, int], ...]:
    """Fresh write-free controllers from exact live Patra to credits.

    Must not load a fixture or compose inventory. Ganon selects arrows only
    through the pause-menu cursor; never ``ADDR_SELECTED_ITEM`` assign.
    """
    select_arrows = NaturalSelectSilverArrowsController()
    patra = GuardedController(NaturalFinalPatraController())
    enter_ganon = NaturalPatraToGanonController()
    ganon = GuardedController(NaturalGanonController())
    power = NaturalPowerTriforceController()
    enter_zelda = NaturalEnterZeldaController()
    rescue = NaturalRescueZeldaController()
    credits = NaturalCreditsController()
    return (
        ("level9_select_silver_arrows", select_arrows, select_arrows.max_frames),
        ("level9_final_patra", patra, patra.max_frames),
        ("level9_enter_ganon", enter_ganon, enter_ganon.max_frames),
        ("level9_ganon", ganon, ganon.max_frames),
        ("level9_power_triforce", power, power.max_frames),
        ("level9_enter_zelda", enter_zelda, enter_zelda.max_frames),
        ("level9_rescue_zelda", rescue, rescue.max_frames),
        ("level9_wait_credits", credits, credits.max_frames),
    )


def l9_hops(
    env: Any | None,
    *,
    route: Level9NaturalRouteSelection = SELECTED_NATURAL_ROUTE,
    handoff: PostLevel8Handoff = UNMEASURED_POST_L8_HANDOFF,
) -> tuple[SpineHop, ...]:
    """Return the four public L9 chapter rows."""

    def entry_ok(snap, **_):
        magic_key = bool(
            env is not None and read_u8(env.get_ram(), ADDR_MAGIC_KEY) > 0
        )
        return level9_entry_stop(snap, magic_key=magic_key)

    def credits_ok(snap, **_):
        deaths = 0
        if env is not None:
            assist = getattr(env, "assist", None)
            telemetry = getattr(assist, "telemetry", None)
            deaths = int(getattr(telemetry, "deaths", 0) or 0)
        return level9_credits_stop(snap, deaths=deaths)

    return (
        SpineHop(
            L9_ENTRY_ENDPOINT.through,
            L9_ENTRY_ENDPOINT.stop,
            lambda: level9_entry_chapter(route, handoff=handoff),
            entry_ok,
        ),
        SpineHop(
            L9_SILVER_ARROWS_ENDPOINT.through,
            L9_SILVER_ARROWS_ENDPOINT.stop,
            lambda: level9_silver_arrows_chapter(route, handoff=handoff),
            lambda snap, **_: level9_silver_arrows_stop(
                snap,
                room=route.silver_arrow_room,
            ),
        ),
        SpineHop(
            L9_PATRA_ENDPOINT.through,
            L9_PATRA_ENDPOINT.stop,
            lambda: level9_patra_chapter(route),
            lambda snap, **_: level9_live_patra_stop(snap),
        ),
        SpineHop(
            L9_CREDITS_ENDPOINT.through,
            L9_CREDITS_ENDPOINT.stop,
            level9_credits_chapter,
            credits_ok,
        ),
    )


__all__ = [
    "Level9NaturalRouteSelection",
    "NaturalSilverArrowsController",
    "SELECTED_NATURAL_ROUTE",
    "UNSELECTED_NATURAL_ROUTE",
    "level9_credits_chapter",
    "level9_entry_chapter",
    "level9_patra_chapter",
    "level9_silver_arrows_chapter",
    "l9_hops",
    "make_natural_silver_arrows_controller",
]
