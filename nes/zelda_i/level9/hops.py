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

from zelda_i.level9.dungeon import (
    L9_CREDITS_ENDPOINT,
    L9_ENTRY_ENDPOINT,
    L9_PATRA_ENDPOINT,
    L9_SELECTED_JOIN_ROOMS,
    L9_SELECTED_PREFIX_ROOMS,
    L9_SILVER_ARROWS_ENDPOINT,
    MISSING_51_NORTH_WALK,
    MISSING_SILVER_ARROW_ROOM,
    PostLevel8Handoff,
    ROOM_SILVER_ARROWS_HYP,
    ROOM_SUFFIX_JOIN,
    UNMEASURED_POST_L8_HANDOFF,
    level9_credits_stop,
    level9_entry_stop,
    level9_live_patra_stop,
    level9_silver_arrows_stop,
)
from zelda_i.overworld.white_sword import make_white_sword_detour_controller
from zelda_i.level9.natural_path import (
    NaturalCreditsController,
    NaturalEnterZeldaController,
    NaturalFinalPatraController,
    NaturalGanonController,
    NaturalPatraJoinController,
    NaturalPatraToGanonController,
    NaturalPowerTriforceController,
    NaturalRescueZeldaController,
    NaturalSelectSilverArrowsController,
    NaturalSilverArrowsController,
    make_natural_patra_join_controller,
    make_natural_silver_arrows_controller,
    make_old_man_tf_gate_controller,
    make_patra_join_unavailable_controller,
    make_post_l8_overworld_controller,
    make_silver_arrows_unavailable_controller,
    make_spectacle_rock_bomb_controller,
)
from zelda_i.ram import ADDR_MAGIC_KEY, read_u8
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


def level9_entry_chapter(
    route: Level9NaturalRouteSelection = SELECTED_NATURAL_ROUTE,
    *,
    handoff: PostLevel8Handoff = UNMEASURED_POST_L8_HANDOFF,
) -> tuple[tuple[str, Any, int], ...]:
    """Post-L8 OW → White Sword detour → Spectacle Rock bomb → L9 room 0x76.

    The detour slots in here because the post-L8 overworld leg already ends on
    0x05, the screen it departs from and returns to, and because Level 9's
    ending contracts need a sword upgrade Link does not otherwise have: the
    power-on run arrives with the wooden sword and 10 heart containers.
    """
    del route
    return (
        _stage("level9_post_l8_overworld", make_post_l8_overworld_controller(handoff)),
        _stage("level9_white_sword", make_white_sword_detour_controller()),
        _stage("level9_spectacle_rock_bomb", make_spectacle_rock_bomb_controller(handoff)),
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
    return (_stage("level9_natural_silver_arrows", make_natural_silver_arrows_controller(handoff=handoff)),)


def level9_patra_chapter(
    route: Level9NaturalRouteSelection = SELECTED_NATURAL_ROUTE,
) -> tuple[tuple[str, Any, int], ...]:
    if route.suffix_join_room is None:
        controller = make_patra_join_unavailable_controller()
        controller.reason = "natural_suffix_join_not_selected"
        return (_stage("level9_natural_patra_join", controller),)
    return (_stage("level9_natural_patra_join", make_natural_patra_join_controller()),)


def level9_credits_chapter() -> tuple[tuple[str, Any, int], ...]:
    """Fresh write-free controllers from exact live Patra to credits.

    Must not load a fixture or compose inventory. Ganon selects arrows only
    through the pause-menu cursor; never ``ADDR_SELECTED_ITEM`` assign.
    """
    select_arrows = NaturalSelectSilverArrowsController()
    patra = NaturalFinalPatraController()
    enter_ganon = NaturalPatraToGanonController()
    ganon = NaturalGanonController()
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
