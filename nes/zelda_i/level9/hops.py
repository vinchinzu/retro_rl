"""Level 9 chapter factories and Survival ``SpineHop`` rows.

The prefix rows are deliberately fail-closed while natural topology is
undecoded.  The ending row is a write-free adapter around policies already
proven from recon fixtures; it does not promote that old evidence.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from zelda_i.level9.dungeon import (
    L9_CREDITS_ENDPOINT,
    L9_ENTRY_ENDPOINT,
    L9_PATRA_ENDPOINT,
    L9_SILVER_ARROWS_ENDPOINT,
    level9_credits_stop,
    level9_entry_stop,
    level9_live_patra_stop,
    level9_silver_arrows_stop,
)
from zelda_i.level9.natural_path import (
    NaturalCreditsController,
    NaturalEnterZeldaController,
    NaturalFinalPatraController,
    NaturalGanonController,
    NaturalPatraToGanonController,
    NaturalPowerTriforceController,
    NaturalRescueZeldaController,
    NaturalRouteUnavailableController,
    NaturalSelectSilverArrowsController,
)
from zelda_i.ram import ADDR_MAGIC_KEY, read_u8
from zelda_i.spine.hops import SpineHop


@dataclass(frozen=True)
class Level9NaturalRouteSelection:
    """Decoded route inputs required before natural prefix controllers exist."""

    topology_decoded: bool = False
    silver_arrow_room: int | None = None
    suffix_join_room: int | None = None
    requires_51_to_41: bool | None = None
    evidence: str = "hypothesis"
    route_eligible: bool = False


UNSELECTED_NATURAL_ROUTE = Level9NaturalRouteSelection()


def _unavailable_stage(chapter: str, reason: str):
    def stages():
        controller = NaturalRouteUnavailableController(chapter, reason)
        return ((chapter, controller, controller.max_frames),)

    return stages


def level9_entry_chapter(
    route: Level9NaturalRouteSelection = UNSELECTED_NATURAL_ROUTE,
) -> tuple[tuple[str, Any, int], ...]:
    reason = (
        "natural_topology_not_decoded"
        if not route.topology_decoded
        else "natural_entry_controller_not_implemented"
    )
    return _unavailable_stage("level9_natural_entry", reason)()


def level9_silver_arrows_chapter(
    route: Level9NaturalRouteSelection = UNSELECTED_NATURAL_ROUTE,
) -> tuple[tuple[str, Any, int], ...]:
    if not route.topology_decoded:
        reason = "natural_topology_not_decoded"
    elif route.silver_arrow_room is None:
        reason = "silver_arrow_room_not_selected"
    else:
        reason = "natural_silver_arrow_controller_not_implemented"
    return _unavailable_stage("level9_natural_silver_arrows", reason)()


def level9_patra_chapter(
    route: Level9NaturalRouteSelection = UNSELECTED_NATURAL_ROUTE,
) -> tuple[tuple[str, Any, int], ...]:
    reason = (
        "natural_suffix_join_not_selected"
        if route.suffix_join_room is None
        else "natural_join_controller_not_implemented"
    )
    return _unavailable_stage("level9_natural_patra_join", reason)()


def level9_credits_chapter() -> tuple[tuple[str, Any, int], ...]:
    """Fresh write-free controllers from exact live Patra to credits."""
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
    route: Level9NaturalRouteSelection = UNSELECTED_NATURAL_ROUTE,
) -> tuple[SpineHop, ...]:
    """Return the four public L9 chapter rows.

    ``route`` is descriptive until decoded room rows and controllers replace
    the unavailable factories.  In particular, ``requires_51_to_41=None``
    keeps rr-yxy6 conditional instead of selecting that recon leaf.
    """

    def entry_ok(snap, **_):
        magic_key = bool(
            env is not None and read_u8(env.get_ram(), ADDR_MAGIC_KEY) > 0
        )
        return level9_entry_stop(snap, magic_key=magic_key)

    return (
        SpineHop(
            L9_ENTRY_ENDPOINT.through,
            L9_ENTRY_ENDPOINT.stop,
            lambda: level9_entry_chapter(route),
            entry_ok,
        ),
        SpineHop(
            L9_SILVER_ARROWS_ENDPOINT.through,
            L9_SILVER_ARROWS_ENDPOINT.stop,
            lambda: level9_silver_arrows_chapter(route),
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
            lambda snap, **_: level9_credits_stop(snap),
        ),
    )


__all__ = [
    "Level9NaturalRouteSelection",
    "UNSELECTED_NATURAL_ROUTE",
    "level9_credits_chapter",
    "level9_entry_chapter",
    "level9_patra_chapter",
    "level9_silver_arrows_chapter",
    "l9_hops",
]
