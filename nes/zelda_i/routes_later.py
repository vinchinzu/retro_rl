"""NamedRoute **stubs** for Zelda I Level 3–9 (planning only).

These routes are graph / catalog scaffolding. Stop-predicate strings are future
controller names — **not** Clean-backed. Edges are not route-ready until live
evidence lands in ``docs/LEVELN_ROUTE.md`` and STATUS pure-first gates clear.

See ``docs/OVERWORLD_DOORS.md`` and ``docs/research/DUNGEON_WALKTHROUGHS.md``.
L1/L2 published routes remain in ``zelda_i.routes`` (do not rewrite them here).
"""

from __future__ import annotations

from retro_harness.adventure.routes import (
    NamedRoute,
    RouteMilestone,
    get_route as _get_route,
    list_routes as _list_routes,
    register_routes,
)
from zelda_i.later_nodes import (
    NODE_LEVEL3_COMPLETE,
    NODE_LEVEL3_DUNGEON,
    NODE_LEVEL3_ENTRANCE,
    NODE_LEVEL4_COMPLETE,
    NODE_LEVEL4_DUNGEON,
    NODE_LEVEL4_ENTRANCE,
    NODE_LEVEL5_COMPLETE,
    NODE_LEVEL5_DUNGEON,
    NODE_LEVEL5_ENTRANCE,
    NODE_LEVEL6_COMPLETE,
    NODE_LEVEL6_DUNGEON,
    NODE_LEVEL6_ENTRANCE,
    NODE_LEVEL7_COMPLETE,
    NODE_LEVEL7_DUNGEON,
    NODE_LEVEL7_ENTRANCE,
    NODE_LEVEL8_COMPLETE,
    NODE_LEVEL8_DUNGEON,
    NODE_LEVEL8_ENTRANCE,
    NODE_LEVEL9_DUNGEON,
    NODE_LEVEL9_ENTRANCE,
    NODE_LEVEL9_GANON,
    NODE_LEVEL9_ZELDA,
    TF_BIT_L3,
    TF_BIT_L4,
    TF_BIT_L5,
    TF_BIT_L6,
    TF_BIT_L7,
    TF_BIT_L8,
    TF_BITS_ALL,
    TRIFORCE_BITS_BY_LEVEL,
)

# Re-export bit map for tests / tooling
__all__ = [
    "ROUTE_LEVEL3_COMPLETE",
    "ROUTE_LEVEL4_COMPLETE",
    "ROUTE_LEVEL5_COMPLETE",
    "ROUTE_LEVEL6_COMPLETE",
    "ROUTE_LEVEL7_COMPLETE",
    "ROUTE_LEVEL8_COMPLETE",
    "ROUTE_LEVEL9_GANON",
    "ROUTE_REGISTRY_LATER",
    "TRIFORCE_BITS_BY_LEVEL",
    "TF_BITS_ALL",
    "get_later_route",
    "list_later_routes",
]


_STUB_NOTE = (
    "PLANNING STUB — not route-ready. Door screens and stop predicates are "
    "placeholders until live recon (OVERWORLD_DOORS.md). No Clean claim."
)


def _door_milestones(
    *,
    level: int,
    name: str,
    entrance_node: str,
    dungeon_node: str,
    door_pred: str,
    enter_pred: str,
) -> tuple[RouteMilestone, ...]:
    return (
        RouteMilestone(
            f"level{level}_entrance",
            entrance_node,
            f"Level {level} ({name}) overworld door (source candidate)",
            door_pred,
        ),
        RouteMilestone(
            f"level{level}_dungeon",
            dungeon_node,
            f"Inside Level {level} ({name})",
            enter_pred,
        ),
    )


def _complete_milestones(
    *,
    level: int,
    name: str,
    entrance_node: str,
    dungeon_node: str,
    complete_node: str,
    tf_bit: int,
    door_pred: str,
    enter_pred: str,
    item_label: str,
) -> tuple[RouteMilestone, ...]:
    return (
        *_door_milestones(
            level=level,
            name=name,
            entrance_node=entrance_node,
            dungeon_node=dungeon_node,
            door_pred=door_pred,
            enter_pred=enter_pred,
        ),
        RouteMilestone(
            f"level{level}_item",
            dungeon_node,
            f"Level {level} item: {item_label} (planning)",
            f"level{level}_item_collected",
        ),
        RouteMilestone(
            f"level{level}_complete",
            complete_node,
            f"Level {level} Triforce shard {level}",
            f"triforce & 0x{tf_bit:02x}",
        ),
    )


ROUTE_LEVEL3_COMPLETE = NamedRoute(
    route_id="zelda_level3_complete",
    display_name="Level 3 Manji (stub) → Triforce 0x04",
    description=(
        f"{_STUB_NOTE} Source door 0x74. Item Raft. Boss Manhandla. TF bit 0x04."
    ),
    milestones=_complete_milestones(
        level=3,
        name="Manji",
        entrance_node=NODE_LEVEL3_ENTRANCE,
        dungeon_node=NODE_LEVEL3_DUNGEON,
        complete_node=NODE_LEVEL3_COMPLETE,
        tf_bit=TF_BIT_L3,
        door_pred="level3_door_screen",
        enter_pred="level3_entrance_success",
        item_label="Raft",
    ),
)

ROUTE_LEVEL4_COMPLETE = NamedRoute(
    route_id="zelda_level4_complete",
    display_name="Level 4 Snake (stub) → Triforce 0x08",
    description=(
        f"{_STUB_NOTE} Source island/door 0x45 via raft dock 0x55 (level4_overworld "
        "hyp). Item Stepladder. Boss Gleeok 2-head. TF bit 0x08. Requires Raft from L3."
    ),
    milestones=_complete_milestones(
        level=4,
        name="Snake",
        entrance_node=NODE_LEVEL4_ENTRANCE,
        dungeon_node=NODE_LEVEL4_DUNGEON,
        complete_node=NODE_LEVEL4_COMPLETE,
        tf_bit=TF_BIT_L4,
        door_pred="level4_door_screen",
        enter_pred="level4_entrance_success",
        item_label="Stepladder",
    ),
)

ROUTE_LEVEL5_COMPLETE = NamedRoute(
    route_id="zelda_level5_complete",
    display_name="Level 5 Lizard (stub) → Triforce 0x10",
    description=(
        f"{_STUB_NOTE} Source door 0x0B via Lost Hills 0x1B ↑×4. Item Whistle. "
        "Boss Digdogger. TF bit 0x10."
    ),
    milestones=_complete_milestones(
        level=5,
        name="Lizard",
        entrance_node=NODE_LEVEL5_ENTRANCE,
        dungeon_node=NODE_LEVEL5_DUNGEON,
        complete_node=NODE_LEVEL5_COMPLETE,
        tf_bit=TF_BIT_L5,
        door_pred="level5_door_screen",
        enter_pred="level5_entrance_success",
        item_label="Whistle",
    ),
)

ROUTE_LEVEL6_COMPLETE = NamedRoute(
    route_id="zelda_level6_complete",
    display_name="Level 6 Dragon (stub) → Triforce 0x20",
    description=(
        f"{_STUB_NOTE} Source door 0x22 near graveyard. Item Magical Rod. "
        "Boss Gohma (arrow eye). TF bit 0x20."
    ),
    milestones=_complete_milestones(
        level=6,
        name="Dragon",
        entrance_node=NODE_LEVEL6_ENTRANCE,
        dungeon_node=NODE_LEVEL6_DUNGEON,
        complete_node=NODE_LEVEL6_COMPLETE,
        tf_bit=TF_BIT_L6,
        door_pred="level6_door_screen",
        enter_pred="level6_entrance_success",
        item_label="Magical Rod",
    ),
)

ROUTE_LEVEL7_COMPLETE = NamedRoute(
    route_id="zelda_level7_complete",
    display_name="Level 7 Demon (stub) → Triforce 0x40",
    description=(
        f"{_STUB_NOTE} Source door 0x42 whistle pond. Item Red Candle. "
        "Boss Aquamentus. TF bit 0x40. Requires Whistle + Bait."
    ),
    milestones=_complete_milestones(
        level=7,
        name="Demon",
        entrance_node=NODE_LEVEL7_ENTRANCE,
        dungeon_node=NODE_LEVEL7_DUNGEON,
        complete_node=NODE_LEVEL7_COMPLETE,
        tf_bit=TF_BIT_L7,
        door_pred="level7_door_screen",
        enter_pred="level7_entrance_success",
        item_label="Red Candle",
    ),
)

ROUTE_LEVEL8_COMPLETE = NamedRoute(
    route_id="zelda_level8_complete",
    display_name="Level 8 Lion (stub) → Triforce 0x80",
    description=(
        f"{_STUB_NOTE} Source door 0x6D candle bush. Items Book of Magic + "
        "Magical Key. Boss Gleeok 4-head. TF bit 0x80."
    ),
    milestones=_complete_milestones(
        level=8,
        name="Lion",
        entrance_node=NODE_LEVEL8_ENTRANCE,
        dungeon_node=NODE_LEVEL8_DUNGEON,
        complete_node=NODE_LEVEL8_COMPLETE,
        tf_bit=TF_BIT_L8,
        door_pred="level8_door_screen",
        enter_pred="level8_entrance_success",
        item_label="Book of Magic / Magical Key",
    ),
)

ROUTE_LEVEL9_GANON = NamedRoute(
    route_id="zelda_level9_ganon",
    display_name="Level 9 Death Mountain (stub) → Ganon",
    description=(
        f"{_STUB_NOTE} Source door 0x05 bomb rock. Items Red Ring + Silver Arrows. "
        "Boss Ganon (stun + Silver Arrow). Requires full Triforce 0xFF at Old Man gate."
    ),
    milestones=(
        *_door_milestones(
            level=9,
            name="Death Mountain",
            entrance_node=NODE_LEVEL9_ENTRANCE,
            dungeon_node=NODE_LEVEL9_DUNGEON,
            door_pred="level9_door_screen",
            enter_pred="level9_entrance_success",
        ),
        RouteMilestone(
            "level9_full_triforce_gate",
            NODE_LEVEL9_DUNGEON,
            "Full Triforce Old Man gate (planning)",
            f"triforce == 0x{TF_BITS_ALL:02x}",
        ),
        RouteMilestone(
            "level9_red_ring",
            NODE_LEVEL9_DUNGEON,
            "Red Ring collected (planning)",
            "level9_red_ring_collected",
        ),
        RouteMilestone(
            "level9_silver_arrows",
            NODE_LEVEL9_DUNGEON,
            "Silver Arrows collected (planning)",
            "level9_silver_arrows_collected",
        ),
        RouteMilestone(
            "level9_ganon",
            NODE_LEVEL9_GANON,
            "Ganon defeated (planning)",
            "level9_ganon_defeated",
        ),
        RouteMilestone(
            "level9_zelda",
            NODE_LEVEL9_ZELDA,
            "Princess Zelda / ending (planning)",
            "level9_ending",
        ),
    ),
)


ROUTE_REGISTRY_LATER: dict[str, NamedRoute] = {}
register_routes(
    ROUTE_REGISTRY_LATER,
    ROUTE_LEVEL3_COMPLETE,
    "level3",
    "level3_complete",
    "triforce_3",
)
register_routes(
    ROUTE_REGISTRY_LATER,
    ROUTE_LEVEL4_COMPLETE,
    "level4",
    "level4_complete",
    "triforce_4",
)
register_routes(
    ROUTE_REGISTRY_LATER,
    ROUTE_LEVEL5_COMPLETE,
    "level5",
    "level5_complete",
    "triforce_5",
)
register_routes(
    ROUTE_REGISTRY_LATER,
    ROUTE_LEVEL6_COMPLETE,
    "level6",
    "level6_complete",
    "triforce_6",
)
register_routes(
    ROUTE_REGISTRY_LATER,
    ROUTE_LEVEL7_COMPLETE,
    "level7",
    "level7_complete",
    "triforce_7",
)
register_routes(
    ROUTE_REGISTRY_LATER,
    ROUTE_LEVEL8_COMPLETE,
    "level8",
    "level8_complete",
    "triforce_8",
)
register_routes(
    ROUTE_REGISTRY_LATER,
    ROUTE_LEVEL9_GANON,
    "level9",
    "level9_ganon",
    "ganon",
)


def get_later_route(route_id: str) -> NamedRoute:
    return _get_route(ROUTE_REGISTRY_LATER, route_id)


def list_later_routes() -> list[NamedRoute]:
    return _list_routes(ROUTE_REGISTRY_LATER)
