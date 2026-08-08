"""Opening-route continuous trunk for A Link to the Past.

Owns the escape graph, opening catalog, work queue, multi-truth anchors, and
live segment scripts (castle→sword, secret-entrance clear, pocket→main hall,
castle_dungeon_prefix; planned escort→Sanctuary scaffold). Core
RAM/primitives/startup stay at ``alttp`` root.

Preferred imports::

    from alttp.opening_route import escape_route_graph, plan_escape_to_sanctuary
    from alttp.opening_route.castle_to_sword import run_natural_chain
    from alttp.opening_route.secret_entrance_clear import run_from_sword
    from alttp.opening_route.segment import get_segment, list_segments
"""

from __future__ import annotations

from alttp.opening_route.escape_graph import (
    CAP_FIGHTER_SWORD,
    CAP_LAMP,
    CAP_SMALL_KEY,
    CAP_ZELDA_FOLLOWER,
    N_CASTLE_GROUNDS,
    N_COURTYARD_SECRET_POCKET,
    N_ROOM_55_SOUTH,
    N_ROOM_55_SWORD,
    N_ROOM_55_UNCLE,
    N_ROOM_60,
    N_ROOM_61,
    N_SANCTUARY,
    capabilities_from_snapshot,
    continuous_spine_legs,
    escape_route_graph,
    escape_route_legs,
    plan_escape_to_sanctuary,
)

__all__ = [
    "CAP_FIGHTER_SWORD",
    "CAP_LAMP",
    "CAP_SMALL_KEY",
    "CAP_ZELDA_FOLLOWER",
    "N_CASTLE_GROUNDS",
    "N_COURTYARD_SECRET_POCKET",
    "N_ROOM_55_SOUTH",
    "N_ROOM_55_SWORD",
    "N_ROOM_55_UNCLE",
    "N_ROOM_60",
    "N_ROOM_61",
    "N_SANCTUARY",
    "capabilities_from_snapshot",
    "continuous_spine_legs",
    "escape_route_graph",
    "escape_route_legs",
    "plan_escape_to_sanctuary",
]
