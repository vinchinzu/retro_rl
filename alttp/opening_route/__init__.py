"""Opening-route continuous trunk for A Link to the Past.

Owns the escape graph, opening catalog, work queue, and live segment
scripts (castle→sword, sword→pocket, pocket→main hall). Core
RAM/primitives/startup stay at ``alttp`` root.

Preferred imports::

    from alttp.opening_route import escape_route_graph, plan_escape_to_sanctuary
    from alttp.opening_route.castle_to_sword import run_natural_chain
    from alttp.opening_route.segment import SEGMENT_REGISTRY

Compat shims at ``alttp.escape_graph``, ``alttp.castle_to_sword``, etc.
re-export this package for older call sites.
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
    "N_ROOM_61",
    "N_SANCTUARY",
    "capabilities_from_snapshot",
    "continuous_spine_legs",
    "escape_route_graph",
    "escape_route_legs",
    "plan_escape_to_sanctuary",
]
