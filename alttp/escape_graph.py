"""Compat shim — prefer ``alttp.opening_route.escape_graph``."""

from __future__ import annotations

from alttp.opening_route.escape_graph import *  # noqa: F403
from alttp.opening_route.escape_graph import (  # noqa: F401
    CAP_FIGHTER_SWORD,
    CAP_LAMP,
    CAP_SMALL_KEY,
    CAP_ZELDA_FOLLOWER,
    N_CASTLE_GROUNDS,
    N_CASTLE_MANTLE,
    N_COURTYARD_SECRET_POCKET,
    N_ROOM_12,
    N_ROOM_01,
    N_ROOM_50,
    N_ROOM_55_KEYED,
    N_ROOM_55_SOUTH,
    N_ROOM_55_SWORD,
    N_ROOM_55_UNCLE,
    N_ROOM_60,
    N_ROOM_61,
    N_ROOM_80,
    N_SANCTUARY,
    N_SEWERS_DARK,
    NATURAL_HOUSE_EXIT_CAPABILITIES,
    VERIFICATION_CONTINUOUS,
    VERIFICATION_ISOLATED,
    VERIFICATION_NATURAL_ENTRY,
    VERIFICATION_PLANNED,
    capabilities_from_snapshot,
    continuous_spine_legs,
    escape_route_graph,
    escape_route_legs,
    escape_route_legs_from_room_55,
    escape_route_legs_key_path,
    plan_escape_to_sanctuary,
)
