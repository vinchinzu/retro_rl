"""Morph graph stage: Ceres → Morph Ball (play spine edges/milestones)."""

from __future__ import annotations

from super_metroid.progression.graph import RoomProgressionGraph
from super_metroid.progression.types import RoomNode
from super_metroid.routes.kpdr.early_spine import MORPH_DOOR_EDGES, MORPH_MILESTONES
from super_metroid.routes.kpdr.room_ids import (
    ROOM_BLUE_BRINSTAR_ELEVATOR,
    ROOM_CERES_ELEVATOR,
    ROOM_CERES_FALLING,
    ROOM_CERES_FLAT,
    ROOM_CERES_MAGNET,
    ROOM_CERES_RIDLEY,
    ROOM_CERES_SCIENTIST,
    ROOM_CLIMB,
    ROOM_CONSTRUCTION,
    ROOM_LANDING_SITE,
    ROOM_MORPH,
    ROOM_PARLOR,
    ROOM_PIT,
)

ROOMS = (
    RoomNode(ROOM_CERES_ELEVATOR, "Ceres Elevator Room", "Ceres"),
    RoomNode(ROOM_CERES_FALLING, "Ceres Falling Tile Room", "Ceres"),
    RoomNode(ROOM_CERES_MAGNET, "Ceres Magnet Stairs Room", "Ceres"),
    RoomNode(ROOM_CERES_SCIENTIST, "Ceres Dead Scientist Room", "Ceres"),
    RoomNode(ROOM_CERES_FLAT, "Ceres Flat Room", "Ceres"),
    RoomNode(ROOM_CERES_RIDLEY, "Ceres Ridley Room", "Ceres", frozenset({"scripted_boss"})),
    RoomNode(ROOM_LANDING_SITE, "Landing Site", "Crateria", frozenset({"ship"})),
    RoomNode(ROOM_PARLOR, "Parlor and Alcatraz", "Crateria"),
    RoomNode(ROOM_CLIMB, "Climb", "Crateria", frozenset({"vertical_shaft"})),
    RoomNode(ROOM_PIT, "Pit Room", "Crateria"),
    RoomNode(ROOM_BLUE_BRINSTAR_ELEVATOR, "Blue Brinstar Elevator Room", "Crateria"),
    RoomNode(ROOM_MORPH, "Morph Ball Room", "Brinstar", frozenset({"item_room"})),
    RoomNode(ROOM_CONSTRUCTION, "Construction Zone", "Brinstar"),
)

# Morph edges/milestones: colocated with play spine in early_spine.
EDGES = MORPH_DOOR_EDGES
MILESTONES = MORPH_MILESTONES

MORPH_GRAPH = RoomProgressionGraph(
    ROOMS,
    EDGES,
    MILESTONES,
    graph_id="morph",
)

__all__ = ["ROOMS", "EDGES", "MILESTONES", "MORPH_GRAPH"]
