"""KPDR K1–K2 Brinstar: Super exit → Red Tower → Bat → Below Spazer → Warehouse.

Continuous door edges for Super+ tips are generated from the KPDR spine
(``continuous_edges_for_tips``), not hand-duplicated here.
"""

from __future__ import annotations

from super_metroid.progression.graph import RoomProgressionGraph
from super_metroid.progression.stages.spore import (
    EDGES as _SPORE_EDGES,
    MILESTONES as _SPORE_MILESTONES,
    ROOMS as _SPORE_ROOMS,
)
from super_metroid.progression.types import ProgressCondition, ProgressionMilestone, RoomNode
from super_metroid.ram import BOMBS_MASK, MORPH_BALL_MASK
from super_metroid.routes.kpdr.room_ids import (
    ROOM_BAT,
    ROOM_BELOW_SPAZER,
    ROOM_EAST_TUNNEL,
    ROOM_FARMING,
    ROOM_GHZ,
    ROOM_GLASS,
    ROOM_NOOB,
    ROOM_RED_TOWER,
    ROOM_SUPER,
    ROOM_WAREHOUSE,
    ROOM_WEST_TUNNEL,
)
from super_metroid.routes.kpdr.spine import continuous_edges_for_tips

# KPDR K1 suffix: Super exit → farming → Big Pink main → GHZ → Noob → Red Tower.
# (Charge Beam return is a side trip and is not on this continuous chain.)
# Continuous door edges: spine tip ``red_tower`` (big_pink_main is in-room only).
ROOMS_RED_TOWER = _SPORE_ROOMS + (
    RoomNode(ROOM_FARMING, "Pink Brinstar Farming Room", "Brinstar"),
    RoomNode(ROOM_GHZ, "Green Hill Zone", "Brinstar"),
    RoomNode(ROOM_NOOB, "Noob Bridge", "Brinstar"),
    RoomNode(ROOM_RED_TOWER, "Red Tower", "Brinstar", frozenset({"vertical_shaft"})),
)

EDGES_RED_TOWER = _SPORE_EDGES + continuous_edges_for_tips("red_tower")

MILESTONES_RED_TOWER = _SPORE_MILESTONES + (
    ProgressionMilestone(
        "spore_supers_collected",
        "Spore Super Missiles capacity 0→5",
        ProgressCondition(
            room_id=ROOM_SUPER,
            collected_items_mask=MORPH_BALL_MASK | BOMBS_MASK,
            minimum_ammo_capacities=(10, 5, 0),
        ),
        requires=frozenset({"morph_ball", "bombs", "missiles"}),
        acquires=frozenset({"super_missiles"}),
        timeout_frames=8_000,
        policy_id="kpdr_super_room",
    ),
    ProgressionMilestone(
        "red_tower_entry",
        "Natural Red Tower entry via Big Pink → GHZ → Noob",
        ProgressCondition(
            room_id=ROOM_RED_TOWER,
            collected_items_mask=MORPH_BALL_MASK | BOMBS_MASK,
            minimum_ammo_capacities=(10, 5, 0),
        ),
        requires=frozenset({"morph_ball", "bombs", "missiles", "super_missiles"}),
        timeout_frames=30_000,
        policy_id="kpdr_k1",
    ),
)

RED_TOWER_GRAPH = RoomProgressionGraph(
    ROOMS_RED_TOWER,
    EDGES_RED_TOWER,
    MILESTONES_RED_TOWER,
    graph_id="red_tower",
)

# KPDR K2.0: Red Tower descent → Bat Room (first continuous hop after K1 tip).
ROOMS_BAT = ROOMS_RED_TOWER + (RoomNode(ROOM_BAT, "Bat Room", "Brinstar"),)

EDGES_BAT = EDGES_RED_TOWER + continuous_edges_for_tips("bat")

MILESTONES_BAT = MILESTONES_RED_TOWER + (
    ProgressionMilestone(
        "bat_room_entry",
        "Natural Bat Room entry via Red Tower descent",
        ProgressCondition(
            room_id=ROOM_BAT,
            collected_items_mask=MORPH_BALL_MASK | BOMBS_MASK,
            minimum_ammo_capacities=(10, 5, 0),
        ),
        requires=frozenset({"morph_ball", "bombs", "missiles", "super_missiles"}),
        timeout_frames=8_000,
        policy_id="kpdr_red_tower",
    ),
)

BAT_GRAPH = RoomProgressionGraph(
    ROOMS_BAT,
    EDGES_BAT,
    MILESTONES_BAT,
    graph_id="bat",
)

# KPDR K2.1: Bat Room three-platform crossing → Below Spazer.
ROOMS_BELOW_SPAZER = ROOMS_BAT + (RoomNode(ROOM_BELOW_SPAZER, "Below Spazer", "Brinstar"),)

EDGES_BELOW_SPAZER = EDGES_BAT + continuous_edges_for_tips("below_spazer")

MILESTONES_BELOW_SPAZER = MILESTONES_BAT + (
    ProgressionMilestone(
        "below_spazer_entry",
        "Natural Below Spazer entry via Bat Room platforms",
        ProgressCondition(
            room_id=ROOM_BELOW_SPAZER,
            collected_items_mask=MORPH_BALL_MASK | BOMBS_MASK,
            minimum_ammo_capacities=(10, 5, 0),
        ),
        requires=frozenset({"morph_ball", "bombs", "missiles", "super_missiles"}),
        timeout_frames=4_000,
        policy_id="kpdr_red_tower",
    ),
)

BELOW_SPAZER_GRAPH = RoomProgressionGraph(
    ROOMS_BELOW_SPAZER,
    EDGES_BELOW_SPAZER,
    MILESTONES_BELOW_SPAZER,
    graph_id="below_spazer",
)

# KPDR K2.3–K2.6: Below Spazer → West → Glass → East → Warehouse Entrance.
ROOMS_WAREHOUSE = ROOMS_BELOW_SPAZER + (
    RoomNode(ROOM_WEST_TUNNEL, "West Tunnel", "Maridia"),
    RoomNode(ROOM_GLASS, "Glass Tunnel", "Maridia"),
    RoomNode(ROOM_EAST_TUNNEL, "East Tunnel", "Maridia"),
    RoomNode(ROOM_WAREHOUSE, "Warehouse Entrance", "Brinstar"),
)

EDGES_WAREHOUSE = EDGES_BELOW_SPAZER + continuous_edges_for_tips("warehouse")

MILESTONES_WAREHOUSE = MILESTONES_BELOW_SPAZER + (
    ProgressionMilestone(
        "warehouse_entry",
        "Natural Warehouse Entrance via Below Spazer tunnels",
        ProgressCondition(
            room_id=ROOM_WAREHOUSE,
            collected_items_mask=MORPH_BALL_MASK | BOMBS_MASK,
            minimum_ammo_capacities=(10, 5, 0),
        ),
        requires=frozenset({"morph_ball", "bombs", "missiles", "super_missiles"}),
        timeout_frames=12_000,
        policy_id="kpdr_red_tower",
    ),
)

WAREHOUSE_GRAPH = RoomProgressionGraph(
    ROOMS_WAREHOUSE,
    EDGES_WAREHOUSE,
    MILESTONES_WAREHOUSE,
    graph_id="warehouse",
)

# Tip of this module for the next stage (kraid path).
ROOMS = ROOMS_WAREHOUSE
EDGES = EDGES_WAREHOUSE
MILESTONES = MILESTONES_WAREHOUSE

__all__ = [
    "ROOMS",
    "EDGES",
    "MILESTONES",
    "ROOMS_RED_TOWER",
    "EDGES_RED_TOWER",
    "MILESTONES_RED_TOWER",
    "ROOMS_BAT",
    "EDGES_BAT",
    "MILESTONES_BAT",
    "ROOMS_BELOW_SPAZER",
    "EDGES_BELOW_SPAZER",
    "MILESTONES_BELOW_SPAZER",
    "ROOMS_WAREHOUSE",
    "EDGES_WAREHOUSE",
    "MILESTONES_WAREHOUSE",
    "RED_TOWER_GRAPH",
    "BAT_GRAPH",
    "BELOW_SPAZER_GRAPH",
    "WAREHOUSE_GRAPH",
]
