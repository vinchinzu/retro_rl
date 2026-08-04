"""KPDR K2.7–K3: Warehouse → Hi-Jump → Kraid approach → Varia.

Continuous Super+ door edges from spine tips ``hijump`` / ``kraid`` / ``varia``.
"""

from __future__ import annotations

from super_metroid.progression.graph import RoomProgressionGraph
from super_metroid.progression.stages.brinstar import (
    EDGES as _WAREHOUSE_EDGES,
    MILESTONES as _WAREHOUSE_MILESTONES,
    ROOMS as _WAREHOUSE_ROOMS,
)
from super_metroid.progression.types import ProgressCondition, ProgressionMilestone, RoomNode
from super_metroid.ram import BOMBS_MASK, HI_JUMP_MASK, MORPH_BALL_MASK, VARIA_MASK
from super_metroid.routes.kpdr.room_ids import (
    ROOM_BABY_KRAID,
    ROOM_BUSINESS,
    ROOM_HJ,
    ROOM_HJ_SHAFT,
    ROOM_KRAID,
    ROOM_KRAID_EYE,
    ROOM_VARIA,
    ROOM_WAREHOUSE_KIHUNTER,
    ROOM_ZEELA,
)
from super_metroid.routes.kpdr.spine import continuous_edges_for_tips

_BASE_CAPS = frozenset({"morph_ball", "bombs", "missiles", "super_missiles"})
_HJ_CAPS = _BASE_CAPS | frozenset({"hi_jump"})
_VARIA_CAPS = _HJ_CAPS | frozenset({"varia_suit"})

# KPDR K2.7–K2.10: Warehouse → Business → Hi-Jump shaft → Hi-Jump collect.
ROOMS_HIJUMP = _WAREHOUSE_ROOMS + (
    RoomNode(ROOM_BUSINESS, "Business Center", "Norfair"),
    RoomNode(ROOM_HJ_SHAFT, "Hi-Jump Boots E-Tank Room", "Norfair"),
    RoomNode(ROOM_HJ, "Hi-Jump Room", "Norfair"),
)

EDGES_HIJUMP = _WAREHOUSE_EDGES + continuous_edges_for_tips("hijump")

MILESTONES_HIJUMP = _WAREHOUSE_MILESTONES + (
    ProgressionMilestone(
        "hijump_collected",
        "Natural Hi-Jump Boots collect from Warehouse continuous tip",
        ProgressCondition(
            room_id=ROOM_HJ,
            collected_items_mask=MORPH_BALL_MASK | BOMBS_MASK | HI_JUMP_MASK,
            minimum_ammo_capacities=(10, 5, 0),
        ),
        requires=_BASE_CAPS,
        acquires=frozenset({"hi_jump"}),
        timeout_frames=20_000,
        policy_id="kpdr_hijump",
    ),
)

HIJUMP_GRAPH = RoomProgressionGraph(
    ROOMS_HIJUMP,
    EDGES_HIJUMP,
    MILESTONES_HIJUMP,
    graph_id="hijump",
)

# KPDR K2.11–K2.18: Hi-Jump return → Warehouse → Zeela → … → natural Kraid entry.
ROOMS_KRAID = ROOMS_HIJUMP + (
    RoomNode(ROOM_ZEELA, "Warehouse Zeela Room", "Brinstar"),
    RoomNode(ROOM_WAREHOUSE_KIHUNTER, "Warehouse Kihunter Room", "Brinstar"),
    RoomNode(ROOM_BABY_KRAID, "Baby Kraid Room", "Brinstar"),
    RoomNode(ROOM_KRAID_EYE, "Kraid Eye Door Room", "Brinstar"),
    RoomNode(ROOM_KRAID, "Kraid's Room", "Brinstar"),
)

EDGES_KRAID = EDGES_HIJUMP + continuous_edges_for_tips("kraid")

MILESTONES_KRAID = MILESTONES_HIJUMP + (
    ProgressionMilestone(
        "kraid_entry",
        "Natural Kraid room entry after Hi-Jump return via Warehouse approach",
        ProgressCondition(
            room_id=ROOM_KRAID,
            collected_items_mask=MORPH_BALL_MASK | BOMBS_MASK | HI_JUMP_MASK,
            minimum_ammo_capacities=(10, 5, 0),
        ),
        requires=_HJ_CAPS,
        timeout_frames=40_000,
        policy_id="kpdr_kraid_approach",
    ),
)

KRAID_GRAPH = RoomProgressionGraph(
    ROOMS_KRAID,
    EDGES_KRAID,
    MILESTONES_KRAID,
    graph_id="kraid",
)

# KPDR K3: Kraid fight → rear exit → natural Varia collect.
ROOMS_VARIA = ROOMS_KRAID + (RoomNode(ROOM_VARIA, "Varia Suit Room", "Brinstar"),)

EDGES_VARIA = EDGES_KRAID + continuous_edges_for_tips("varia")

MILESTONES_VARIA = MILESTONES_KRAID + (
    ProgressionMilestone(
        "varia_collected",
        "Natural Varia collect after Kraid fight from continuous chain",
        ProgressCondition(
            room_id=ROOM_VARIA,
            collected_items_mask=(
                MORPH_BALL_MASK | BOMBS_MASK | HI_JUMP_MASK | VARIA_MASK
            ),
            minimum_ammo_capacities=(10, 5, 0),
        ),
        requires=_HJ_CAPS,
        acquires=frozenset({"varia_suit"}),
        timeout_frames=12_000,
        policy_id="kpdr_kraid_combat",
    ),
)

VARIA_GRAPH = RoomProgressionGraph(
    ROOMS_VARIA,
    EDGES_VARIA,
    MILESTONES_VARIA,
    graph_id="varia",
)

# Tip of this module for the K4 speed stage.
ROOMS = ROOMS_VARIA
EDGES = EDGES_VARIA
MILESTONES = MILESTONES_VARIA
BASE_CAPS = _BASE_CAPS
HJ_CAPS = _HJ_CAPS
VARIA_CAPS = _VARIA_CAPS

__all__ = [
    "ROOMS",
    "EDGES",
    "MILESTONES",
    "BASE_CAPS",
    "HJ_CAPS",
    "VARIA_CAPS",
    "ROOMS_HIJUMP",
    "EDGES_HIJUMP",
    "MILESTONES_HIJUMP",
    "ROOMS_KRAID",
    "EDGES_KRAID",
    "MILESTONES_KRAID",
    "ROOMS_VARIA",
    "EDGES_VARIA",
    "MILESTONES_VARIA",
    "HIJUMP_GRAPH",
    "KRAID_GRAPH",
    "VARIA_GRAPH",
]
