"""Room progression graph package.

Public surface matches the former monolithic ``progression`` module so
``from super_metroid.progression import ...`` stays stable.
"""

from __future__ import annotations

# types/graph before data: data may import Super+ DoorEdges from the KPDR spine,
# and spine→runtime may re-enter this package while data is still loading.
from super_metroid.progression.types import (
    VERIFICATION_RANK,
    DoorEdge,
    ObservedTransition,
    ProgressCondition,
    ProgressionMilestone,
    RoomNode,
    capabilities_from_state,
)
from super_metroid.progression.graph import RoomProgressionGraph
from super_metroid.progression.data import (
    BAT_GRAPH,
    BELOW_SPAZER_GRAPH,
    EARLY_GAME_GRAPH,
    HIJUMP_GRAPH,
    KRAID_GRAPH,
    MORPH_GRAPH,
    RED_TOWER_GRAPH,
    SPEED_GRAPH,
    SPORE_GRAPH,
    VARIA_GRAPH,
    WAREHOUSE_GRAPH,
)

__all__ = [
    "BAT_GRAPH",
    "BELOW_SPAZER_GRAPH",
    "EARLY_GAME_GRAPH",
    "HIJUMP_GRAPH",
    "KRAID_GRAPH",
    "MORPH_GRAPH",
    "RED_TOWER_GRAPH",
    "SPEED_GRAPH",
    "SPORE_GRAPH",
    "VARIA_GRAPH",
    "WAREHOUSE_GRAPH",
    "VERIFICATION_RANK",
    "DoorEdge",
    "ObservedTransition",
    "ProgressCondition",
    "ProgressionMilestone",
    "RoomNode",
    "RoomProgressionGraph",
    "capabilities_from_state",
]
