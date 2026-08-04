"""Staged room/edge/milestone tables that compose into public *GRAPH constants.

Each module owns one or more graph stages and exports intermediate ``ROOMS`` /
``EDGES`` / ``MILESTONES`` tuples for the next stage to extend. Public graphs
are re-exported here and via :mod:`super_metroid.progression.data`.
"""

from __future__ import annotations

from super_metroid.progression.stages.brinstar import (
    BAT_GRAPH,
    BELOW_SPAZER_GRAPH,
    RED_TOWER_GRAPH,
    WAREHOUSE_GRAPH,
)
from super_metroid.progression.stages.early import EARLY_GAME_GRAPH
from super_metroid.progression.stages.kraid import HIJUMP_GRAPH, KRAID_GRAPH, VARIA_GRAPH
from super_metroid.progression.stages.morph import MORPH_GRAPH
from super_metroid.progression.stages.speed import SPEED_GRAPH
from super_metroid.progression.stages.spore import SPORE_GRAPH

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
]
