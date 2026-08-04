"""Staged progression graphs — public *GRAPH export surface.

Room/edge/milestone tables live in :mod:`super_metroid.progression.stages`
(one module per stage group). This module re-exports the composed graphs so
``from super_metroid.progression.data import MORPH_GRAPH`` (and package
``__init__``) stay stable.

Stage ownership:

- **Morph** rooms + edges/milestones from
  :mod:`super_metroid.routes.kpdr.early_spine` → ``stages.morph``
- **Bombs → Spore** hand-authored edges → ``stages.early`` / ``stages.spore``
- **Super+** continuous product door edges generated from
  :mod:`super_metroid.routes.kpdr.spine` → ``stages.brinstar`` / ``stages.kraid``
- **K4** reverse / branch / shortcut edges (hand) → ``stages.speed``
"""

from __future__ import annotations

from super_metroid.progression.stages import (
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
]
