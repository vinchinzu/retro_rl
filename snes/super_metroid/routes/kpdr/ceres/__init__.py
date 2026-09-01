"""Ceres station reactive route policies (outbound + escape).

Package layout
--------------
* ``geometry`` — elev/magnet bands and hop data (PlatformHop tables)
* ``arm_pump`` — classic L↔R pump + knockback recovery
* ``magnet`` — Magnet Stairs + Falling reverse + Elevator shaft to ship
* ``scientist`` — Dead Scientist never-jump run (outbound RIGHT, reverse LEFT)
* ``outbound`` — play_ceres_outbound_to_ridley / play_ceres_escape_to_landing
  (Ceres 1 moonfall + Ceres 2 magnet-feet + Ceres 3 jump-before-ledge)
* ``spine`` — CERES_SPINE / CERES_DOOR_EDGES / CERES_MILESTONES / boot /
  TAS comparison clocks. Morph composes this prefix.
* ``data/`` — living pins (first-control + hop leaves). Pin table in plan.md.

Takeoff types live in ``super_metroid.takeoff``. Knockback lives in
``routes.skills.knockback``. Import those from the owning module.

Public play names remain re-exported from ``early_spine`` for continuous/morph
import stability.
"""

from __future__ import annotations

from super_metroid.routes.kpdr.ceres.arm_pump import _arm_pump_dash_spans
from super_metroid.routes.kpdr.ceres.geometry import _CERES_ARM_PUMP_PERIOD
from super_metroid.routes.kpdr.ceres.outbound import (
    play_ceres_escape_to_landing,
    play_ceres_outbound_to_ridley,
    play_ceres_to_ridley_door,
)
from super_metroid.routes.kpdr.ceres.spine import (
    CERES_DOOR_EDGES,
    CERES_MILESTONES,
    CERES_SCIENTIST_MAX_FRAMES,
    CERES_SPINE,
    ceres_hops_vs_tas,
    play_boot_to_ceres,
    play_boot_to_ceres_tas,
)

__all__ = [
    "_CERES_ARM_PUMP_PERIOD",
    "_arm_pump_dash_spans",
    "play_ceres_to_ridley_door",
    "play_ceres_outbound_to_ridley",
    "play_ceres_escape_to_landing",
    "CERES_SPINE",
    "CERES_DOOR_EDGES",
    "CERES_MILESTONES",
    "CERES_SCIENTIST_MAX_FRAMES",
    "ceres_hops_vs_tas",
    "play_boot_to_ceres",
    "play_boot_to_ceres_tas",
]
