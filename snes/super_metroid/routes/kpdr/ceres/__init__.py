"""Ceres station reactive route policies (outbound + escape).

Package layout
--------------
* ``geometry`` — elev/magnet bands and hop *data* (``CERES_ELEV_HOPS``)
* ``arm_pump`` — classic L↔R pump + knockback recovery
* ``magnet`` — Magnet Stairs + Falling Tile reverse
* ``elev_escape`` — elev shaft climb → ship leave
* ``outbound`` — play_ceres_outbound_to_ridley / play_ceres_escape_to_landing
* ``first_room_fixture`` — searched hop/tape for first Ceres room (Elevator → Falling)

Takeoff types live in ``super_metroid.takeoff``. Knockback lives in
``routes.skills.knockback``. Import those from the owning module.

Public play names remain re-exported from ``early_spine`` for continuous/morph
import stability.
"""

from __future__ import annotations

from super_metroid.routes.kpdr.ceres.arm_pump import _arm_pump_dash_spans
from super_metroid.routes.kpdr.ceres.first_room_fixture import (
    CeresFirstRoomFixture,
    search_ceres_first_room,
    validate_ceres_first_room,
)
from super_metroid.routes.kpdr.ceres.geometry import _CERES_ARM_PUMP_PERIOD
from super_metroid.routes.kpdr.ceres.outbound import (
    play_ceres_escape_to_landing,
    play_ceres_outbound_to_ridley,
    play_ceres_to_ridley_door,
)

__all__ = [
    "_CERES_ARM_PUMP_PERIOD",
    "_arm_pump_dash_spans",
    "play_ceres_to_ridley_door",
    "play_ceres_outbound_to_ridley",
    "play_ceres_escape_to_landing",
    "CeresFirstRoomFixture",
    "search_ceres_first_room",
    "validate_ceres_first_room",
]
