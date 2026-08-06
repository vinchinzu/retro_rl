"""Ceres station reactive route policies (outbound + escape).

Extracted from :mod:`super_metroid.routes.kpdr.early_spine` so morph SpineHop
tables stay separate from elevator / magnet / arm-pump controllers.

Package layout
--------------
* ``geometry`` — elev/magnet bands, pose sets, room chains
* ``arm_pump`` — classic L↔R pump + knockback recovery
* ``magnet`` — Magnet Stairs + Falling Tile reverse
* ``elev_escape`` — elev shaft climb → ship leave
* ``outbound`` — play_ceres_outbound_to_ridley / play_ceres_escape_to_landing

Public play names remain re-exported from ``early_spine`` for continuous/morph
import stability.
"""

from __future__ import annotations

from super_metroid.routes.kpdr.ceres.arm_pump import (
    _arm_pump_dash_spans,
    _ceres_arm_pump_step,
    _ceres_arm_pump_until,
    _ceres_clear_knockback,
    _ceres_enemy_near,
    _ceres_is_knockback,
    _ceres_wait_ordinary,
)
from super_metroid.routes.kpdr.ceres.elev_escape import (
    _ceres_elev_leaving,
    _ceres_elev_ship_band,
    _ceres_elev_top_to_ship,
    _ceres_on_elev_ledge,
    _ceres_reactive_elev_climb,
    _ceres_shaft_spans,
)
from super_metroid.routes.kpdr.ceres.geometry import (
    _CERES_ARM_PUMP_PERIOD,
    _CERES_ELEV_BOTTOM_Y,
    _CERES_ELEV_LEDGE_POSE,
    _CERES_ELEV_LEDGE_Y,
    _CERES_ELEV_SHIP_X,
    _CERES_ELEV_SHIP_Y,
    _CERES_ELEV_TOP_X,
    _CERES_ELEV_TOP_Y,
    _CERES_ESCAPE_CHAIN,
    _CERES_KB_POSES,
    _CERES_MAGNET_EXIT_Y,
    _CERES_OUTBOUND_CHAIN,
    _CERES_WALL_LATCH,
)
from super_metroid.routes.kpdr.ceres.magnet import (
    _ceres_magnet_reached_falling,
    _ceres_magnet_step,
    _ceres_reactive_falling,
    _ceres_reactive_magnet_escape,
)
from super_metroid.routes.kpdr.ceres.outbound import (
    _ceres_escape_spans,
    _ceres_outbound_to_scientist_spans,
    play_ceres_escape_to_landing,
    play_ceres_outbound_to_ridley,
)

__all__ = [
    "_CERES_ARM_PUMP_PERIOD",
    "_CERES_KB_POSES",
    "_CERES_WALL_LATCH",
    "_CERES_ELEV_SHIP_Y",
    "_CERES_ELEV_SHIP_X",
    "_CERES_ELEV_TOP_Y",
    "_CERES_ELEV_TOP_X",
    "_CERES_ELEV_LEDGE_Y",
    "_CERES_ELEV_LEDGE_POSE",
    "_CERES_ELEV_BOTTOM_Y",
    "_CERES_MAGNET_EXIT_Y",
    "_CERES_OUTBOUND_CHAIN",
    "_CERES_ESCAPE_CHAIN",
    "_arm_pump_dash_spans",
    "_ceres_is_knockback",
    "_ceres_enemy_near",
    "_ceres_arm_pump_step",
    "_ceres_clear_knockback",
    "_ceres_arm_pump_until",
    "_ceres_wait_ordinary",
    "_ceres_magnet_reached_falling",
    "_ceres_magnet_step",
    "_ceres_reactive_magnet_escape",
    "_ceres_reactive_falling",
    "_ceres_shaft_spans",
    "_ceres_elev_ship_band",
    "_ceres_elev_leaving",
    "_ceres_on_elev_ledge",
    "_ceres_reactive_elev_climb",
    "_ceres_elev_top_to_ship",
    "_ceres_outbound_to_scientist_spans",
    "_ceres_escape_spans",
    "play_ceres_outbound_to_ridley",
    "play_ceres_escape_to_landing",
]
