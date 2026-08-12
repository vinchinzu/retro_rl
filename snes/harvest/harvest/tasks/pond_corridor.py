"""Pond corridor charge scripts + multi-hop densify for can refill.

Compatibility barrel: public API re-exported from focused modules.

- ``pond_policy`` — CorridorNavKind, decide_after_*, PondCorridorController
- ``pond_charges`` — build_* scripted charge action lists
- ``pond_hop`` — compute_refill_hop_goal multi-hop densify
- ``pond_thrash`` — densify thrash region rules + evaluate_corridor_thrash

Extracted from CropWaterTask (rr-ds3) so thrash counters, scripted
east→south / west→south-lip / gap-south routes, and charge-*completion*
policy live outside the crop_planter mono. Behavior is intentional copy
of the prior private methods / navigate thrash branches.
"""

from __future__ import annotations

from harvest.tasks.pond_charges import (
    FENCE_WALL_END_X,
    SOFT_BLOCK_Y_BAND,
    SOUTH_LIP_Y,
    build_east_south_corridor_charge,
    build_gap_south_fallback,
    build_west_south_lip_charge,
)
from harvest.tasks.pond_hop import PathFn, compute_refill_hop_goal
from harvest.tasks.pond_policy import (
    ALT_SOUTH_LIP_STAND,
    KIND_ACT_AT_STAND,
    KIND_ARM_F0_AND_LIP,
    KIND_COMMIT_MULTIHOP_MAYBE_ACT_OR_REFILL,
    KIND_COMMIT_MULTIHOP_OR_REFILL,
    KIND_QUEUE_EAST_SOUTH,
    KIND_QUEUE_GAP_SOUTH,
    KIND_QUEUE_WEST_SOUTH_LIP,
    KIND_TRY_MULTIHOP_CONTINUE,
    KIND_TRY_MULTIHOP_MAYBE_ACT_CONTINUE,
    PRIMARY_POND_FACE,
    PRIMARY_POND_STAND,
    CorridorNavDecision,
    CorridorNavKind,
    PondCorridorController,
    decide_after_east_south_charge,
    decide_after_gap_reseat,
    decide_after_multihop_drop,
    decide_after_south_lip_charge,
    pond_corridor_gap_open,
)
from harvest.tasks.pond_thrash import (
    CORRIDOR_THRASH_RULES,
    CorridorThrashRule,
    ThrashChargeKind,
    ThrashCounters,
    ThrashEvalResult,
    ThrashFireMode,
    evaluate_corridor_thrash,
    match_thrash_rule,
)

__all__ = [
    "ALT_SOUTH_LIP_STAND",
    "CORRIDOR_THRASH_RULES",
    "CorridorNavDecision",
    "CorridorNavKind",
    "CorridorThrashRule",
    "FENCE_WALL_END_X",
    "KIND_ACT_AT_STAND",
    "KIND_ARM_F0_AND_LIP",
    "KIND_COMMIT_MULTIHOP_MAYBE_ACT_OR_REFILL",
    "KIND_COMMIT_MULTIHOP_OR_REFILL",
    "KIND_QUEUE_EAST_SOUTH",
    "KIND_QUEUE_GAP_SOUTH",
    "KIND_QUEUE_WEST_SOUTH_LIP",
    "KIND_TRY_MULTIHOP_CONTINUE",
    "KIND_TRY_MULTIHOP_MAYBE_ACT_CONTINUE",
    "PRIMARY_POND_FACE",
    "PRIMARY_POND_STAND",
    "PathFn",
    "SOFT_BLOCK_Y_BAND",
    "SOUTH_LIP_Y",
    "PondCorridorController",
    "ThrashChargeKind",
    "ThrashCounters",
    "ThrashEvalResult",
    "ThrashFireMode",
    "build_east_south_corridor_charge",
    "build_gap_south_fallback",
    "build_west_south_lip_charge",
    "compute_refill_hop_goal",
    "decide_after_east_south_charge",
    "decide_after_gap_reseat",
    "decide_after_multihop_drop",
    "decide_after_south_lip_charge",
    "evaluate_corridor_thrash",
    "match_thrash_rule",
    "pond_corridor_gap_open",
]
