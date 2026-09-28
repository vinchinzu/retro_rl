"""Barn cow phase specs (primitives only; composed lists live in catalog)."""

from __future__ import annotations

from typing import List

from harvest.planner.day_phase_types import PhaseSpec

# ── Barn cow phases ──

NAV_TO_BARN_PHASE = PhaseSpec(
    "NAV_TO_BARN",
    "multi_nav",
    {"route": "farm_to_barn", "timeout": 10000},
    required_maps=(0x00,),
    estimated_frames=800,
)

ENTER_BARN_PHASE = PhaseSpec(
    "ENTER_BARN",
    "directional_transition",
    {
        "direction": "up",
        "origin_tilemap": 0x00,
        "target_tilemap": 0x27,
        "timeout": 900,
        "stand_tile": (20, 22),
        "stand_tolerance": 0,
        "target_stand_tile": (8, 22),
        "target_stand_tolerance": 1,
        "settle_frames": 45,
        "door_align_px": 20 * 16 + 8,
        "overshoot_limit_px": 330,
        "require_empty_hands": True,
    },
    required_maps=(0x00,),
    estimated_frames=120,
)

COW_CHORES_PHASE = PhaseSpec(
    "COW_CHORES",
    "cow_chores",
    {"talk": True, "brush": True, "milk": True, "feed": True},
    required_maps=(0x27,),
    estimated_frames=8000,
    failure_modes=("slot_timeout", "nav_unreachable", "pixel_nav_stall", "sleeping_animal", "no_grass"),
)

EXIT_BARN_PHASE = PhaseSpec(
    "EXIT_BARN",
    "directional_transition",
    {
        "direction": "down",
        "origin_tilemap": 0x27,
        "target_tilemap": 0x00,
        "timeout": 1800,
        "stand_tile": (8, 22),
        "stand_tolerance": 1,
        "door_align_px": 8 * 16 + 8,
        "settle_frames": 5,
    },
    required_maps=(0x27,),
    estimated_frames=100,
)

BARN_CURRENT_COW_PHASES: List[PhaseSpec] = [
    COW_CHORES_PHASE,
    EXIT_BARN_PHASE,
]

__all__ = [
    "NAV_TO_BARN_PHASE",
    "ENTER_BARN_PHASE",
    "COW_CHORES_PHASE",
    "EXIT_BARN_PHASE",
    "BARN_CURRENT_COW_PHASES",
]
