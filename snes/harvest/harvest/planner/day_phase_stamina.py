"""Spa + evening leftover-clear phase helpers.

Keep these off ``day_plan_phases`` (already ~1k LOC). D2 morning never
inserts spa; evening leftover rocks do when stamina cannot finish an
8-swing 2×2 (``rr-pzw``).
"""

from __future__ import annotations

from typing import List, Optional

from harvest.core.stamina import Stamina
from harvest.planner.day_phase_catalog import HOT_SPRING_STAMINA_PHASE
from harvest.planner.day_phase_types import DayPlannerPolicy, PhaseSpec


def full_restore_spa_phase() -> PhaseSpec:
    """Soak until current == max, then return to farm."""
    params = dict(HOT_SPRING_STAMINA_PHASE.params)
    params["min_stamina"] = "full"
    params["return_to_farm"] = True
    return PhaseSpec(
        HOT_SPRING_STAMINA_PHASE.phase,
        HOT_SPRING_STAMINA_PHASE.kind,
        params,
        failure_policy=HOT_SPRING_STAMINA_PHASE.failure_policy,
        required_ram=HOT_SPRING_STAMINA_PHASE.contract.required_ram,
        estimated_frames=HOT_SPRING_STAMINA_PHASE.contract.estimated_frames,
        failure_modes=HOT_SPRING_STAMINA_PHASE.contract.failure_modes,
    )


def evening_clear_field_phase() -> PhaseSpec:
    return PhaseSpec(
        "CLEAR_FIELD",
        "clear_field",
        {"timeout": 15000},
        failure_policy="optional",
        required_maps=(0x00,),
        estimated_frames=15000,
        failure_modes=("timeout_budget", "tool_missing", "stamina_low"),
    )


def coerce_stamina(value: Stamina | int | None) -> Optional[Stamina]:
    if value is None:
        return None
    if isinstance(value, Stamina):
        return value
    return Stamina(current=int(value), maximum=max(100, int(value)))


def evening_clear_phases(
    *,
    has_debris: bool,
    late_day: bool,
    policy: DayPlannerPolicy,
    stamina: Stamina | int | None = None,
) -> List[PhaseSpec]:
    """Bush/weed/rock clear only after the shipping window.

    Daytime CLEAR thrash was starving berry ship + seed buy on Spring D2.
    When evening stamina cannot finish a 6-hit/8-swing rock, insert spa
    first so leftover smash does not waste half a boulder.
    """
    if not late_day or not policy.include_field_clear or not has_debris:
        return []
    phases: List[PhaseSpec] = []
    stam = coerce_stamina(stamina)
    include_spa = bool(getattr(policy, "include_spa", True))
    if include_spa and stam is not None and not stam.can_finish_multi_hit():
        phases.append(full_restore_spa_phase())
    phases.append(evening_clear_field_phase())
    return phases


__all__ = [
    "coerce_stamina",
    "evening_clear_field_phase",
    "evening_clear_phases",
    "full_restore_spa_phase",
]
