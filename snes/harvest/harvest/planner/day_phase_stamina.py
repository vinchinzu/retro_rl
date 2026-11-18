"""Spa + evening leftover-clear phase helpers.

Keep these off ``day_plan_phases`` (already ~1k LOC).
Weeding is done after dark to not waste daylight and maintained throughout the game.
Early Spring aggressively clears weeds; Fall re-clears bushes.
When stamina is empty/low, hot spring refreshes to full.
In Spring, ground is cleared and tilled.
"""

from __future__ import annotations

import re
from typing import List, Optional

from harvest.core.stamina import Stamina
from harvest.planner.day_phase_catalog import (
    CLEAR_BUSHES_PHASE,
    HOT_SPRING_STAMINA_PHASE,
    TILL_FIELD_PHASE,
)
from harvest.planner.day_phase_types import DayPlannerPolicy, PhaseSpec


# Hot-spring refills a day. Watering is 2 stamina a tile and a soak restores the whole bar, so each
# soak funds ~45 more tiles; the old cap of one refill per phase name left 20-50 tiles of a 100-tile
# Summer dry every night. A soak is ~5,000 frames, which costs wall time only (work runs on after
# the 18:xx clock freeze), so watering may refill again while its last pass watered something.
SPA_REFRESHES_PER_DAY = 2
WATER_SPA_REFRESHES_PER_DAY = 4
# Whole-field cover (``COVER_FIELD`` sows, ``COVER_TILL`` hoes) comes after the watering and has no daily refill cap (owner,
# 2026-11-03: "hoe the entire thing in one day and refill in spa"). A soak is ~6.7k frames of wall time and the clock is
# frozen at 18:xx, so the field's ~6,200 stamina is about sixty soaks. Another soak is allowed only while the last pass
# hoed or sowed something, which is what keeps a wedged planter from soaking for ever.
# Whole-field fodder cuts use the same guard with measured hay gains toward their stock target.
COVER_PHASES = frozenset({"COVER_FIELD", "COVER_TILL", "CUT_FODDER"})
# A pass that hoed and sowed nothing is forgiven this many times an evening: a soak can fail on a blocked gate (ROM probe:
# "no_path sealed" at the west gate, then the next try went through), and ending the sowing there would carry the bags overnight.
COVER_IDLE_PASSES = 3
_PROGRESS = re.compile(r"(?:tilled|planted|hay_added)=(\d+)")


def _made_progress(result_reason: str) -> bool:
    return any(int(n) > 0 for n in _PROGRESS.findall(result_reason or ""))


def _idle_marks(phase: str, refreshed_phases: set[str]) -> int:
    return sum(1 for mark in refreshed_phases if mark.startswith(f"{phase}#idle"))


def refresh_marks(phase: str, refreshed_phases: set[str], result_reason: str = "") -> set[str]:
    """What to record when a soak is spliced for ``phase``: its name, plus an idle mark for a cover pass that did nothing."""
    marks = {phase}
    # A repeat soak with no progress behind it is idle, whether the pass ran and did nothing or never started because the
    # last soak failed and left the bar empty (``empty_before_start``, no reason): both count, or a dog parked at the gate loops for ever.
    if phase in COVER_PHASES and phase in refreshed_phases and not _made_progress(result_reason):
        marks.add(f"{phase}#idle{_idle_marks(phase, refreshed_phases) + 1}")
    return marks


def spa_refresh_allowed(
    phase: str, refreshes_done: int, refreshed_phases: set[str], result_reason: str = ""
) -> bool:
    """Whether a ``stamina_low`` phase may splice another hot-spring refill and rerun."""
    if phase in COVER_PHASES:
        if phase not in refreshed_phases or _made_progress(result_reason):
            return True
        return _idle_marks(phase, refreshed_phases) < COVER_IDLE_PASSES
    limit = {"CROP_WATER": WATER_SPA_REFRESHES_PER_DAY}.get(phase, SPA_REFRESHES_PER_DAY)
    if refreshes_done >= limit:
        return False
    if phase not in refreshed_phases:
        return True
    if phase != "CROP_WATER":
        return False
    watered = re.search(r"watered=(\d+)", result_reason or "")
    unfunded = re.search(r"unfunded=(\d+)", result_reason or "")
    return bool(watered and int(watered.group(1)) > 0 and unfunded and int(unfunded.group(1)) > 0)


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


def evening_weed_phase(season: int = 0, day: int = 1) -> PhaseSpec:
    """After-dark weed clearing. Weed lift costs 0 stamina.

    Aggressive timeout on early days (D2-D5) to clear wilderness weeds;
    re-clear in Fall to clear respawned bushes; maintain through year.
    """
    if season == 0 and day <= 5:
        timeout = 18000
    elif season == 2:
        timeout = 16000
    else:
        timeout = 12000
    params = dict(CLEAR_BUSHES_PHASE.params)
    params["timeout"] = timeout
    return PhaseSpec(
        CLEAR_BUSHES_PHASE.phase,
        CLEAR_BUSHES_PHASE.kind,
        params,
        failure_policy="optional",
        required_maps=(0x00,),
        estimated_frames=timeout,
        failure_modes=("timeout_budget",),
    )


def till_field_phase(target_tiles: int = 32, timeout: int = 12000) -> PhaseSpec:
    """Till open ground with hoe in Spring."""
    params = dict(TILL_FIELD_PHASE.params)
    params["target_tiles"] = target_tiles
    params["timeout"] = timeout
    return PhaseSpec(
        TILL_FIELD_PHASE.phase,
        TILL_FIELD_PHASE.kind,
        params,
        failure_policy="optional",
        required_maps=(0x00,),
        estimated_frames=timeout,
        failure_modes=("timeout_budget", "stamina_low"),
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
    season: Optional[int] = None,
    day: int = 1,
) -> List[PhaseSpec]:
    """Bush/weed/rock clear + hot spring refresh + Spring tilling after dark.

    Weeding costs 0 stamina and is done after dark to not waste daylight.
    Early Spring aggressively clears weeds; Fall re-clears bushes.
    When stamina is low/exhausted, refresh at the hot spring.
    Spring tills and clears open ground.
    """
    if not late_day or not has_debris or (season is not None and season == 3):
        return []
    phases: List[PhaseSpec] = []
    include_spa = bool(getattr(policy, "include_spa", True))
    stam = coerce_stamina(stamina)

    # 1. Weeding after dark (costs 0 stamina, 0 daylight minutes)
    include_bush_clear = bool(getattr(policy, "include_bush_clear", True))
    if season is not None and include_bush_clear:
        first_days_spring = season == 0 and day <= 5
        fall_bushes = season == 2
        if has_debris or first_days_spring or fall_bushes:
            phases.append(evening_weed_phase(season=season, day=day))

    # 2. Debris clear (rocks, stumps) with tools
    if policy.include_field_clear and has_debris:
        if include_spa and (stam is None or stam.current <= 4 or not stam.can_finish_multi_hit()):
            phases.append(full_restore_spa_phase())
            stam = Stamina(100, 100)
        phases.append(evening_clear_field_phase())

    # 3. Ground mostly tilled and cleared in Spring
    if season == 0 and bool(getattr(policy, "include_spring_till", True)) and policy.include_field_clear:
        if include_spa and (stam is None or stam.current <= 4 or not stam.can_finish_multi_hit()):
            phases.append(full_restore_spa_phase())
        phases.append(till_field_phase())

    return phases


__all__ = [
    "coerce_stamina",
    "evening_clear_field_phase",
    "evening_clear_phases",
    "evening_weed_phase",
    "full_restore_spa_phase",
    "till_field_phase",
]


# A failed shed pickup of the watering can loses the night for every plant (a crop only grows on a watered night):
# 17 of 592 Year-3 Summer pickups failed across the r41..r47 batch and its previews, all but one right after the evening
# hay cut, and the day then had no second try. The farmer is outdoors with the whole evening ahead (the clock freezes at
# 18:xx), so a walk that wedged on the way is tried again before the night is given up. Only the walking failures are
# transient: of 42 retries 3 worked, all three "route failed" / "exit_to_farm failed"; "enter failed" at the shed door
# (farmer on the door stand, held=0x00, lock 1, no sprite near) and "missing after shelf" (pair [sickle, milker], sprinkler on the
# shelf, A ignored) fail the same way every time, so they are not retried (open bugs in NEXT_SESSION_TODO.md).
WATER_RETRY_ENV = "HARVEST_WATER_RETRY"
WATER_RETRY_FROM_YEAR = 2          # 0-based: Year 3 only; a Year 1-2 change would re-roll the timeline for no reason
MAX_WATER_RETRIES = 2
WATER_RETRY_REASONS = ("route failed", "exit_to_farm failed", "nav wedged")


def splice_water_retry(schedule, index: int, spec: PhaseSpec, reason: str, ram, done: set) -> bool:
    """On a shed-pickup failure of ``ENSURE_WATERING_CAN`` or ``CROP_WATER``, leave the building and try the pickup again.

    ``done`` is the day's one-shot ledger (tokens ``WATER_RETRY_n``); at most ``MAX_WATER_RETRIES`` a day, Year 3 only.
    """
    import os

    from harvest.core.ram_catalog import read_ram_value
    from harvest.planner.day_phase_catalog import ENSURE_WATERING_CAN_PHASE, EXIT_TO_FARM_PHASE

    if spec.phase not in {"ENSURE_WATERING_CAN", "CROP_WATER"} or not any(m in str(reason or "") for m in WATER_RETRY_REASONS):
        return False
    if os.environ.get(WATER_RETRY_ENV, "1").strip().lower() in {"", "0", "false", "no", "off"}:
        return False
    if int(read_ram_value(ram, "year", raw=True) or 0) < WATER_RETRY_FROM_YEAR:
        return False
    tries = sum(1 for token in done if str(token).startswith("WATER_RETRY"))
    if tries >= MAX_WATER_RETRIES:
        return False
    done.add(f"WATER_RETRY_{tries + 1}")
    retry = [EXIT_TO_FARM_PHASE, spec] if spec.phase == "ENSURE_WATERING_CAN" else [EXIT_TO_FARM_PHASE, ENSURE_WATERING_CAN_PHASE, spec]
    schedule.splice_at(index, retry)
    print(f"[DAY_PLAN] {spec.phase} failed ({reason}); leaving the building and fetching the can again ({tries + 1}/{MAX_WATER_RETRIES})")
    return True
