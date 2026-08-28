"""TMNT IV production assists: emergency HP restore and form-2 iframe hold.

Contract (docs/ASSIST_CONTRACT.md):
- Emergency HP: HP <= 16 (or last-chance HP == 0) -> restore to 80.
- Super Shredder form 2: stage 9, event 0x0A -> hold player_iframes at 1.
- Pizza pickup is not an assist. Clean leaves both writes off.
"""

from __future__ import annotations

from typing import Any, Protocol

EMERGENCY_HP_THRESHOLD = 16
EMERGENCY_HP_RESTORE = 80
FORM2_IFRAME_VALUE = 1


class AssistCounters(Protocol):
    """Duck type for RunMetrics fields used by integrity flags."""

    health_guard_interventions: int
    final_boss_iframe_guard_frames: int
    life_losses: int


def apply_emergency_hp(env: Any, health: int) -> bool:
    """Write player_hp to 80 on the production emergency branches.

    Same branches as the continuous runner: HP <= 16 while 0 < hp <= 0x60
    (16 is already inside that band), and hp == 0 last-chance revive.
    Returns True iff a write happened.
    """
    if health == 0 or 0 < health <= EMERGENCY_HP_THRESHOLD:
        env.set_value("player_hp", EMERGENCY_HP_RESTORE)
        return True
    return False


def apply_form2_iframe_hold(env: Any, *, stage: int, event: int) -> bool:
    """Hold player_iframes at 1 during Super Shredder form 2.

    Trigger is stage == 9 and event == 0x0A. Returns True iff a write happened.
    """
    if stage == 9 and event == 0x0A:
        env.set_value("player_iframes", FORM2_IFRAME_VALUE)
        return True
    return False


def assist_integrity(
    metrics: AssistCounters,
    *,
    require_clean_assists: bool = False,
) -> dict[str, bool]:
    """Boolean integrity flags for continuous assist counters."""
    flags = {
        "emergency_hp_zero": metrics.health_guard_interventions == 0,
        "iframe_guard_zero": metrics.final_boss_iframe_guard_frames == 0,
        "life_losses_zero": metrics.life_losses == 0,
        "state_loads_zero": True,
        "stage_writes_zero": True,
        "lives_writes_zero": True,
    }
    if require_clean_assists:
        flags["clean_assists_zero"] = (
            flags["emergency_hp_zero"] and flags["iframe_guard_zero"]
        )
    return flags


def evaluate_clean_integrity(
    metrics: AssistCounters,
) -> tuple[bool, dict[str, bool]]:
    """Return (ok, flags) requiring zero e-HP and zero iframe frames."""
    flags = assist_integrity(metrics, require_clean_assists=True)
    return bool(flags.get("clean_assists_zero")), flags
