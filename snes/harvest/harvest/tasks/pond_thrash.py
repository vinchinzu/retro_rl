"""Data-driven pond-corridor densify thrash rules.

Extracted from ``CropRefillMixin._find_nav_path`` so geometry / stall
thresholds live as declarative rules instead of nested ifs. Pure:
``evaluate_corridor_thrash`` needs only start/goal + charge counters —
no emulator, no task instance.

Behavior is an intentional copy of the prior inline thrash branches
(rr-o00y / rr-5in / rr-qc9r / rr-5go9).
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Callable, List, Optional, Tuple

from harvest.tasks.nav import tile_dist

Tile = Tuple[int, int]
# densify last key: (start, goal) pair recorded while thrashing
ThrashLast = Tuple[Tile, Tile]
MatchFn = Callable[["ThrashCounters", Tile, Tile], bool]
LogFn = Callable[[Tile, int], str]


class ThrashChargeKind(str, Enum):
    """Scripted charge builders applied by CropRefillMixin."""

    EAST_SOUTH = "east_south"
    WEST_SOUTH_LIP = "west_south_lip"


class ThrashFireMode(str, Enum):
    """When a matched rule queues a charge."""

    # Fire on first match; clear densify stalls.
    IMMEDIATE = "immediate"
    # Increment stalls; fire when same (start,goal) last and stalls >= threshold.
    STALL_SAME_GOAL = "stall_same_goal"
    # Increment stalls; fire when stalls >= threshold (+ optional fire_gate).
    STALL_COUNT = "stall_count"


@dataclass(frozen=True)
class ThrashCounters:
    """Snapshot of PondCorridorController thrash fields used by rules."""

    east_south_charges: int = 0
    south_lip_charges: int = 0
    refill_densify_stalls: int = 0
    refill_densify_last: Optional[ThrashLast] = None


@dataclass(frozen=True)
class CorridorThrashRule:
    """One densify-thrash region / bail-out.

    ``match(counters, start, goal)`` is pure geometry + charge caps.
    ``fire_gate`` (optional) further gates STALL_* fire (e.g. near-F0
    short charge only while south_lip_charges < 4).
    """

    name: str
    priority: int  # lower runs first
    mode: ThrashFireMode
    charge: ThrashChargeKind
    match: MatchFn
    stall_threshold: int = 0  # IMMEDIATE ignores; STALL_* uses this
    require_same_last: bool = False  # STALL_SAME_GOAL
    fire_gate: Optional[Callable[[ThrashCounters], bool]] = None
    log: Optional[LogFn] = None


@dataclass(frozen=True)
class ThrashEvalResult:
    """Outcome of one densify thrash evaluation (apply in _find_nav_path)."""

    fire_charge: bool
    charge: Optional[ThrashChargeKind]
    log: str
    rule_name: Optional[str]
    # Updated densify counters (always applied by caller)
    refill_densify_stalls: int
    refill_densify_last: Optional[ThrashLast]


# ── match helpers (pure geometry + charge caps) ──────────────────────


def _match_past_fence_north(c: ThrashCounters, start: Tile, goal: Tile) -> bool:
    """Past fence end still north of wall — pure-south scripted charge."""
    return (
        start[0] >= 31
        and start[1] <= 31
        and goal[1] >= 32
        and c.east_south_charges < 6
    )


def _match_north_thrash(c: ThrashCounters, start: Tile, goal: Tile) -> bool:
    """Under / north of fence densify thrash toward F0 band."""
    return (
        start[1] <= 31
        and 18 <= start[0] <= 32
        and goal[0] >= 30
        and goal[1] >= 30
        and c.east_south_charges < 6
    )


def _match_near_f0(c: ThrashCounters, start: Tile, goal: Tile) -> bool:
    """Near F0 on south lip: prefer short densify, not long lip re-arm.

    Charge-cap for the eventual short fire lives in fire_gate, not match,
    so stalls still accumulate while south_lip is exhausted.
    """
    del c  # geometry-only
    return (
        start[1] >= 33
        and 26 <= start[0] <= 31
        and goal[0] >= 32
        and goal[1] >= 33
        and tile_dist(start, goal) <= 5
    )


def _match_south_thrash(c: ThrashCounters, start: Tile, goal: Tile) -> bool:
    """South of wall densify thrash (not near-F0 — that rule is higher priority)."""
    if _match_near_f0(c, start, goal):
        return False
    return (
        start[1] >= 32
        and start[0] <= 31
        and goal[0] >= 30
        and goal[1] >= 33
        and c.south_lip_charges < 8
    )


def _match_east_thrash(c: ThrashCounters, start: Tile, goal: Tile) -> bool:
    """East of pond densify (41,32)→F0 never moves — west lip charge."""
    return (
        start[0] >= 36
        and goal[0] <= 34
        and goal[1] >= 30
        and c.south_lip_charges < 8
    )


def _log_past_fence(start: Tile, stalls: int) -> str:
    del stalls
    return f"[CROP] Past-fence north at {start}; pure-south charge"


def _log_west_lip_thrash(start: Tile, stalls: int) -> str:
    return (
        f"[CROP] Densify thrash at {start}→F0 (n={stalls}); "
        f"west→south-lip charge"
    )


def _log_east_south_thrash(start: Tile, stalls: int) -> str:
    return (
        f"[CROP] Densify thrash at {start}→F0 (n={stalls}); "
        f"east→south corridor charge"
    )


def _log_near_f0(start: Tile, stalls: int) -> str:
    del stalls
    return f"[CROP] Near-F0 densify stall at {start}; short lip charge"


def _near_f0_fire_gate(c: ThrashCounters) -> bool:
    return c.south_lip_charges < 4


# Priority order mirrors the prior nested ifs:
#   1) past-fence immediate pure-south
#   2) north / south / east densify stall (shared stall counter; first match)
#   3) near-F0 long-stall short charge (elif in original)
# south_thrash already excludes near_f0 geometry; near_f0 is listed after
# south so an accidental dual-match still prefers south only when not near_f0.
CORRIDOR_THRASH_RULES: List[CorridorThrashRule] = [
    CorridorThrashRule(
        name="past_fence_north",
        priority=0,
        mode=ThrashFireMode.IMMEDIATE,
        charge=ThrashChargeKind.EAST_SOUTH,
        match=_match_past_fence_north,
        log=_log_past_fence,
    ),
    CorridorThrashRule(
        name="north_thrash",
        priority=10,
        mode=ThrashFireMode.STALL_SAME_GOAL,
        charge=ThrashChargeKind.EAST_SOUTH,
        match=_match_north_thrash,
        stall_threshold=2,
        require_same_last=True,
        log=_log_east_south_thrash,
    ),
    CorridorThrashRule(
        name="south_thrash",
        priority=20,
        mode=ThrashFireMode.STALL_SAME_GOAL,
        charge=ThrashChargeKind.WEST_SOUTH_LIP,
        match=_match_south_thrash,
        stall_threshold=2,
        require_same_last=True,
        log=_log_west_lip_thrash,
    ),
    CorridorThrashRule(
        name="east_thrash",
        priority=30,
        mode=ThrashFireMode.STALL_SAME_GOAL,
        charge=ThrashChargeKind.WEST_SOUTH_LIP,
        match=_match_east_thrash,
        stall_threshold=2,
        require_same_last=True,
        log=_log_west_lip_thrash,
    ),
    CorridorThrashRule(
        name="near_f0",
        priority=40,
        mode=ThrashFireMode.STALL_COUNT,
        charge=ThrashChargeKind.WEST_SOUTH_LIP,
        match=_match_near_f0,
        stall_threshold=6,
        require_same_last=False,
        fire_gate=_near_f0_fire_gate,
        log=_log_near_f0,
    ),
]


def _sorted_rules(rules: Optional[List[CorridorThrashRule]] = None) -> List[CorridorThrashRule]:
    src = rules if rules is not None else CORRIDOR_THRASH_RULES
    return sorted(src, key=lambda r: r.priority)


def match_thrash_rule(
    start: Tile,
    goal: Tile,
    counters: ThrashCounters,
    *,
    rules: Optional[List[CorridorThrashRule]] = None,
) -> Optional[CorridorThrashRule]:
    """First matching rule by priority, or None (caller resets stalls)."""
    for rule in _sorted_rules(rules):
        if rule.match(counters, start, goal):
            return rule
    return None


def evaluate_corridor_thrash(
    start: Tile,
    goal: Tile,
    counters: ThrashCounters,
    *,
    rules: Optional[List[CorridorThrashRule]] = None,
) -> ThrashEvalResult:
    """Evaluate densify thrash for one pathfind attempt.

    Pure: returns fire decision + updated stall counters. Caller queues
    the named charge builder and writes counters back onto the controller.
    """
    rule = match_thrash_rule(start, goal, counters, rules=rules)
    if rule is None:
        return ThrashEvalResult(
            fire_charge=False,
            charge=None,
            log="",
            rule_name=None,
            refill_densify_stalls=0,
            refill_densify_last=None,
        )

    if rule.mode is ThrashFireMode.IMMEDIATE:
        log = rule.log(start, 0) if rule.log else ""
        return ThrashEvalResult(
            fire_charge=True,
            charge=rule.charge,
            log=log,
            rule_name=rule.name,
            refill_densify_stalls=0,
            refill_densify_last=None,
        )

    stalls = counters.refill_densify_stalls + 1
    last_key: ThrashLast = (start, goal)

    if rule.mode is ThrashFireMode.STALL_SAME_GOAL:
        same = (
            counters.refill_densify_last == last_key
            if rule.require_same_last
            else True
        )
        if same and stalls >= rule.stall_threshold:
            if rule.fire_gate is not None and not rule.fire_gate(counters):
                return ThrashEvalResult(
                    fire_charge=False,
                    charge=None,
                    log="",
                    rule_name=rule.name,
                    refill_densify_stalls=stalls,
                    refill_densify_last=last_key,
                )
            log = rule.log(start, stalls) if rule.log else ""
            return ThrashEvalResult(
                fire_charge=True,
                charge=rule.charge,
                log=log,
                rule_name=rule.name,
                refill_densify_stalls=0,
                refill_densify_last=None,
            )
        return ThrashEvalResult(
            fire_charge=False,
            charge=None,
            log="",
            rule_name=rule.name,
            refill_densify_stalls=stalls,
            refill_densify_last=last_key,
        )

    # STALL_COUNT (near_f0): fire on threshold without requiring same last
    # (original always re-stamped last then checked stalls >= 6).
    if rule.mode is ThrashFireMode.STALL_COUNT:
        gate_ok = rule.fire_gate is None or rule.fire_gate(counters)
        if stalls >= rule.stall_threshold and gate_ok:
            log = rule.log(start, stalls) if rule.log else ""
            return ThrashEvalResult(
                fire_charge=True,
                charge=rule.charge,
                log=log,
                rule_name=rule.name,
                refill_densify_stalls=0,
                refill_densify_last=None,
            )
        return ThrashEvalResult(
            fire_charge=False,
            charge=None,
            log="",
            rule_name=rule.name,
            refill_densify_stalls=stalls,
            refill_densify_last=last_key,
        )

    # Defensive fallback — unknown mode: no fire, clear stalls like no match.
    return ThrashEvalResult(
        fire_charge=False,
        charge=None,
        log="",
        rule_name=rule.name,
        refill_densify_stalls=0,
        refill_densify_last=None,
    )



__all__ = [
    "CORRIDOR_THRASH_RULES",
    "CorridorThrashRule",
    "ThrashChargeKind",
    "ThrashCounters",
    "ThrashEvalResult",
    "ThrashFireMode",
    "evaluate_corridor_thrash",
    "match_thrash_rule",
]
