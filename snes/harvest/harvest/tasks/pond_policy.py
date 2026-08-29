"""Pond corridor charge-completion policy and thrash state.

CorridorNavKind + decide_after_* helpers + PondCorridorController.
Charge action builders live in pond_charges; multi-hop densify in pond_hop.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Callable, List, Optional, Tuple

from harvest.maps.map_config import (
    FARM_MAIN_POND_STANDS,
    FARM_POND_ACCESS_FENCE_X_RANGE,
    farm_pond_refill_primary_stand,
)
from harvest.tasks.nav import tile_dist


# ── charge-completion policy kinds (applied by CropWaterTask) ────────
# Pure decide_* helpers return these; the mono only queues / commits.

class CorridorNavKind(str, Enum):
    """Kinds returned by decide_after_* (CropWaterTask applies them)."""

    QUEUE_EAST_SOUTH = "queue_east_south"
    QUEUE_GAP_SOUTH = "queue_gap_south"
    QUEUE_WEST_SOUTH_LIP = "queue_west_south_lip"
    ARM_F0_AND_LIP = "arm_f0_and_lip"  # set F0 targets + west lip charge
    ACT_AT_STAND = "act_at_stand"
    # Try multihop; on failure re-decide with a skip flag (south-lip fallthrough).
    TRY_MULTIHOP_CONTINUE = "try_multihop_continue"
    # Try multihop; on success maybe snap to act if adjacent; on fail re-decide.
    TRY_MULTIHOP_MAYBE_ACT_CONTINUE = "try_multihop_maybe_act_continue"
    # Multihop; on failure start_refill under water phase (no act snap).
    COMMIT_MULTIHOP_OR_REFILL = "commit_multihop_or_refill"
    # Multihop; if committed and adjacent stand → act; else start_refill.
    COMMIT_MULTIHOP_MAYBE_ACT_OR_REFILL = "commit_multihop_maybe_act_or_refill"


# Back-compat aliases (str Enum values compare equal to the old string constants).
KIND_QUEUE_EAST_SOUTH = CorridorNavKind.QUEUE_EAST_SOUTH
KIND_QUEUE_GAP_SOUTH = CorridorNavKind.QUEUE_GAP_SOUTH
KIND_QUEUE_WEST_SOUTH_LIP = CorridorNavKind.QUEUE_WEST_SOUTH_LIP
KIND_ARM_F0_AND_LIP = CorridorNavKind.ARM_F0_AND_LIP
KIND_ACT_AT_STAND = CorridorNavKind.ACT_AT_STAND
KIND_TRY_MULTIHOP_CONTINUE = CorridorNavKind.TRY_MULTIHOP_CONTINUE
KIND_TRY_MULTIHOP_MAYBE_ACT_CONTINUE = CorridorNavKind.TRY_MULTIHOP_MAYBE_ACT_CONTINUE
KIND_COMMIT_MULTIHOP_OR_REFILL = CorridorNavKind.COMMIT_MULTIHOP_OR_REFILL
KIND_COMMIT_MULTIHOP_MAYBE_ACT_OR_REFILL = CorridorNavKind.COMMIT_MULTIHOP_MAYBE_ACT_OR_REFILL

# Alternate F0 lip stand used when closer than primary after lip overshoot.
ALT_SOUTH_LIP_STAND: Tuple[int, int] = (33, 34)

# ── named stands ─────────────────────────────────────────────────────

# Primary F0 south-lip fill (map_config); do not hardcode bare (32,34) elsewhere.
PRIMARY_POND_STAND, PRIMARY_POND_FACE = farm_pond_refill_primary_stand()
assert PRIMARY_POND_STAND == FARM_MAIN_POND_STANDS[0][0]


# ── thrash / charge state ────────────────────────────────────────────

@dataclass
class PondCorridorController:
    """Owns scripted-charge counters and pending flags for pond approach.

    CropWaterTask holds one instance and delegates queue builders here so
    navigate no longer grows nested thrash special cases in-file.
    """

    east_south_charges: int = 0
    south_lip_charges: int = 0
    pending_gap_charge: bool = False
    pending_south_lip_charge: bool = False
    gap_south_tried: bool = False
    east_south_stuck_at: Optional[Tuple[int, int]] = None
    refill_densify_stalls: int = 0
    refill_densify_last: Optional[Tuple[int, int]] = None

    def reset(self) -> None:
        self.east_south_charges = 0
        self.south_lip_charges = 0
        self.pending_gap_charge = False
        self.pending_south_lip_charge = False
        self.gap_south_tried = False
        self.east_south_stuck_at = None
        self.refill_densify_stalls = 0
        self.refill_densify_last = None

    def soft_reset_charges(self) -> None:
        """Clear thrash counters after a failed refill tile (rr-qc9r).

        Does not clear pending_* charge flags — caller only invokes when not
        mid-scripted charge.
        """
        self.south_lip_charges = 0
        self.east_south_charges = 0
        self.refill_densify_stalls = 0
        self.refill_densify_last = None
        self.gap_south_tried = False
        self.east_south_stuck_at = None

    def note_east_south_queued(self) -> int:
        self.pending_gap_charge = True
        self.east_south_charges += 1
        return self.east_south_charges

    def note_gap_south_queued(self) -> int:
        self.gap_south_tried = True
        self.pending_gap_charge = True
        self.east_south_charges += 1
        return self.east_south_charges

    def note_south_lip_queued(self) -> int:
        self.pending_south_lip_charge = True
        self.south_lip_charges += 1
        return self.south_lip_charges


# ── gap predicate ────────────────────────────────────────────────────

def pond_corridor_gap_open(
    blocking_fence_count: int,
    fence_open_attempts: int = 0,
    *,
    full_wall_count: Optional[int] = None,
) -> bool:
    """True when the y=31 wall has a usable gap (not sealed, not unknown).

    Partial wall (some posts remain, ≥1 missing) is the common post-
    FenceClearLoopTask state. A completely empty fence row is only treated
    as open after we actually ran fence-open — otherwise blank unit maps
    with 0 fences would falsely multi-hop-commit to F0 stands.
    """
    if full_wall_count is None:
        x0, x1 = FARM_POND_ACCESS_FENCE_X_RANGE
        full_wall_count = (x1 - x0) + 1
    n = blocking_fence_count
    if 0 < n < full_wall_count:
        return True
    if n == 0 and fence_open_attempts > 0:
        return True
    return False


# ── charge-completion policy (navigate thrash sub-policy) ────────────
# Pure decide_* helpers: position + thrash counters → next corridor action.
# CropWaterTask applies kinds (queue / commit / act). Intentionally mirrors
# the prior _handle_navigate pending_* completion branches (rr-ds3 slice 2).

@dataclass(frozen=True)
class CorridorNavDecision:
    """Next action after a scripted pond-corridor charge completes."""

    kind: str
    log: str = ""
    stand: Optional[Tuple[int, int]] = None
    face: Optional[str] = None
    # When re-queueing east→south under the wall, record stuck tile if set.
    set_east_south_stuck_at: Optional[Tuple[int, int]] = None


def decide_after_multihop_drop(
    player: Tuple[int, int],
    *,
    south_lip_charges: int = 0,
) -> CorridorNavDecision:
    """Hands empty after fence-open local drop — south lip vs east→south."""
    if player[1] >= 32:
        log = f"[CROP] Hands empty south of wall at {player}; multi-hop F0"
        if (
            tile_dist(player, PRIMARY_POND_STAND) > 3
            and player[0] <= 28
            and south_lip_charges < 2
        ):
            return CorridorNavDecision(
                kind=KIND_ARM_F0_AND_LIP,
                log=log,
                stand=PRIMARY_POND_STAND,
                face=PRIMARY_POND_FACE,
            )
        return CorridorNavDecision(
            kind=KIND_COMMIT_MULTIHOP_OR_REFILL,
            log=log,
        )
    return CorridorNavDecision(
        kind=KIND_QUEUE_EAST_SOUTH,
        log=(
            f"[CROP] Hands empty north of wall at {player}; "
            f"east→south corridor charge"
        ),
    )


def decide_after_gap_reseat(
    player: Tuple[int, int],
) -> CorridorNavDecision:
    """After N/E gap-drop nudge drains — re-cross or multi-hop F0."""
    if player[1] < 32:
        return CorridorNavDecision(kind=KIND_QUEUE_EAST_SOUTH)
    return CorridorNavDecision(
        kind=KIND_COMMIT_MULTIHOP_OR_REFILL,
        log=f"[CROP] Gap nudge done at {player}; multi-hop F0",
    )


def decide_after_east_south_charge(
    player: Tuple[int, int],
    *,
    east_south_charges: int,
    south_lip_charges: int = 0,
    east_south_stuck_at: Optional[Tuple[int, int]] = None,
    gap_south_tried: bool = False,
) -> CorridorNavDecision:
    """After east→south (or gap-south) scripted charge queue drains."""
    header = (
        f"[CROP] Corridor charge done at {player}; multi-hop F0 "
        f"(y={'ok' if player[1] >= 32 else 'still_north'})"
    )
    # Still north: re-queue. East-only while x<31; south once x≥31.
    # Cap 6 — power-on (29,30) soft-block needs several east legs.
    # After 3 failed ends at same tile, fall back to gap-south charge
    # (power-on residual: RIGHT never advances past x=29).
    n_es = east_south_charges
    if player[1] <= 31 and n_es < 6:
        stuck_same = (
            player[0] <= 29
            and n_es >= 3
            and east_south_stuck_at == player
        )
        if stuck_same and not gap_south_tried:
            return CorridorNavDecision(
                kind=KIND_QUEUE_GAP_SOUTH,
                log=(
                    f"{header}\n"
                    f"[CROP] East stuck at {player} n={n_es}; "
                    f"gap-south fallback via open corridor"
                ),
            )
        stuck = player if player[0] <= 29 else None
        return CorridorNavDecision(
            kind=KIND_QUEUE_EAST_SOUTH,
            log=(
                f"{header}\n"
                f"[CROP] Still north at {player}; re-queue east→south "
                f"(need x≥31 past fence end then y≥32)"
            ),
            set_east_south_stuck_at=stuck,
        )
    # Landed south but not at stand: script west→south-lip / east-to-F0.
    # rr-qc9r: after corridor, prior water tiles may have exhausted
    # charges — still arm if under raised cap so late spring refills.
    if (
        player[1] >= 32
        and tile_dist(player, PRIMARY_POND_STAND) > 1
        and south_lip_charges < 10
    ):
        return CorridorNavDecision(
            kind=KIND_QUEUE_WEST_SOUTH_LIP,
            log=(
                f"{header}\n"
                f"[CROP] South-of-wall at {player}; "
                f"queue west→south-lip to F0"
            ),
        )
    return CorridorNavDecision(
        kind=KIND_COMMIT_MULTIHOP_OR_REFILL,
        log=header,
    )


def decide_after_south_lip_charge(
    player: Tuple[int, int],
    *,
    south_lip_charges: int,
    east_south_charges: int,
    north_band_multihop_tried: bool = False,
    near_f0_multihop_tried: bool = False,
) -> CorridorNavDecision:
    """After west→south-lip charge queue drains (soft-block / south thrash).

    ``north_band_multihop_tried`` / ``near_f0_multihop_tried`` skip the
    fall-through try branches after a failed multihop (composer re-calls).
    """
    header = f"[CROP] South-lip charge done at {player}; multi-hop/act F0"
    # On/near south-lip stand: act fill immediately.
    if (
        tile_dist(player, PRIMARY_POND_STAND) <= 1
        or tile_dist(player, ALT_SOUTH_LIP_STAND) <= 1
    ):
        stand = (
            PRIMARY_POND_STAND
            if tile_dist(player, PRIMARY_POND_STAND)
            <= tile_dist(player, ALT_SOUTH_LIP_STAND)
            else ALT_SOUTH_LIP_STAND
        )
        return CorridorNavDecision(
            kind=KIND_ACT_AT_STAND,
            log=header,
            stand=stand,
            face=PRIMARY_POND_FACE,
        )
    # Far east of pond ONLY when south of fence (y≥32). North+east
    # ~(36,24) is mountain drift — pure south, not west thrash (rr-5go9).
    if (
        player[0] >= 36
        and player[1] >= 32
        and tile_dist(player, PRIMARY_POND_STAND) > 1
        and south_lip_charges < 12
    ):
        return CorridorNavDecision(
            kind=KIND_QUEUE_WEST_SOUTH_LIP,
            log=(
                f"{header}\n"
                f"[CROP] South-lip east-of-pond at {player}; "
                f"west charge to F0"
            ),
        )
    # Drifted north of wall at high x (mountain lip): pure south back.
    if (
        player[1] <= 30
        and player[0] >= 30
        and east_south_charges < 6
    ):
        return CorridorNavDecision(
            kind=KIND_QUEUE_EAST_SOUTH,
            log=(
                f"{header}\n"
                f"[CROP] South-lip north drift at {player}; "
                f"pure-south corridor charge"
            ),
        )
    # Near pond north lip after overshoot (e.g. (33,30)): multihop/act F0.
    # Try once; on failure fall through to near-F0 / re-queue checks.
    if (
        not north_band_multihop_tried
        and 30 <= player[0] < 36
        and player[1] <= 31
    ):
        return CorridorNavDecision(
            kind=KIND_TRY_MULTIHOP_CONTINUE,
            log=(
                f"{header}\n"
                f"[CROP] South-lip near north pond band at {player}; "
                f"multi-hop F0 (no east re-cross)"
            ),
        )
    # Near F0 on south lip (dist≤4, x~28–34, y≥33): multihop/act —
    # do NOT re-queue long RIGHT charges that overshoot to (36,36)
    # (dry fixture residual after soft charge lands ~(29,35)).
    near_f0 = (
        tile_dist(player, PRIMARY_POND_STAND) <= 4
        and player[1] >= 33
        and 26 <= player[0] <= 35
    )
    if near_f0 and not near_f0_multihop_tried:
        return CorridorNavDecision(
            kind=KIND_TRY_MULTIHOP_MAYBE_ACT_CONTINUE,
            log=(
                f"{header}\n"
                f"[CROP] South-lip near F0 at {player}; multi-hop/act "
                f"(skip re-charge thrash)"
            ),
        )
    # Still south short of stand: re-queue lip (position-banded).
    # Cap 8 — higher caps + pure-east overshot past F0 (rr-qc9r dry).
    lip_cap = 8 if player[1] >= 34 else 7
    if (
        tile_dist(player, PRIMARY_POND_STAND) > 1
        and player[1] >= 32
        and south_lip_charges < lip_cap
    ):
        return CorridorNavDecision(
            kind=KIND_QUEUE_WEST_SOUTH_LIP,
            log=(
                f"{header}\n"
                f"[CROP] South-lip short of F0 at {player}; re-queue lip charge"
            ),
        )
    # Drifted north of wall: re-cross (prefer east when near fence end).
    if (
        player[1] <= 31
        and player[0] < 31
        and east_south_charges < 6
    ):
        return CorridorNavDecision(
            kind=KIND_QUEUE_EAST_SOUTH,
            log=(
                f"{header}\n"
                f"[CROP] South-lip drifted north at {player}; "
                f"east→south re-cross"
            ),
        )
    return CorridorNavDecision(
        kind=KIND_COMMIT_MULTIHOP_MAYBE_ACT_OR_REFILL,
        log=header,
    )


# ── densify-thrash rule table (applied by CropRefillMixin) ───────────

Tile = Tuple[int, int]
ThrashLast = Tuple[Tile, Tile]
MatchFn = Callable[["ThrashCounters", Tile, Tile], bool]
LogFn = Callable[[Tile, int], str]


class ThrashChargeKind(str, Enum):
    """Scripted charge builders applied by CropRefillMixin."""

    EAST_SOUTH = "east_south"
    WEST_SOUTH_LIP = "west_south_lip"


class ThrashFireMode(str, Enum):
    """When a matched rule queues a charge."""

    IMMEDIATE = "immediate"
    STALL_SAME_GOAL = "stall_same_goal"
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
    """One densify-thrash region / bail-out."""

    name: str
    priority: int
    mode: ThrashFireMode
    charge: ThrashChargeKind
    match: MatchFn
    stall_threshold: int = 0
    require_same_last: bool = False
    fire_gate: Optional[Callable[["ThrashCounters"], bool]] = None
    log: Optional[LogFn] = None


@dataclass(frozen=True)
class ThrashEvalResult:
    """Outcome of one densify thrash evaluation (apply in _find_nav_path)."""

    fire_charge: bool
    charge: Optional[ThrashChargeKind]
    log: str
    rule_name: Optional[str]
    refill_densify_stalls: int
    refill_densify_last: Optional[ThrashLast]


def _match_past_fence_north(c: ThrashCounters, start: Tile, goal: Tile) -> bool:
    return (
        start[0] >= 31
        and start[1] <= 31
        and goal[1] >= 32
        and c.east_south_charges < 6
    )


def _match_north_thrash(c: ThrashCounters, start: Tile, goal: Tile) -> bool:
    return (
        start[1] <= 31
        and 18 <= start[0] <= 32
        and goal[0] >= 30
        and goal[1] >= 30
        and c.east_south_charges < 6
    )


def _match_near_f0(c: ThrashCounters, start: Tile, goal: Tile) -> bool:
    del c
    return (
        start[1] >= 33
        and 26 <= start[0] <= 31
        and goal[0] >= 32
        and goal[1] >= 33
        and tile_dist(start, goal) <= 5
    )


def _match_south_thrash(c: ThrashCounters, start: Tile, goal: Tile) -> bool:
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


# Priority: past-fence immediate → north/south/east stall → near-F0 long stall.
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


def _thrash_result(
    *,
    fire: bool = False,
    charge: Optional[ThrashChargeKind] = None,
    log: str = "",
    name: Optional[str] = None,
    stalls: int = 0,
    last: Optional[ThrashLast] = None,
) -> ThrashEvalResult:
    return ThrashEvalResult(
        fire_charge=fire,
        charge=charge,
        log=log,
        rule_name=name,
        refill_densify_stalls=stalls,
        refill_densify_last=last,
    )


def evaluate_corridor_thrash(
    start: Tile,
    goal: Tile,
    counters: ThrashCounters,
    *,
    rules: Optional[List[CorridorThrashRule]] = None,
) -> ThrashEvalResult:
    """Pure densify-thrash decision + updated stall counters."""
    rule = match_thrash_rule(start, goal, counters, rules=rules)
    if rule is None:
        return _thrash_result()

    if rule.mode is ThrashFireMode.IMMEDIATE:
        log = rule.log(start, 0) if rule.log else ""
        return _thrash_result(fire=True, charge=rule.charge, log=log, name=rule.name)

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
                return _thrash_result(name=rule.name, stalls=stalls, last=last_key)
            log = rule.log(start, stalls) if rule.log else ""
            return _thrash_result(
                fire=True, charge=rule.charge, log=log, name=rule.name
            )
        return _thrash_result(name=rule.name, stalls=stalls, last=last_key)

    if rule.mode is ThrashFireMode.STALL_COUNT:
        gate_ok = rule.fire_gate is None or rule.fire_gate(counters)
        if stalls >= rule.stall_threshold and gate_ok:
            log = rule.log(start, stalls) if rule.log else ""
            return _thrash_result(
                fire=True, charge=rule.charge, log=log, name=rule.name
            )
        return _thrash_result(name=rule.name, stalls=stalls, last=last_key)

    return _thrash_result(name=rule.name)


__all__ = [
    "ALT_SOUTH_LIP_STAND",
    "CORRIDOR_THRASH_RULES",
    "CorridorNavDecision",
    "CorridorNavKind",
    "CorridorThrashRule",
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
    "PondCorridorController",
    "ThrashChargeKind",
    "ThrashCounters",
    "ThrashEvalResult",
    "ThrashFireMode",
    "decide_after_east_south_charge",
    "decide_after_gap_reseat",
    "decide_after_multihop_drop",
    "decide_after_south_lip_charge",
    "evaluate_corridor_thrash",
    "match_thrash_rule",
    "pond_corridor_gap_open",
]
