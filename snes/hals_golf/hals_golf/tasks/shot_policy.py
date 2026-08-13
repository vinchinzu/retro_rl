"""Pure shot / putt planning for stroke play and VS HAL.

``StrokePlayMission`` only constructs tasks from these intents. Hole-in-one
search and harder difficulties should swap or wrap this layer — not grow the
mission phase machine.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Protocol, Sequence, runtime_checkable

from hals_golf.tasks.profile import MissionProfile
from hals_golf.tasks.routes.metal import (
    METAL_HOLE_CLUB_PLANS,
    METAL_HOLE_PUTT_PLANS,
    METAL_HOLE_SHOT_PLANS,
)
from hals_golf.tasks.routes.pro import (
    PRO_HOLE_CLUB_PLANS,
    PRO_HOLE_PUTT_PLANS,
    PRO_HOLE_SHOT_PLANS,
)
from hals_golf.tasks.routes.rest_bands import REST_BAND_OVERRIDES
from hals_golf.tasks.routes.rest_bands import RestBandOverride as RestBandOverride
from hals_golf.tasks.routes.tables import (
    HOLE_CLUB_PLANS,
    HOLE_PUTT_PLANS,
    HOLE_SHOT_PLANS,
    VS_HAL_HOLE_CLUB_PLANS,
    VS_HAL_HOLE_PUTT_PLANS,
    VS_HAL_HOLE_SHOT_PLANS,
    VS_HAL_METAL_HOLE_CLUB_PLANS,
    VS_HAL_METAL_HOLE_SHOT_PLANS,
)

PUTT_RETRY_ADJUSTMENTS = (0, -1, 1, -2, 2, -3, 3)
STALL_NUDGE_AIMS = (0, -12, 12, -24, 24, -36, 36)


@dataclass(frozen=True)
class ShotSituation:
    """Observable inputs for one command-menu decision."""

    hole: int
    strokes: int
    rest: int
    lie: int
    stall_nudges: int = 0
    putt_retries: int = 0
    last_putt_rest: int = -1
    default_power: int = 42


@dataclass(frozen=True)
class ShotIntent:
    """Full-swing parameters plus settle policy for ``ShotTask``."""

    power: int
    aim: int
    club_downs: int
    require_rest_change: bool
    complete_on_rest_zero: bool


@dataclass(frozen=True)
class PuttIntent:
    """Putting-meter parameters plus settle policy for ``PuttTask``."""

    power: int
    require_rest_change: bool
    complete_on_rest_zero: bool
    putt_retries: int
    last_putt_rest: int


def _merged_plans(
    hole: int,
    profile: MissionProfile,
) -> tuple[dict[int, tuple[int, int]], dict[int, int]]:
    if not profile.is_vs_hal:
        plan = dict(HOLE_SHOT_PLANS.get(hole, {}))
        club_plan = dict(HOLE_CLUB_PLANS.get(hole, {}))
        if profile.uses_metal:
            plan.update(METAL_HOLE_SHOT_PLANS.get(hole, {}))
            club_plan.update(METAL_HOLE_CLUB_PLANS.get(hole, {}))
        if profile.is_pro:
            plan.update(PRO_HOLE_SHOT_PLANS.get(hole, {}))
            club_plan.update(PRO_HOLE_CLUB_PLANS.get(hole, {}))
        return plan, club_plan
    plan = {
        **HOLE_SHOT_PLANS.get(hole, {}),
        **VS_HAL_HOLE_SHOT_PLANS.get(hole, {}),
    }
    club_plan = {
        **HOLE_CLUB_PLANS.get(hole, {}),
        **VS_HAL_HOLE_CLUB_PLANS.get(hole, {}),
    }
    if profile.uses_metal:
        plan.update(VS_HAL_METAL_HOLE_SHOT_PLANS.get(hole, {}))
        club_plan.update(VS_HAL_METAL_HOLE_CLUB_PLANS.get(hole, {}))
    if profile.is_pro:
        # Empty overlays today: a no-op merge until Pro tees are calibrated.
        plan.update(PRO_HOLE_SHOT_PLANS.get(hole, {}))
        club_plan.update(PRO_HOLE_CLUB_PLANS.get(hole, {}))
    return plan, club_plan


def _distance_fallback(
    rest: int,
    default_power: int,
) -> tuple[tuple[int, int], int]:
    if 0 < rest <= 55:
        return (32, 0), 12  # SW
    if rest <= 100:
        return (36, 0), 10  # 9I
    if rest <= 140:
        return (38, 0), 8  # 7I
    if rest <= 190:
        return (40, 0), 6  # 5I
    return (default_power, 0), 0


def _generic_putt_power(rest: int) -> int:
    if rest <= 5:
        return 42
    if rest <= 8:
        return 37 + rest
    if rest <= 12:
        return 42
    if rest <= 15:
        return 37 + rest
    return 52


def plan_putt(
    situation: ShotSituation,
    profile: MissionProfile,
) -> PuttIntent:
    """Choose putting-meter power for the current green leave."""
    base_power = _generic_putt_power(situation.rest)
    base_power = HOLE_PUTT_PLANS.get(situation.hole, {}).get(
        situation.rest,
        base_power,
    )
    if profile.uses_metal and not profile.is_vs_hal:
        base_power = METAL_HOLE_PUTT_PLANS.get(situation.hole, {}).get(
            situation.rest,
            base_power,
        )
    if profile.is_vs_hal:
        base_power = VS_HAL_HOLE_PUTT_PLANS.get(situation.hole, {}).get(
            situation.rest,
            base_power,
        )
    if profile.is_pro:
        # Empty overlay today: a no-op until Pro greens are calibrated.
        base_power = PRO_HOLE_PUTT_PLANS.get(situation.hole, {}).get(
            situation.rest,
            base_power,
        )

    if situation.rest == situation.last_putt_rest:
        putt_retries = situation.putt_retries + 1
        last_putt_rest = situation.last_putt_rest
    else:
        putt_retries = 0
        last_putt_rest = situation.rest

    adjustment = PUTT_RETRY_ADJUSTMENTS[
        putt_retries % len(PUTT_RETRY_ADJUSTMENTS)
    ]
    adjustment += situation.stall_nudges
    putt_power = max(10, min(54, base_power + adjustment))
    return PuttIntent(
        power=putt_power,
        require_rest_change=False,
        complete_on_rest_zero=profile.is_vs_hal,
        putt_retries=putt_retries,
        last_putt_rest=last_putt_rest,
    )


def _apply_rest_bands(
    situation: ShotSituation,
    profile: MissionProfile,
    power: int,
    aim: int,
    club_downs: int,
) -> tuple[int, int, int]:
    for rule in REST_BAND_OVERRIDES:
        if rule.matches(situation, profile):
            club_downs = rule.club_downs
            power = rule.power
            aim = rule.aim
    return power, aim, club_downs


def _apply_hole3_rest_bands(
    situation: ShotSituation,
    profile: MissionProfile,
    power: int,
    aim: int,
    club_downs: int,
) -> tuple[int, int, int]:
    """Hole 3 uses an if/elif REST ladder (not independent band ranges)."""
    if not profile.is_vs_hal:
        return power, aim, club_downs
    if situation.hole != 3 or situation.lie in (3, 6):
        return power, aim, club_downs
    rest = situation.rest
    lie = situation.lie
    if profile.uses_metal:
        if rest >= 450 and lie == 1:
            return 42, -6, 0
        if rest >= 250:
            return 42, -6, 2
        if rest >= 140:
            return 38, -2, 8
        return power, aim, club_downs
    if rest >= 450 and lie == 1:
        return 32, -20, 9
    if rest >= 350:
        return 42, -6, 4
    if rest >= 250:
        return 42, 0, 4
    if rest >= 140:
        return 36, 0, 10
    return power, aim, club_downs


def plan_shot(
    situation: ShotSituation,
    profile: MissionProfile,
) -> ShotIntent:
    """Choose full-swing power / aim / club for the current lie."""
    plan, club_plan = _merged_plans(situation.hole, profile)
    fallback, fallback_club = _distance_fallback(
        situation.rest,
        situation.default_power,
    )
    lie = situation.lie
    rest = situation.rest
    strokes = situation.strokes
    vs_hal = profile.is_vs_hal

    if lie == 1 and 0 in plan:
        power, aim = plan[0]
        club_downs = club_plan.get(0, 0)
    elif strokes in plan and not (strokes == 0 and lie != 1):
        power, aim = plan[strokes]
        club_downs = club_plan.get(strokes, 0)
        if (
            vs_hal
            and rest <= 55
            and lie not in (1, 3)
            and club_downs <= 2
        ):
            power, aim = fallback
            club_downs = fallback_club
    else:
        power, aim = fallback
        club_downs = fallback_club

    if vs_hal and lie == 3:
        if rest <= 80:
            club_downs = 0
            power = 36
            aim = 4
        elif rest <= 220:
            club_downs = 4
            power = 42
            if aim == 0:
                aim = -4
        else:
            club_downs = 0
            power = 38
            aim = -12
    elif lie == 3 and rest > 80 and club_downs < 8:
        club_downs = 8
        power = min(power, 38)
        if aim == 0:
            aim = -4

    if vs_hal and lie not in (1, 3) and club_downs <= 2 and rest <= 100:
        power, aim = fallback
        club_downs = fallback_club

    if vs_hal and lie not in (1, 3, 6) and 0 < rest <= 40:
        club_downs = 0
        power = 32
        aim = 0

    power, aim, club_downs = _apply_rest_bands(
        situation,
        profile,
        power,
        aim,
        club_downs,
    )
    power, aim, club_downs = _apply_hole3_rest_bands(
        situation,
        profile,
        power,
        aim,
        club_downs,
    )

    if situation.stall_nudges:
        if vs_hal and lie == 3 and rest > 220:
            power = max(
                36,
                min(40, 38 + (situation.stall_nudges % 3) - 1),
            )
            aim = -12
        else:
            aim += STALL_NUDGE_AIMS[
                situation.stall_nudges % len(STALL_NUDGE_AIMS)
            ]
            power = max(
                28,
                min(44, power + (situation.stall_nudges % 3) - 1),
            )

    return ShotIntent(
        power=power,
        aim=aim,
        club_downs=club_downs,
        # Stroke play and VS HAL both need this: a command-panel flash during
        # WAIT_FLIGHT used to count as success before the ball moved, which
        # desynced stroke-indexed plans (H2/H3 tee phantoms → double-digit
        # scores). Putts keep their own flag via ``plan_putt``.
        require_rest_change=True,
        # Stroke play must NOT finish on REST==0: the byte flashes zero during
        # ordinary flight and was recording fake HIOs (H13→1) that desynced
        # the rest of the round. VS HAL still needs the early exit.
        complete_on_rest_zero=vs_hal,
    )


@dataclass(frozen=True)
class SearchSpec:
    """Neighborhood + limits for :class:`HoleInOneSearchPolicy`.

    Deltas are applied additively to the deterministic base tee intent. The
    zero delta must stay first in each tuple so the base intent leads the
    expanded sequence.
    """

    power_deltas: tuple[int, ...] = (0, -2, 2, -4, 4)
    aim_deltas: tuple[int, ...] = (0, -4, 4, -8, 8)
    club_deltas: tuple[int, ...] = (0,)
    max_candidates: int = 25
    power_min: int = 28
    power_max: int = 44


@runtime_checkable
class RoutePolicy(Protocol):
    """Pluggable planner the mission consults for putts / shots / search.

    ``candidates`` is a pure, emulator-free expansion used by hole-in-one and
    recovery exploration. ``plan_putt`` / ``plan_shot`` remain the single
    intent the mission executes each command-menu decision.
    """

    def plan_putt(
        self,
        situation: ShotSituation,
        profile: MissionProfile,
    ) -> PuttIntent: ...

    def plan_shot(
        self,
        situation: ShotSituation,
        profile: MissionProfile,
    ) -> ShotIntent: ...

    def candidates(
        self,
        situation: ShotSituation,
        profile: MissionProfile,
    ) -> Sequence[ShotIntent]: ...


@dataclass(frozen=True)
class DeterministicRoutePolicy:
    """Default policy: exactly the calibrated ``plan_putt`` / ``plan_shot``.

    ``candidates`` returns the single deterministic intent so verified
    Amateur / VS HAL clears are preserved bit-for-bit.
    """

    def plan_putt(
        self,
        situation: ShotSituation,
        profile: MissionProfile,
    ) -> PuttIntent:
        return plan_putt(situation, profile)

    def plan_shot(
        self,
        situation: ShotSituation,
        profile: MissionProfile,
    ) -> ShotIntent:
        return plan_shot(situation, profile)

    def candidates(
        self,
        situation: ShotSituation,
        profile: MissionProfile,
    ) -> Sequence[ShotIntent]:
        return (plan_shot(situation, profile),)


def _is_tee_shot(situation: ShotSituation) -> bool:
    """Only expand the opening tee shot (stroke 0 on the tee lie)."""
    return situation.strokes == 0 and situation.lie == 1


@dataclass(frozen=True)
class HoleInOneSearchPolicy:
    """Expand a bounded neighborhood around the deterministic tee intent.

    Putts and non-tee shots defer to the deterministic planner. On the tee the
    base intent leads, followed by deduplicated power/aim/club variations up to
    ``spec.max_candidates``. The expansion is pure — it never touches the
    emulator — so callers can score candidates however they like.
    """

    spec: SearchSpec = field(default_factory=SearchSpec)

    def plan_putt(
        self,
        situation: ShotSituation,
        profile: MissionProfile,
    ) -> PuttIntent:
        return plan_putt(situation, profile)

    def plan_shot(
        self,
        situation: ShotSituation,
        profile: MissionProfile,
    ) -> ShotIntent:
        return plan_shot(situation, profile)

    def candidates(
        self,
        situation: ShotSituation,
        profile: MissionProfile,
    ) -> Sequence[ShotIntent]:
        base = plan_shot(situation, profile)
        if not _is_tee_shot(situation):
            return (base,)
        spec = self.spec
        results: list[ShotIntent] = [base]
        seen: set[tuple[int, int, int]] = {
            (base.power, base.aim, base.club_downs)
        }
        for power_delta in spec.power_deltas:
            for aim_delta in spec.aim_deltas:
                for club_delta in spec.club_deltas:
                    power = max(
                        spec.power_min,
                        min(spec.power_max, base.power + power_delta),
                    )
                    aim = base.aim + aim_delta
                    club_downs = max(0, base.club_downs + club_delta)
                    key = (power, aim, club_downs)
                    if key in seen:
                        continue
                    seen.add(key)
                    results.append(
                        replace(
                            base,
                            power=power,
                            aim=aim,
                            club_downs=club_downs,
                        )
                    )
                    if len(results) >= spec.max_candidates:
                        return tuple(results)
        return tuple(results)


def candidate_intents(
    situation: ShotSituation,
    profile: MissionProfile,
    *,
    policy: RoutePolicy | None = None,
    hole_in_one_search: bool = False,
) -> Sequence[ShotIntent]:
    """Search-space entry point for hole-in-one / recovery exploration.

    Defaults to the deterministic policy (a single verified intent). Pass an
    explicit ``policy`` or set ``hole_in_one_search`` to expand the tee
    neighborhood without touching ``StrokePlayMission``.
    """
    if policy is None:
        policy = (
            HoleInOneSearchPolicy()
            if hole_in_one_search
            else DeterministicRoutePolicy()
        )
    return policy.candidates(situation, profile)
