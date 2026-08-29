"""Failure / thrash recovery policy for ReturnHomeTask.

``home_approach`` owns approach geometry; this module owns what to do when
child phases fail (exit-to-farm dialogue mash, enter-house south recover,
nav near-door force enter, mid-yard re-nav, south softlock escape).

Pure helpers only — ReturnHomeTask applies decisions (counters, queues, prints).
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import List, Optional

import numpy as np

from harvest.planner.tasks.home_approach import (
    EAST_AROUND_FENCE_X,
    deep_south_of_house,
    south_of_fence_wall,
)
from harvest.planner.tasks.transitions import (
    multi_face_toss_actions,
    toss_held_actions,
)
from harvest.tasks.nav import Point, make_action


class RecoverKind(str, Enum):
    """What ReturnHomeTask should do after a child FAILURE/BLOCKED."""

    QUEUE_EXIT_MASH = "queue_exit_mash"
    FAIL_EXIT = "fail_exit"
    RETRY_ENTER_SOUTH = "retry_enter_south"
    RETRY_ENTER_RESTART = "retry_enter_restart"
    FORCE_ENTER = "force_enter"
    MID_YARD_RENAV = "mid_yard_renav"
    SOUTH_ESCAPE = "south_escape"
    RETRY_DROP_THEN_NAV = "retry_drop_then_nav"
    HARD_FAIL = "hard_fail"


@dataclass(frozen=True)
class RecoverDecision:
    kind: RecoverKind
    reason: str = ""
    # SOUTH_ESCAPE: prefer far-east thrash (west then north).
    far_east: bool = False
    # Enter-fail: reset drop budget when reason mentions hands.
    hands_not_clear: bool = False
    # HARD_FAIL after hands-full path mutates phase before reporting.
    clear_task: bool = False
    set_phase: Optional[str] = None
    # SOUTH_ESCAPE log / result.reason variant (drop vs house nav).
    escape_from_drop: bool = False


def exit_to_farm_recover_actions() -> List[np.ndarray]:
    """Multi-face A/B mash used after ExitToFarm dialogue/unknown thrash."""
    actions: List[np.ndarray] = []
    for face in ("down", "left", "right", "up"):
        actions.extend(make_action(**{face: True}) for _ in range(4))
        actions.extend(make_action(a=True, b=True) for _ in range(8))
        actions.extend(make_action() for _ in range(4))
    return actions


def enter_fail_south_recovery_actions(
    pos: Point, front: Point
) -> Optional[List[np.ndarray]]:
    """South walk-out when enter failed north of the outdoor stand.

    Returns None when geometry does not apply (``pos.y >= front.y - 16``).
    """
    if pos.y >= front.y - 16:
        return None
    actions: List[np.ndarray] = []
    actions.extend(make_action(left=True) for _ in range(10))
    actions.extend(make_action(down=True, b=True) for _ in range(40))
    actions.extend(make_action(right=True) for _ in range(12))
    actions.extend(make_action(down=True, b=True) for _ in range(24))
    actions.extend(make_action() for _ in range(8))
    return actions


def decide_exit_to_farm_failure(
    *,
    retries: int,
    retry_limit: int,
    reason: str,
) -> RecoverDecision:
    if retries < retry_limit:
        return RecoverDecision(kind=RecoverKind.QUEUE_EXIT_MASH, reason=reason)
    return RecoverDecision(kind=RecoverKind.FAIL_EXIT, reason=reason)


def decide_enter_house_failure(
    *,
    pos: Point,
    front: Point,
    enter_retries: int,
    enter_retry_limit: int = 4,
    reason: str,
) -> Optional[RecoverDecision]:
    """Return a retry decision, or None if enter retries are exhausted."""
    if enter_retries >= enter_retry_limit:
        return None
    hands_not_clear = "hands not clear" in reason
    if enter_fail_south_recovery_actions(pos, front) is not None:
        return RecoverDecision(
            kind=RecoverKind.RETRY_ENTER_SOUTH,
            reason=reason,
            hands_not_clear=hands_not_clear,
        )
    return RecoverDecision(
        kind=RecoverKind.RETRY_ENTER_RESTART,
        reason=reason,
        hands_not_clear=hands_not_clear,
    )


def decide_nav_failure(
    *,
    phase: str,
    pos: Point,
    front: Point,
    hands_clear: bool,
    drop_attempts: int,
    drop_attempt_limit: int,
    offstand_corrections: int,
    south_escape_attempts: int,
    south_escape_limit: int,
    reason: str,
) -> RecoverDecision:
    """Policy for nav_house_front / nav_drop_spot FAILURE or BLOCKED."""
    # Hands still full: drop then re-nav. Exhausted budget mutates phase to
    # start then hard-fails (same as prior inline path).
    if phase in {"nav_house_front", "nav_drop_spot"} and not hands_clear:
        if drop_attempts < drop_attempt_limit:
            return RecoverDecision(
                kind=RecoverKind.RETRY_DROP_THEN_NAV,
                reason=reason,
            )
        return RecoverDecision(
            kind=RecoverKind.HARD_FAIL,
            reason=reason,
            clear_task=True,
            set_phase="start",
        )

    if phase == "nav_house_front":
        dx = abs(pos.x - front.x)
        dy = abs(pos.y - front.y)
        # Generous near-door force-enter after multi_nav timeout:
        # - D12 residual (118,486) ~62px south of stand
        # - Gate B D5 (190,423) vs front (136,424): dx=54 lateral, dy=1
        #   (old dx<=48 left that as HARD_FAIL → terminal return_home)
        near_door = (dx <= 48 and dy <= 80) or (dx <= 72 and dy <= 32)
        if near_door:
            return RecoverDecision(kind=RecoverKind.FORCE_ENTER, reason=reason)
        # Mid-yard south of door (north of fence): simple re-nav north.
        if (
            not south_of_fence_wall(pos)
            and pos.y > front.y + 24
            and dx <= 80
            and offstand_corrections < 6
        ):
            return RecoverDecision(kind=RecoverKind.MID_YARD_RENAV, reason=reason)
        # South-of-fence / far-from-door multi_nav timeout: B-run escape.
        # Lateral miss past force-enter but not deep south: still try renav once
        # via escape only when truly far.
        door_far = dx + dy > 80
        if (
            (
                deep_south_of_house(pos, front)
                or south_of_fence_wall(pos)
                or door_far
            )
            and south_escape_attempts < south_escape_limit
        ):
            return RecoverDecision(
                kind=RecoverKind.SOUTH_ESCAPE,
                reason=reason,
                far_east=pos.x > EAST_AROUND_FENCE_X,
                escape_from_drop=False,
            )

    if phase == "nav_drop_spot":
        if (
            (deep_south_of_house(pos, front) or south_of_fence_wall(pos))
            and south_escape_attempts < south_escape_limit
        ):
            return RecoverDecision(
                kind=RecoverKind.SOUTH_ESCAPE,
                reason=reason,
                far_east=pos.x > EAST_AROUND_FENCE_X,
                escape_from_drop=True,
            )

    return RecoverDecision(kind=RecoverKind.HARD_FAIL, reason=reason)


def decide_child_failure(
    *,
    phase: str,
    pos: Point,
    front: Point,
    reason: str,
    hands_clear: bool,
    exit_to_farm_retries: int,
    exit_to_farm_retry_limit: int,
    enter_retries: int,
    enter_retry_limit: int = 4,
    drop_attempts: int,
    drop_attempt_limit: int,
    offstand_corrections: int,
    south_escape_attempts: int,
    south_escape_limit: int,
) -> RecoverDecision:
    """Top-level policy for ReturnHomeTask child FAILURE/BLOCKED."""
    if phase == "exit_to_farm":
        return decide_exit_to_farm_failure(
            retries=exit_to_farm_retries,
            retry_limit=exit_to_farm_retry_limit,
            reason=reason,
        )
    if phase == "enter_house":
        enter = decide_enter_house_failure(
            pos=pos,
            front=front,
            enter_retries=enter_retries,
            enter_retry_limit=enter_retry_limit,
            reason=reason,
        )
        if enter is not None:
            return enter
        return RecoverDecision(kind=RecoverKind.HARD_FAIL, reason=reason)
    if phase in {"nav_house_front", "nav_drop_spot"}:
        return decide_nav_failure(
            phase=phase,
            pos=pos,
            front=front,
            hands_clear=hands_clear,
            drop_attempts=drop_attempts,
            drop_attempt_limit=drop_attempt_limit,
            offstand_corrections=offstand_corrections,
            south_escape_attempts=south_escape_attempts,
            south_escape_limit=south_escape_limit,
            reason=reason,
        )
    return RecoverDecision(kind=RecoverKind.HARD_FAIL, reason=reason)


def _charge_mix(direction: str, frames: int, *, a_every: int) -> List[np.ndarray]:
    actions: List[np.ndarray] = []
    for i in range(frames):
        if i % a_every == 0:
            actions.append(make_action(**{direction: True, "a": True}))
        else:
            actions.append(make_action(**{direction: True, "b": True}))
    return actions


def drop_carried_actions(attempt: int) -> List[np.ndarray]:
    """Toss held debris so building doors accept entry."""
    if attempt <= 1:
        return list(toss_held_actions(face="down", step_away=True)) + list(
            multi_face_toss_actions(prefer_south=True)
        )
    if attempt <= 3:
        return list(multi_face_toss_actions(prefer_south=True))
    actions: List[np.ndarray] = []
    for face in ("down", "left", "right"):
        actions.extend(toss_held_actions(face=face, step_away=True))
    return actions


def south_escape_actions(
    *, long_east: bool = False, far_east: bool = False
) -> List[np.ndarray]:
    """Leave south-of-fence softlock for re-nav.

    Far-east pond latitude: west toward free lane then north. SW pocket:
    north-first then east. Mid-wall: east first. Mix A so a blocking weed
    can be lifted.
    """
    if far_east:
        actions = (
            _charge_mix("left", 90, a_every=20)
            + _charge_mix("up", 80, a_every=20)
            + _charge_mix("left", 70, a_every=20)
            + _charge_mix("up", 90, a_every=20)
            + _charge_mix("left", 40, a_every=20)
            + _charge_mix("up", 60, a_every=20)
        )
    elif long_east:
        actions = (
            _charge_mix("up", 80, a_every=20)
            + _charge_mix("right", 70, a_every=20)
            + _charge_mix("up", 90, a_every=20)
            + _charge_mix("right", 60, a_every=20)
            + _charge_mix("up", 80, a_every=20)
            + _charge_mix("left", 20, a_every=20)
        )
    else:
        actions = (
            _charge_mix("right", 70, a_every=20)
            + _charge_mix("up", 70, a_every=20)
            + _charge_mix("right", 50, a_every=20)
            + _charge_mix("up", 80, a_every=20)
            + _charge_mix("left", 16, a_every=20)
        )
    actions.extend(make_action() for _ in range(8))
    return actions


def short_east_north_actions() -> List[np.ndarray]:
    """Compact east→north charge when outer timeout is almost gone."""
    actions = _charge_mix("right", 50, a_every=16) + _charge_mix(
        "up", 70, a_every=16
    )
    actions.extend(make_action() for _ in range(6))
    return actions
