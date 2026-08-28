"""Goal labels and Join checks for the generalist contractor."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from super_metroid.hop_glance import final_from_state, grade_final, pose_class
from super_metroid.leave_specs import LeaveSpec
from super_metroid.practice_repertoire.catalog import (
    PRODUCT_CATEGORY,
    RepertoireSession,
    get_session,
    neighbors,
)

JOIN_XY_BAND = 64
GOAL_VEC_DIM = 12


@dataclass(frozen=True)
class Goal:
    """Where the contractor should Join. Not a Skill."""

    session_id: str
    room_id: int
    x: int
    y: int
    pose: int | None = None
    any_door: bool = False
    node_type: int = 0  # 0 door, 1 item, 2 other
    start_room_id: int | None = None

    @property
    def resolved(self) -> bool:
        return self.room_id > 0


def parse_goal(text: str) -> Goal:
    """CLI Goal: ``session:kpdr25/crateria/morph``, a bare session id, or ``any``."""

    raw = (text or "").strip()
    if not raw or raw in {"any", "any-door", "any_door"}:
        return Goal(session_id="", room_id=0, x=0, y=0, any_door=True)
    if raw.startswith("session:"):
        return goal_from_session(raw.split(":", 1)[1])
    if raw.startswith("door:"):
        raise ValueError(
            f"unrecognized goal {raw!r}: door sides are not implemented"
        )
    if "/" in raw:
        return goal_from_session(raw)
    raise ValueError(f"unrecognized goal {raw!r}")


def goal_from_session(
    session_id: str,
    *,
    next_session: RepertoireSession | None = None,
) -> Goal:
    """Default autopilot Goal: next repertoire session pin."""

    current = get_session(session_id)
    nxt = next_session
    if nxt is None:
        _prev, nxt = neighbors(session_id, category=current.category)
    if nxt is None:
        return Goal(
            session_id=current.id,
            room_id=int(current.room_id or 0),
            x=int(current.x or 0),
            y=int(current.y or 0),
            pose=current.pose,
            start_room_id=current.room_id,
        )
    return Goal(
        session_id=nxt.id,
        room_id=int(nxt.room_id or 0),
        x=int(nxt.x or 0),
        y=int(nxt.y or 0),
        pose=nxt.pose,
        start_room_id=current.room_id,
    )


def leave_spec_for(goal: Goal) -> LeaveSpec:
    """Join is hop_glance against this spec, not room-id alone."""

    klass = "any"
    if goal.pose is not None:
        classified = pose_class(int(goal.pose))
        if classified == "morph":
            klass = "morph"
        elif classified in {"stand", "air"}:
            klass = "door"
    return LeaveSpec(
        hop=goal.session_id or "generalist",
        room=int(goal.room_id),
        x=(int(goal.x) - JOIN_XY_BAND, int(goal.x) + JOIN_XY_BAND),
        y=(int(goal.y) - JOIN_XY_BAND, int(goal.y) + JOIN_XY_BAND),
        pose_class=klass,
        gs=8,
        dt=0,
    )


def is_join(state: Any, goal: Goal) -> bool:
    """True when the still would glance-pass the Goal LeaveSpec."""

    if not goal.resolved:
        return False
    misses = grade_final(final_from_state(state), leave_spec_for(goal))
    return not misses


def goal_vector(
    state: Any,
    goal: Goal,
    *,
    prev_action: int = 0,
    steer_x: int | None = None,
    steer_y: int | None = None,
) -> list[float]:
    """12-float Goal tail. Locked by the observation contract.

    ``steer_x``/``steer_y`` change only the relative dx/dy (in-room Join xy,
    first bounded Goal-route door, or a nearest-door fallback). Absolute Goal
    slots stay the Join pin.
    """

    sx = int(getattr(state, "samus_x", getattr(state, "x", 0)) or 0)
    sy = int(getattr(state, "samus_y", getattr(state, "y", 0)) or 0)
    room = int(getattr(state, "room_id", getattr(state, "room", 0)) or 0)
    tx = int(goal.x if steer_x is None else steer_x)
    ty = int(goal.y if steer_y is None else steer_y)
    dx = float(tx - sx)
    dy = float(ty - sy)
    dist = (dx * dx + dy * dy) ** 0.5
    return [
        max(-4.0, min(4.0, dx / 512.0)),
        max(-4.0, min(4.0, dy / 512.0)),
        float(goal.node_type),
        float(goal.room_id) / 65535.0,
        1.0 if goal.any_door else 0.0,
        float(prev_action) / 26.0,
        float(goal.start_room_id or 0) / 65535.0,
        min(4.0, dist / 1024.0),
        1.0 if room == goal.room_id else 0.0,
        float(goal.x) / 4096.0,
        float(goal.y) / 4096.0,
        1.0 if goal.resolved else 0.0,
    ]


def default_category() -> str:
    return PRODUCT_CATEGORY


__all__ = [
    "GOAL_VEC_DIM",
    "JOIN_XY_BAND",
    "Goal",
    "default_category",
    "goal_from_session",
    "goal_vector",
    "is_join",
    "leave_spec_for",
    "parse_goal",
]
