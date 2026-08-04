"""Room-agnostic pose sets, xy bands, true-ground, and climb track.

Skills and product hops pass ``room_id`` / pose sets from policy — this module
never hardcodes Bubble Mountain constants.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from super_metroid.ram import SuperMetroidState
from super_metroid.routes.controller_common import hold
from super_metroid.routes.runtime import ControllerSession

# Shared pose families (room-agnostic defaults; policies may override).
TRUE_GROUND = frozenset({1, 2, 9, 10})
STAND_PIN = frozenset({1, 2, 9, 10, 25, 26, 27, 28})
STANDING_POSES = frozenset({1, 2, 9, 10, 25, 26, 27, 28, 37, 38, 137, 138})
POSE_KNOCKBACK = frozenset({137, 138})
POSE_STAND_LEFT = frozenset({2, 10})
POSE_STAND_RIGHT = frozenset({1, 9})


class GeometryPolicy(Protocol):
    """Minimal policy surface for climb tracking / wrong-door steer."""

    ROOM_ID: int
    TRUE_GROUND: frozenset[int]
    STAND_PIN: frozenset[int]
    MID_Y: int
    MID_STAND_X: tuple[int, int]
    LIP_X: tuple[int, int]
    LIP_Y: tuple[int, int]
    RIGHT_SHELF_X: int
    RIGHT_SHELF_Y: int
    CAVITY_X_MAX: int
    SAVE_RUNWAY_X: tuple[int, int]
    SAVE_RUNWAY_Y: tuple[int, int]
    SAVE_RUNWAY_FIRE_X: tuple[int, int]
    PHASE_C_X_MIN: int
    PHASE_C_Y_MIN: int
    PHASE_C_Y_MAX: int
    PHASE_D_X: int
    PHASE_D_Y: int


# ---------------------------------------------------------------------------
# Pure geometry predicates
# ---------------------------------------------------------------------------


def is_true_ground(
    st: SuperMetroidState,
    *,
    poses: frozenset[int] = TRUE_GROUND,
    max_vy: int = 1,
) -> bool:
    """True solid ground (charge/reseat/land). Spin apex pose 25 is NOT ground."""
    return abs(int(st.velocity_y)) <= max_vy and int(st.pose) in poses


def is_stand_pin_pose(
    st: SuperMetroidState,
    *,
    poses: frozenset[int] = STAND_PIN,
    max_vy: int = 1,
) -> bool:
    """Stand-pin pose class at rest (includes 25/26).

    Use for lip *detection* and approach walk. Do **not** use for mid-air
    charge/reseat (pose 25 + vy≈0 is spin apex, not a land).
    """
    return abs(int(st.velocity_y)) <= max_vy and int(st.pose) in poses


def in_xy_band(
    st: SuperMetroidState,
    *,
    room_id: int | None = None,
    x_min: int | None = None,
    x_max: int | None = None,
    y_min: int | None = None,
    y_max: int | None = None,
) -> bool:
    """Optional room + axis-aligned band check."""
    if room_id is not None and int(st.room_id) != room_id:
        return False
    x, y = int(st.samus_x), int(st.samus_y)
    if x_min is not None and x < x_min:
        return False
    if x_max is not None and x > x_max:
        return False
    if y_min is not None and y < y_min:
        return False
    if y_max is not None and y > y_max:
        return False
    return True


def phase_c_usable_right_contact(
    st: SuperMetroidState, policy: GeometryPolicy
) -> bool:
    """Usable right-structure contact at height (not floor thrash)."""
    return (
        int(st.room_id) == policy.ROOM_ID
        and policy.PHASE_C_X_MIN <= int(st.samus_x) <= policy.CAVITY_X_MAX
        and policy.PHASE_C_Y_MIN <= int(st.samus_y) <= policy.PHASE_C_Y_MAX
    )


def phase_d_top_band(st: SuperMetroidState, policy: GeometryPolicy) -> bool:
    """Top band before Super door."""
    return (
        int(st.room_id) == policy.ROOM_ID
        and int(st.samus_y) <= policy.PHASE_D_Y
        and int(st.samus_x) >= policy.PHASE_D_X
    )


def phase_d_near_top(
    st: SuperMetroidState, policy: GeometryPolicy, slack: int = 40
) -> bool:
    """Looser top band for door-start / near-top handoff (slack px)."""
    if phase_d_top_band(st, policy):
        return True
    return (
        int(st.room_id) == policy.ROOM_ID
        and int(st.samus_y) <= policy.PHASE_D_Y + slack
        and int(st.samus_x) >= policy.PHASE_D_X - slack
    )


def on_mid_iso_pin(st: SuperMetroidState, policy: GeometryPolicy) -> bool:
    """Mid-iso pin class (stand_pin poses + velocity + xy band)."""
    stand_lo, stand_hi = policy.MID_STAND_X
    return (
        abs(int(st.velocity_y)) <= 2
        and int(st.pose) in policy.STAND_PIN
        and stand_lo <= int(st.samus_x) <= stand_hi
        and int(st.samus_y) <= policy.MID_Y + 10
    )


def on_launch_lip(st: SuperMetroidState, policy: GeometryPolicy) -> bool:
    """Solid save-door lip seat (stand-pin poses)."""
    lip_lo, lip_hi = policy.LIP_X
    lip_y_lo, lip_y_hi = policy.LIP_Y
    return (
        is_stand_pin_pose(st, poses=policy.STAND_PIN)
        and lip_lo <= st.samus_x <= lip_hi
        and lip_y_lo <= st.samus_y <= lip_y_hi
    )


def on_right_shelf(st: SuperMetroidState, policy: GeometryPolicy) -> bool:
    """Right-structure shelf class (stand-pin)."""
    return (
        is_stand_pin_pose(st, poses=policy.STAND_PIN)
        and st.samus_x >= policy.RIGHT_SHELF_X
        and st.samus_x <= policy.CAVITY_X_MAX
        and st.samus_y <= policy.RIGHT_SHELF_Y
        and st.samus_y >= 200
    )


def on_save_runway(st: SuperMetroidState, policy: GeometryPolicy) -> bool:
    """Save-door outer runway for left climb (grounded stand-pin band)."""
    x_lo, x_hi = policy.SAVE_RUNWAY_X
    y_lo, y_hi = policy.SAVE_RUNWAY_Y
    return (
        is_stand_pin_pose(st, poses=policy.STAND_PIN)
        and x_lo <= int(st.samus_x) <= x_hi
        and y_lo <= int(st.samus_y) <= y_hi
    )


# ---------------------------------------------------------------------------
# Climb track (metrics + optional recon dump; no mid-loop control)
# ---------------------------------------------------------------------------


class PhaseStop(Exception):
    """Diagnostic early exit when a pure probe stops at a phase pin.

    Probe CLI may treat this as success for capture/recon only — never as
    hop GREEN / continuous evidence.
    """

    def __init__(
        self,
        phase: str,
        state: SuperMetroidState,
        *,
        metrics: dict[str, Any] | None = None,
        label: str = "phase_stop",
    ) -> None:
        self.phase = phase
        self.state = state
        self.metrics = dict(metrics or {})
        super().__init__(
            f"{label}:{phase} room=0x{state.room_id:04X} "
            f"pose={state.pose} xy=({state.samus_x},{state.samus_y}) "
            f"vx={state.velocity_x} vy={state.velocity_y}"
        )


# Historical name used by probes / k4 re-exports (message prefix preserved).
class BubblePhaseStop(PhaseStop):
    """Bubble hop PhaseStop with legacy message prefix."""

    def __init__(
        self,
        phase: str,
        state: SuperMetroidState,
        *,
        metrics: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(
            phase, state, metrics=metrics, label="bubble_phase_stop"
        )


@dataclass
class ClimbTrack:
    """Climb metrics and optional Phase-C recon flags.

    Mid-loop control state (phase / frame budget / height class) lives inside
    the product mid loop as locals — not on this track.
    """

    label: str = "climb"
    max_x: int = 0
    min_y: int = 0
    mid_reached: bool = False
    top_reached: bool = False
    door_reached: bool = False
    standing_mid_pinned: bool = False
    launched: bool = False
    phase_c_hit: bool = False
    # Recon-only; product path leaves defaults.
    dump_path: Path | None = None
    phase_c_dumped: bool = False
    stop_at_phase_c: bool = False

    def metrics_dict(self) -> dict[str, Any]:
        return {
            "max_x": self.max_x,
            "min_y": self.min_y,
            "mid_reached": self.mid_reached,
            "top_reached": self.top_reached,
            "phase_c_hit": self.phase_c_hit,
            "dump_phase_c": str(self.dump_path) if self.dump_path else None,
            "dumped": self.phase_c_dumped,
        }


# Alias kept for product hop / tests that still say BubbleTrack.
BubbleTrack = ClimbTrack


def _maybe_dump_phase_c(
    session: ControllerSession,
    state: SuperMetroidState,
    track: ClimbTrack,
    *,
    is_phase_c: Callable[[SuperMetroidState], bool],
) -> None:
    """Save first Phase-C pin when probe session exposes ``env``."""
    if track.dump_path is None or track.phase_c_dumped:
        return
    if not is_phase_c(state):
        return
    env = getattr(session, "env", None)
    if env is None:
        return
    from super_metroid.dev.common import save_dev_state

    save_dev_state(env, track.dump_path)
    track.phase_c_dumped = True


def track_state(
    session: ControllerSession,
    track: ClimbTrack,
    state: SuperMetroidState,
    policy: GeometryPolicy,
    *,
    phase_stop_cls: type[PhaseStop] = BubblePhaseStop,
) -> None:
    """Update climb metrics; optional Phase-C dump/stop for recon tracks."""
    track.max_x = max(track.max_x, state.samus_x)
    track.min_y = min(track.min_y, state.samus_y)
    if state.samus_y <= policy.MID_Y and state.samus_x >= 90:
        track.mid_reached = True
    if phase_d_top_band(state, policy):
        track.top_reached = True
    if phase_c_usable_right_contact(state, policy):
        if not track.phase_c_hit:
            track.phase_c_hit = True
            _maybe_dump_phase_c(
                session,
                state,
                track,
                is_phase_c=lambda s: phase_c_usable_right_contact(s, policy),
            )
            if track.stop_at_phase_c:
                raise phase_stop_cls(
                    "C", state, metrics=track.metrics_dict()
                )


def avoid_wrong_door(
    session: ControllerSession,
    track: ClimbTrack,
    state: SuperMetroidState,
    policy: GeometryPolicy,
) -> bool:
    """Hard-steer away from side doors; return True if a steer was applied."""
    label = track.label
    x, y = state.samus_x, state.samus_y
    # Left doors: Rising Tide ~y624, Save ~y368, Missiles Super ~y112.
    # Fire window needs x down to ~25 at y~395. Only force RIGHT when deeper
    # than the fire band or not at save height.
    y_lo, y_hi = policy.SAVE_RUNWAY_Y
    fire_lo, _fire_hi = policy.SAVE_RUNWAY_FIRE_X
    on_save_platform = y_lo <= y <= y_hi
    if x < 22 or (x < 55 and not on_save_platform):
        hold(session, 1, "RIGHT", "B", reason=f"{label}_avoid_left")
        return True
    if on_save_platform and x < fire_lo - 1:
        # Soft nudge off the door shell (x≲24) without abandoning fire seat.
        hold(session, 1, "RIGHT", "B", reason=f"{label}_avoid_save_door")
        return True
    # Right mid Single Chamber ~y368 / x≈496.
    if x > 470 and 300 <= y <= 430:
        hold(session, 1, "LEFT", "B", reason=f"{label}_avoid_sc")
        return True
    return False


def new_climb_track(
    session: ControllerSession,
    *,
    label: str = "climb",
    dump_phase_c: Path | str | None = None,
    stop_at_phase_c: bool = False,
) -> ClimbTrack:
    """Fresh climb track seeded from current session position."""
    st = session.state
    return ClimbTrack(
        label=label,
        max_x=st.samus_x,
        min_y=st.samus_y,
        dump_path=Path(dump_phase_c) if dump_phase_c is not None else None,
        stop_at_phase_c=stop_at_phase_c,
    )


__all__ = [
    "TRUE_GROUND",
    "STAND_PIN",
    "STANDING_POSES",
    "POSE_KNOCKBACK",
    "POSE_STAND_LEFT",
    "POSE_STAND_RIGHT",
    "BubblePhaseStop",
    "BubbleTrack",
    "ClimbTrack",
    "PhaseStop",
    "avoid_wrong_door",
    "in_xy_band",
    "is_stand_pin_pose",
    "is_true_ground",
    "new_climb_track",
    "on_launch_lip",
    "on_mid_iso_pin",
    "on_right_shelf",
    "on_save_runway",
    "phase_c_usable_right_contact",
    "phase_d_near_top",
    "phase_d_top_band",
    "track_state",
]
