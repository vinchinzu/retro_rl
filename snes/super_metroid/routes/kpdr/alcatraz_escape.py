"""Morph-only Alcatraz escape after Bomb Torizo.

This is the short left-chimney route from the natural post-Torizo Parlor
seat. It uses ordinary controller inputs and live pose/geometry gates; it
does not load state, write RAM, or require Bombs after entry.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

from super_metroid.ram import GameplayPhase
from super_metroid.routes.controller_common import MORPH_POSES, POSE_WALL_LATCH
from super_metroid.routes.kpdr.room_ids import ROOM_PARLOR
from super_metroid.routes.runtime import ControllerSession, hold

_GROUNDED_POSES = frozenset({1, 2, 5, 6, 7, 8, 9, 10})
_PROBE_MORPH_POSES = MORPH_POSES | frozenset({165, 166, 167})


@dataclass(frozen=True)
class WallJumpPulse:
    """Turn away from a wall without jump, then jump while holding away."""

    away: str
    turn_frames: int
    jump_frames: int


@dataclass(frozen=True)
class AlcatrazEscapeEvidence:
    """Geometry proof returned by :func:`play_alcatraz_escape`."""

    entry_frame: int
    base_frame: int
    ledge_frame: int
    walljump_frame: int
    morph_frame: int
    exit_frame: int
    exit_x: int
    exit_y: int

    def to_dict(self) -> dict[str, int]:
        return asdict(self)


# Release-before-jump is the essential walljump mechanic. These timings were
# verified twice from the natural post-Torizo continuous pin.
_LOWER_WALLJUMPS = (
    WallJumpPulse("RIGHT", turn_frames=2, jump_frames=40),
    WallJumpPulse("LEFT", turn_frames=2, jump_frames=30),
)
_UPPER_WALLJUMP = WallJumpPulse("RIGHT", turn_frames=2, jump_frames=34)
_FINAL_TURN_FRAMES = 5


def _play_walljump_pulse(
    session: ControllerSession,
    pulse: WallJumpPulse,
    *,
    reason: str,
) -> None:
    """Emit one real walljump: away/release-A, then away+A."""
    hold(
        session,
        pulse.turn_frames,
        pulse.away,
        reason=f"{reason}_turn",
    )
    hold(
        session,
        pulse.jump_frames,
        pulse.away,
        "A",
        reason=f"{reason}_jump",
    )


def _require_geometry(
    session: ControllerSession,
    label: str,
    *,
    x_range: tuple[int, int],
    y_range: tuple[int, int],
) -> None:
    state = session.state
    if not (
        state.room_id == ROOM_PARLOR
        and x_range[0] <= state.samus_x <= x_range[1]
        and y_range[0] <= state.samus_y <= y_range[1]
    ):
        raise RuntimeError(
            f"Alcatraz {label} missed: room=0x{state.room_id:04X} "
            f"xy=({state.samus_x},{state.samus_y}) pose={state.pose} "
            f"frame={session.frame}"
        )


def _unmorph_probe_pose(session: ControllerSession) -> None:
    if session.state.pose not in _PROBE_MORPH_POSES:
        return
    hold(session, 8, "UP", reason="alcatraz_base_unmorph")
    hold(session, 6, reason="alcatraz_base_unmorph_settle")


def _land_left_wall_base(session: ControllerSession) -> int:
    hold(session, 4, "LEFT", reason="alcatraz_base_face")
    for frame in range(18):
        run_button = "Y" if frame % 6 == 0 else "B"
        hold(session, 1, "LEFT", run_button, reason="alcatraz_base_run")

    for jump_frames in (10, 12, 16):
        hold(session, 2, "LEFT", reason="alcatraz_base_turn")
        hold(
            session,
            jump_frames,
            "LEFT",
            "A",
            reason="alcatraz_base_hop",
        )
        hold(session, 18, reason="alcatraz_base_land")
        _unmorph_probe_pose(session)
        state = session.state
        if (
            state.samus_y <= 550
            and state.samus_x <= 820
            and state.pose in _GROUNDED_POSES
        ):
            break

    _require_geometry(
        session,
        "left-wall base",
        x_range=(795, 820),
        y_range=(535, 550),
    )
    return session.frame


def _reach_mid_ledge(session: ControllerSession) -> int:
    hold(session, 2, "RIGHT", reason="alcatraz_ledge_face")
    for _ in range(3):
        hold(session, 40, "RIGHT", "A", reason="alcatraz_ledge_cross")
        hold(session, 2, "LEFT", reason="alcatraz_ledge_turn")
        hold(session, 28, "LEFT", "A", reason="alcatraz_ledge_latch")
        if (
            session.state.samus_y <= 470
            and session.state.pose in _GROUNDED_POSES
        ):
            break
    hold(session, 25, reason="alcatraz_ledge_settle")
    _require_geometry(
        session,
        "mid ledge",
        x_range=(820, 838),
        y_range=(450, 470),
    )
    return session.frame


def _climb_chimney(session: ControllerSession) -> tuple[int, int]:
    hold(session, 3, "LEFT", reason="alcatraz_chimney_face")
    hold(
        session,
        22,
        "LEFT",
        "B",
        "A",
        reason="alcatraz_chimney_left_wall",
    )
    hold(session, 4, "LEFT", reason="alcatraz_chimney_contact")

    for index, pulse in enumerate(_LOWER_WALLJUMPS, start=1):
        _play_walljump_pulse(
            session,
            pulse,
            reason=f"alcatraz_walljump_{index}",
        )
    _require_geometry(
        session,
        "lower walljumps",
        x_range=(795, 815),
        y_range=(350, 375),
    )

    _play_walljump_pulse(
        session,
        _UPPER_WALLJUMP,
        reason="alcatraz_walljump_3",
    )
    hold(
        session,
        _FINAL_TURN_FRAMES,
        "LEFT",
        reason="alcatraz_walljump_4_turn",
    )
    for _ in range(90):
        hold(
            session,
            1,
            "LEFT",
            "A",
            reason="alcatraz_walljump_4_jump",
        )
        if session.state.samus_y <= 225 and session.state.pose == POSE_WALL_LATCH:
            walljump_frame = session.frame
            break
    else:
        raise RuntimeError(f"Alcatraz final walljump missed: {session.state}")

    # Holding jump after a walljump makes DOWN an instant aerial Morph. LEFT
    # then carries the ball through the Alcatraz opening.
    hold(session, 1, "DOWN", "A", reason="alcatraz_instant_morph")
    morph_frame = session.frame
    for _ in range(80):
        hold(session, 1, "LEFT", reason="alcatraz_escape")
        state = session.state
        if (
            state.samus_x <= 760
            and state.samus_y <= 230
            and state.pose in MORPH_POSES
        ):
            return walljump_frame, morph_frame
    raise RuntimeError(f"Alcatraz Morph opening missed: {session.state}")


def play_alcatraz_escape(session: ControllerSession) -> AlcatrazEscapeEvidence:
    """Escape Alcatraz from the natural post-Torizo Parlor entry seat."""
    state = session.state
    if not (
        state.room_id == ROOM_PARLOR
        and state.phase is GameplayPhase.ORDINARY_GAMEPLAY
        and state.game_state == 8
        and state.door_transition == 0
        and state.morph_ball
        and state.samus_x == 968
        and state.samus_y == 651
        and state.pose == 2
    ):
        raise RuntimeError(f"Alcatraz natural entry mismatch: {state}")

    entry_frame = session.frame
    base_frame = _land_left_wall_base(session)
    ledge_frame = _reach_mid_ledge(session)
    walljump_frame, morph_frame = _climb_chimney(session)
    state = session.state
    return AlcatrazEscapeEvidence(
        entry_frame=entry_frame,
        base_frame=base_frame,
        ledge_frame=ledge_frame,
        walljump_frame=walljump_frame,
        morph_frame=morph_frame,
        exit_frame=session.frame,
        exit_x=state.samus_x,
        exit_y=state.samus_y,
    )


__all__ = [
    "AlcatrazEscapeEvidence",
    "WallJumpPulse",
    "play_alcatraz_escape",
]
