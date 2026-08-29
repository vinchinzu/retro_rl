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
_ROLLOUT_MORPH_POSES = _PROBE_MORPH_POSES

# World pins from live parlor_chimney_double recon + Alcatraz-escape refs.
SHAFT_LIP_Y = 210
ROLLOUT_MAX_X = 760
ROLLOUT_MAX_Y = 230
_LEFT_WALL_BASE = ((795, 820), (535, 550))
_MID_LEDGE = ((820, 838), (450, 470))
_LOWER_WJ_BAND = ((795, 815), (350, 375))


def at_left_wall_base(state) -> bool:
    x_range, y_range = _LEFT_WALL_BASE
    return (
        int(state.room_id) == ROOM_PARLOR
        and x_range[0] <= int(state.samus_x) <= x_range[1]
        and y_range[0] <= int(state.samus_y) <= y_range[1]
        and int(state.pose) in _GROUNDED_POSES
    )


def at_mid_ledge(state) -> bool:
    x_range, y_range = _MID_LEDGE
    return (
        int(state.room_id) == ROOM_PARLOR
        and x_range[0] <= int(state.samus_x) <= x_range[1]
        and y_range[0] <= int(state.samus_y) <= y_range[1]
        and int(state.pose) in _GROUNDED_POSES
    )


def at_shaft_lip(state) -> bool:
    """Air/latch in the morph-hole band. y<=210 is the lip class."""
    return (
        int(state.room_id) == ROOM_PARLOR
        and int(state.samus_y) <= SHAFT_LIP_Y
        and 790 <= int(state.samus_x) <= 870
    )


def at_alcatraz_rollout(state) -> bool:
    """Morph ball left of the chimney opening, still in Parlor."""
    return (
        int(state.room_id) == ROOM_PARLOR
        and int(state.samus_x) <= ROLLOUT_MAX_X
        and int(state.samus_y) <= ROLLOUT_MAX_Y
        and int(state.pose) in _ROLLOUT_MORPH_POSES
    )


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
    """One dash-jump up the door slope onto the left-wall platform."""
    hold(session, 2, "LEFT", reason="alcatraz_base_face")
    hold(session, 30, "LEFT", "B", reason="alcatraz_base_run")
    hold(session, 18, "LEFT", "A", reason="alcatraz_base_hop")
    hold(session, 16, reason="alcatraz_base_land")
    _unmorph_probe_pose(session)
    _require_geometry(
        session,
        "left-wall base",
        x_range=_LEFT_WALL_BASE[0],
        y_range=_LEFT_WALL_BASE[1],
    )
    return session.frame


def _reach_mid_ledge(session: ControllerSession) -> int:
    hold(session, 2, "RIGHT", reason="alcatraz_ledge_face")
    for _ in range(3):
        hold(session, 40, "RIGHT", "A", reason="alcatraz_ledge_cross")
        hold(session, 2, "LEFT", reason="alcatraz_ledge_turn")
        hold(session, 28, "LEFT", "A", reason="alcatraz_ledge_latch")
        if at_mid_ledge(session.state) or (
            session.state.samus_y <= 470
            and session.state.pose in _GROUNDED_POSES
        ):
            break
    hold(session, 16, reason="alcatraz_ledge_settle")
    _unmorph_probe_pose(session)
    hold(session, 8, reason="alcatraz_ledge_stand")
    _require_geometry(
        session,
        "mid ledge",
        x_range=_MID_LEDGE[0],
        y_range=_MID_LEDGE[1],
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
        x_range=_LOWER_WJ_BAND[0],
        y_range=_LOWER_WJ_BAND[1],
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

    morph_frame = _instant_morph_rollout(session)
    return walljump_frame, morph_frame


def _instant_morph_rollout(session: ControllerSession) -> int:
    """WJ mockball: keep A, tap DOWN once, then roll LEFT through the hole."""
    hold(session, 1, "DOWN", "A", reason="alcatraz_instant_morph")
    morph_frame = session.frame
    for _ in range(80):
        hold(session, 1, "LEFT", reason="alcatraz_escape")
        if at_alcatraz_rollout(session.state):
            return morph_frame
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
    "ROLLOUT_MAX_X",
    "ROLLOUT_MAX_Y",
    "SHAFT_LIP_Y",
    "WallJumpPulse",
    "at_alcatraz_rollout",
    "at_left_wall_base",
    "at_mid_ledge",
    "at_shaft_lip",
    "play_alcatraz_escape",
]
