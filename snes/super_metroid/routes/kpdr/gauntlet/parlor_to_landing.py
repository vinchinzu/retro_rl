"""Parlor Flyway door (post-BT) → Landing Site bottom-left cave.

Reuses the product Alcatraz chimney + upper-platform open-loop (same as
``play_parlor_to_main_shaft``) through the top bomb tunnel, then runs **right**
through the blue door into Landing instead of left to Terminator.
"""

from __future__ import annotations

from super_metroid.routes.controller_common import (
    WallJumpTiming,
    consecutive_walljumps,
    hold,
    require_room,
    settle_hold,
    unmorph,
    wait_ordinary_room,
)
from super_metroid.routes.kpdr.room_ids import ROOM_LANDING_SITE, ROOM_PARLOR
from super_metroid.routes.runtime import ControllerSession
from super_metroid.routes.skills.geometry import PhaseStop

# Legacy right-side Parlor chimney (Gauntlet side quest). Product Spore
# spine uses alcatraz_escape, not these timings.
_PARLOR_CHIMNEY_RIGHT = WallJumpTiming(
    into="RIGHT",
    flip="RIGHT",
    into_frames=30,
    amid_frames=0,
    flip_frames=0,
    delay_into_frames=0,
)
_PARLOR_CHIMNEY_LEFT = WallJumpTiming(
    into="LEFT",
    flip="LEFT",
    into_frames=30,
    amid_frames=0,
    flip_frames=0,
    delay_into_frames=0,
)
_PARLOR_CHIMNEY_WJ: tuple[WallJumpTiming, ...] = (
    *(_PARLOR_CHIMNEY_RIGHT for _ in range(6)),
    *(_PARLOR_CHIMNEY_LEFT for _ in range(2)),
)
_PARLOR_CHIMNEY_GAP = 12

_UPPER_PLATFORMS: tuple[tuple[tuple[str, ...], int], ...] = (
    (("LEFT", "A"), 30),
    (("LEFT",), 30),
    ((), 60),
    (("LEFT", "A"), 40),
    ((), 100),
    (("RIGHT", "A"), 30),
    (("RIGHT",), 15),
    (("LEFT",), 10),
    ((), 100),
    (("RIGHT", "A"), 10),
    (("RIGHT",), 40),
    (("LEFT",), 10),
    ((), 100),
    (("RIGHT", "A"), 20),
    (("RIGHT",), 30),
    ((), 100),
    (("LEFT", "A"), 40),
    (("LEFT",), 16),
    ((), 30),
    (("RIGHT", "B"), 21),
    (("RIGHT", "A", "B"), 8),
    (("LEFT",), 8),
    (("LEFT", "A"), 50),
    ((), 40),
    (("RIGHT", "A"), 35),
    ((), 100),
)


def _align_flyway(session: ControllerSession) -> None:
    require_room(session, ROOM_PARLOR, "parlor_to_landing")
    if session.state.samus_x > 956:
        hold(
            session,
            10 if session.state.pose == 2 else 15,
            "LEFT",
            reason="parlor_flyway_align",
        )
        hold(session, 10, reason="parlor_flyway_align")


def climb_parlor_to_top_tunnel(session: ControllerSession) -> None:
    """Alcatraz WJ + upper platforms + morph bomb tunnel to the top-right door."""
    for _ in range(2):
        hold(session, 20, "LEFT", "A", "B", "X", reason="parlor_left_traverse")
        hold(session, 12, "LEFT", "B", "X", reason="parlor_left_traverse")
    hold(session, 50, reason="parlor_left_traverse_settle")
    consecutive_walljumps(
        session,
        _PARLOR_CHIMNEY_WJ,
        reason="parlor_chimney_wj",
        gap_frames=_PARLOR_CHIMNEY_GAP,
        stop_when=lambda s: int(s.room_id) != ROOM_PARLOR,
    )
    settle_hold(session, _PARLOR_CHIMNEY_GAP, reason="parlor_chimney_wj_tail")
    hold(session, 100, reason="parlor_chimney_settle")
    for names, frames in _UPPER_PLATFORMS:
        hold(session, frames, *names, reason="parlor_upper_platforms")
        if session.state.room_id != ROOM_PARLOR:
            return
    hold(session, 2, "DOWN", reason="parlor_bomb_tunnel_morph")
    hold(session, 3, reason="parlor_bomb_tunnel_morph")
    hold(session, 2, "DOWN", reason="parlor_bomb_tunnel_morph")
    hold(session, 10, reason="parlor_bomb_tunnel_morph")
    for _ in range(10):
        hold(session, 45, "RIGHT", "X", reason="parlor_bomb_tunnel")
        hold(session, 15, "RIGHT", reason="parlor_bomb_tunnel")
        if session.state.room_id != ROOM_PARLOR:
            return
    hold(session, 40, reason="parlor_bomb_tunnel_settle")


def _exit_to_landing(session: ControllerSession) -> None:
    hold(session, 80, "RIGHT", "B", reason="parlor_landing_door")
    if session.state.room_id == ROOM_PARLOR:
        hold(session, 40, "RIGHT", "A", "B", reason="parlor_landing_door_spin")
        hold(session, 80, "RIGHT", "B", reason="parlor_landing_door_run")
    wait_ordinary_room(
        session,
        ROOM_LANDING_SITE,
        settle_frames=280,
        label="parlor_to_landing",
    )
    unmorph(session)


def play_parlor_to_landing(
    session: ControllerSession,
    *,
    stop_at: str | None = None,
) -> None:
    """Flyway door → Landing Site bottom-left cave (node 2)."""
    _align_flyway(session)
    if stop_at == "flyway":
        raise PhaseStop("flyway", session.state, label="parlor_to_landing")
    climb_parlor_to_top_tunnel(session)
    if stop_at == "parlor_top":
        raise PhaseStop("parlor_top", session.state, label="parlor_to_landing")
    _exit_to_landing(session)


__all__ = [
    "climb_parlor_to_top_tunnel",
    "play_parlor_to_landing",
]
