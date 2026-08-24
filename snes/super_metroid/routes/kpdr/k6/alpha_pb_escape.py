"""Reactive Alpha Power Bomb room escape to Caterpillar."""

from __future__ import annotations

from super_metroid.ram import SuperMetroidState
from super_metroid.routes.controller_common import hold, require_room, select_weapon, wait_ordinary_room
from super_metroid.routes.runtime import ControllerSession

ROOM_ALPHA_PB = 0xA3AE
ROOM_CATERPILLAR = 0xA322

_ESCAPE_BUDGET = 2200
_PROGRESS_WINDOW = 42


def _clear_obstacle(session: ControllerSession, *, label: str) -> None:
    """Jump and multi-shot diagonally through a stalled obstacle/enemy."""
    for _ in range(18):
        hold(session, 1, "A", reason=f"{label}_jump")
    for frame in range(34):
        buttons = ["R"]
        if frame % 3 == 0:
            buttons.append("X")
        hold(session, 1, *buttons, reason=f"{label}_aim_shoot")
    hold(session, 10, reason=f"{label}_land")


def play_alpha_pb_to_caterpillar(session: ControllerSession) -> SuperMetroidState:
    """Leave collected Alpha PB rightward despite enemy timing differences.

    Public policy: jump the five midair platforms back to the right door.
    Do not fall into the floor Samus Eaters; Boyons can knock Samus off.
    Skip the missile-tank wall behind the Chozo. Ice-pin collect leave is
    ``(341,171)`` p138 facing left — turn and run right.
    https://wiki.supermetroid.run/Alpha_Power_Bomb_Room
    """
    require_room(session, ROOM_ALPHA_PB, "alpha_pb_to_caterpillar")
    select_weapon(session, 0)

    best_x = int(session.state.samus_x)
    stale = 0
    for frame in range(_ESCAPE_BUDGET):
        state = session.state
        if int(state.room_id) == ROOM_CATERPILLAR:
            wait_ordinary_room(
                session,
                ROOM_CATERPILLAR,
                settle_frames=260,
                label="alpha_pb_to_caterpillar",
                x_range=(20, 80),
                y_range=(1920, 1940),
            )
            for _ in range(60):
                state = session.state
                if int(state.samus_y) >= 1930 and int(state.velocity_y) == 0:
                    return state
                hold(session, 1, reason="alpha_pb_to_caterpillar_land")
            raise TimeoutError(
                "alpha_pb_to_caterpillar: Caterpillar entry did not land: "
                f"{session.state}"
            )
        if int(state.room_id) != ROOM_ALPHA_PB:
            raise RuntimeError(
                "alpha_pb_to_caterpillar: unexpected room "
                f"0x{int(state.room_id):04X}"
            )
        if int(state.max_power_bombs) <= 0:
            raise RuntimeError("alpha_pb_to_caterpillar: Alpha PB is not collected")

        x = int(state.samus_x)
        if x > best_x + 2:
            best_x = x
            stale = 0
        else:
            stale += 1

        if stale >= _PROGRESS_WINDOW:
            _clear_obstacle(session, label="alpha_pb_escape_stall")
            stale = 0
            best_x = int(session.state.samus_x)
            continue

        buttons = ["RIGHT", "B", "X"]
        if frame % 52 < 30:
            buttons.append("A")
        hold(session, 1, *buttons, reason="alpha_pb_escape_advance")

    state = session.state
    raise TimeoutError(
        "alpha_pb_to_caterpillar: escape timeout: "
        f"xy=({state.samus_x},{state.samus_y}) pose={state.pose}"
    )


__all__ = ["ROOM_ALPHA_PB", "ROOM_CATERPILLAR", "play_alpha_pb_to_caterpillar"]
