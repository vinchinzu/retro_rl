"""Alpha PB escape and K6 approach tapes to Moat."""

from __future__ import annotations

from pathlib import Path

from super_metroid.ram import SuperMetroidState
from super_metroid.routes.controller_common import hold, require_room, select_weapon, wait_ordinary_room
from super_metroid.routes.kpdr.room_ids import (
    ROOM_ALPHA_PB,
    ROOM_CATERPILLAR,
    ROOM_CRATERIA_KIHUNTER,
    ROOM_MOAT,
    ROOM_RED_BRINSTAR_ELEVATOR,
)
from super_metroid.routes.rle import load_rle_json, play_rle_room_exit
from super_metroid.routes.runtime import ControllerSession

_DATA = Path(__file__).resolve().parents[1] / "data"
_CATERPILLAR_RLE = load_rle_json(_DATA / "caterpillar_to_elevator_human_rle.json")
_ELEVATOR_RLE = load_rle_json(_DATA / "elevator_to_kihunter_human_rle.json")
_KIHUNTER_RLE = load_rle_json(_DATA / "kihunter_to_moat_human_rle.json")

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


def play_caterpillar_to_elevator(session: ControllerSession) -> SuperMetroidState:
    """Replay the dual-green climb from the natural Alpha PB return seat."""
    return play_rle_room_exit(
        session,
        from_room=ROOM_CATERPILLAR,
        to_room=ROOM_RED_BRINSTAR_ELEVATOR,
        script=_CATERPILLAR_RLE,
        label="caterpillar_to_elevator",
        settle_frames=260,
    )


def play_elevator_to_kihunter(session: ControllerSession) -> SuperMetroidState:
    """Ride up, traverse the connector, and settle in Kihunter."""
    return play_rle_room_exit(
        session,
        from_room=ROOM_RED_BRINSTAR_ELEVATOR,
        to_room=ROOM_CRATERIA_KIHUNTER,
        script=_ELEVATOR_RLE,
        label="elevator_to_kihunter",
    )


def play_kihunter_to_moat(session: ControllerSession) -> SuperMetroidState:
    """Traverse Kihunter from its natural elevator entry and settle in Moat."""
    return play_rle_room_exit(
        session,
        from_room=ROOM_CRATERIA_KIHUNTER,
        to_room=ROOM_MOAT,
        script=_KIHUNTER_RLE,
        label="kihunter_to_moat",
        settle_frames=260,
    )


__all__ = [
    "ROOM_ALPHA_PB",
    "ROOM_CATERPILLAR",
    "play_alpha_pb_to_caterpillar",
    "play_caterpillar_to_elevator",
    "play_elevator_to_kihunter",
    "play_kihunter_to_moat",
]
