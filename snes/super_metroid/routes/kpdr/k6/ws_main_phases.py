"""Named in-room phases for powered Main Shaft → Attic.

Hard-room split (rr-kw8t hop 2). Full hop GREEN is Attic ``0xCA52`` gs=8
only. Phase dumps / ``PhaseStop`` are not hop GREEN.

https://wiki.supermetroid.run/Wrecked_Ship_Main_Shaft
"""

from __future__ import annotations

from super_metroid.ram import SuperMetroidState
from super_metroid.routes.kpdr.k6.ws_main_actions import (
    FIRST_JUMP_LAND_X,
    FIRST_JUMP_LAND_Y,
    WS_MAIN_ATTIC_DOOR_X,
    at_ws_main_attic_door_seat,
    at_ws_main_first_jump_land,
    ws_main_attic_settled,
)
from super_metroid.routes.kpdr.room_ids import ROOM_WS_ATTIC, ROOM_WS_MAIN
from super_metroid.routes.skills.geometry import PhaseStop

ROOM_WS_WEST_SUPER = 0xCDA8
ROOM_WS_SAVE = 0xCE8A

# Seat → height → mid → top → door. grate_seat is the right hatch-lip
# ~(1177,1883). Do not open a later phase while an earlier one is red.
WS_MAIN_PHASES: tuple[str, ...] = (
    "pit_shot",
    "grate_seat",
    "west_super",
    "mid_climb",
    "attic_seat",
    "attic_door",
)

# Right hatch-lip. Floor HiJump peaks ~1868; left (1075,1845) is above it.
GRATE_SEAT_X = FIRST_JUMP_LAND_X
GRATE_SEAT_Y = FIRST_JUMP_LAND_Y
WEST_SUPER_Y = (1650, 1700)
MID_CLIMB_Y = (655, 705)
SHAFT_X = (1080, 1220)


def ws_main_phase_index(name: str) -> int:
    """Index in ``WS_MAIN_PHASES``. Raises on unknown names."""
    key = str(name).strip().lower().replace("-", "_")
    try:
        return WS_MAIN_PHASES.index(key)
    except ValueError:
        raise ValueError(
            f"unknown Main Shaft phase {name!r}; use one of {WS_MAIN_PHASES}"
        ) from None


def _in_main(state: SuperMetroidState) -> bool:
    return int(state.room_id) == ROOM_WS_MAIN and int(state.game_state) == 8


def at_ws_main_grate_seat(state: SuperMetroidState) -> bool:
    """Held right hatch-lip ~(1184, 1883). First stable seat."""
    return _in_main(state) and at_ws_main_first_jump_land(
        int(state.samus_x),
        int(state.samus_y),
        int(state.pose),
        int(state.velocity_y),
    )


def at_ws_main_west_super_band(state: SuperMetroidState) -> bool:
    """First shaft hop y~1675. Must stay in Main — save/west-super doors are out."""
    if int(state.room_id) in (ROOM_WS_WEST_SUPER, ROOM_WS_SAVE, ROOM_WS_ATTIC):
        return False
    x, y = int(state.samus_x), int(state.samus_y)
    return (
        _in_main(state)
        and SHAFT_X[0] <= x <= SHAFT_X[1]
        and WEST_SUPER_Y[0] <= y <= WEST_SUPER_Y[1]
    )


def at_ws_main_mid_climb(state: SuperMetroidState) -> bool:
    """Mid-shaft hop y~680 (past sponge / save height)."""
    if int(state.room_id) in (ROOM_WS_WEST_SUPER, ROOM_WS_SAVE, ROOM_WS_ATTIC):
        return False
    x, y = int(state.samus_x), int(state.samus_y)
    return (
        _in_main(state)
        and SHAFT_X[0] <= x <= SHAFT_X[1]
        and MID_CLIMB_Y[0] <= y <= MID_CLIMB_Y[1]
    )


def classify_ws_main_phase(state: SuperMetroidState) -> str:
    """Highest phase this still satisfies. Pin floor is ``pit_shot``."""
    if ws_main_attic_settled(state) or int(state.room_id) == ROOM_WS_ATTIC:
        return "attic_door"
    if at_ws_main_attic_door_seat(state) or (
        _in_main(state) and int(state.samus_y) <= 160
    ):
        return "attic_seat"
    x, y = int(state.samus_x), int(state.samus_y)
    in_shaft = _in_main(state) and SHAFT_X[0] <= x <= SHAFT_X[1]
    if at_ws_main_mid_climb(state) or (in_shaft and y <= MID_CLIMB_Y[1]):
        return "mid_climb"
    if at_ws_main_west_super_band(state) or (
        in_shaft and y <= WEST_SUPER_Y[1] and y < GRATE_SEAT_Y[0]
    ):
        return "west_super"
    if at_ws_main_grate_seat(state):
        return "grate_seat"
    return "pit_shot"


def ws_main_phase_done(state: SuperMetroidState, phase: str) -> bool:
    """True when ``phase``'s exit gate holds (or a later one)."""
    idx = ws_main_phase_index(phase)
    if idx >= ws_main_phase_index("attic_door"):
        return ws_main_attic_settled(state)
    if idx >= ws_main_phase_index("attic_seat"):
        return at_ws_main_attic_door_seat(state) or ws_main_attic_settled(state)
    if idx >= ws_main_phase_index("mid_climb"):
        return at_ws_main_mid_climb(state) or int(state.samus_y) <= MID_CLIMB_Y[1]
    if idx >= ws_main_phase_index("west_super"):
        return at_ws_main_west_super_band(state) or int(state.samus_y) <= WEST_SUPER_Y[1]
    if idx >= ws_main_phase_index("grate_seat"):
        return at_ws_main_grate_seat(state) or int(state.samus_y) <= GRATE_SEAT_Y[1]
    return _in_main(state)


__all__ = [
    "GRATE_SEAT_X",
    "GRATE_SEAT_Y",
    "MID_CLIMB_Y",
    "PhaseStop",
    "SHAFT_X",
    "WEST_SUPER_Y",
    "WS_MAIN_PHASES",
    "at_ws_main_attic_door_seat",
    "at_ws_main_grate_seat",
    "at_ws_main_mid_climb",
    "at_ws_main_west_super_band",
    "classify_ws_main_phase",
    "ws_main_attic_settled",
    "ws_main_phase_done",
    "ws_main_phase_index",
    "WS_MAIN_ATTIC_DOOR_X",
]
