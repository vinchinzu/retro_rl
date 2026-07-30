"""Big Pink → Pink Power Bomb room door entries."""

from __future__ import annotations

from collections.abc import Callable

from super_metroid.ram import SuperMetroidState
from super_metroid.routes.controller_common import (
    ensure_morph,
    hold,
    is_morph,
    require_room,
    select_weapon,
    unmorph,
    wait_until,
)
from super_metroid.routes.post_spore.morph_bomb_roll import bomb_roll_left_safe
from super_metroid.routes.post_spore.rooms import (
    ROOM_BIG_PINK,
    ROOM_FARMING,
    ROOM_PINK_PB,
    ROOM_SUPER,
    SuperCollectEvidence,
)
from super_metroid.routes.runtime import ControllerSession

_hold = hold
_require_room = require_room
_select_weapon = select_weapon
_unmorph = unmorph

def play_big_pink_enter_pb_door_from_sill(
    session: ControllerSession,
    *,
    settle_frames: int = 200,
) -> SuperMetroidState:
    """Enter Pink PB room ``0x9E11`` from a Big Pink PB door ledge.

    Proven controller sequence (no place/WRAM once on the door ledge):

    1. Run left + shoot to open the blue door.
    2. Spin-jump left through the door alcove.
    3. Hold left until the room transition (~45–100 frames).
    4. Idle until ordinary gameplay (game state 8, no door transition).

    Works from either door once Samus is on the ledge:

    - **Top door** ``0x8DDE`` (preferred): solid ledge **x≈520–548, y≈907**
      → spawn ~y=130 (Mission Impossible top / crumble path).
    - **Bottom door** ``0x8E02``: place-bridge midair ~``(580,1136)`` or alcove
      ~``(530,1163)`` → spawn ~y=395. Not a hop-to island from main/upper
      (wall@613 full height; y1051 ledge is corridor roof).

    Pure climb onto either ledge from main shaft is still open.
    """
    _require_room(session, ROOM_BIG_PINK, "enter_pb_door_from_sill")
    state = session.state
    # Run-shoot opens blue door; spin carries through alcove.
    _hold(session, 10, "LEFT", "B", reason="pb_door_run")
    _hold(session, 4, "LEFT", "B", "X", reason="pb_door_shoot")
    _hold(session, 30, "LEFT", "B", "A", reason="pb_door_spin")
    entered = False
    for _ in range(120):
        state = _hold(session, 1, "LEFT", reason="pb_door_hold")
        if state.room_id == ROOM_PINK_PB:
            entered = True
            break
    if not entered:
        raise TimeoutError(
            f"pb_door_from_sill: did not reach 0x{ROOM_PINK_PB:04X}: {session.state}"
        )
    # Wait for multi-frame door load to ordinary gameplay.
    for frame in range(settle_frames):
        state = _hold(session, 1, reason="pb_door_settle")
        if (
            state.room_id == ROOM_PINK_PB
            and state.game_state == 8
            and state.door_transition == 0
            and frame > 20
        ):
            break
    if state.room_id != ROOM_PINK_PB:
        raise RuntimeError(
            f"pb_door_settle: left 0x{ROOM_PINK_PB:04X}: {session.state}"
        )
    return state



def play_big_pink_enter_pb_door_from_top_ledge(
    session: ControllerSession,
    *,
    settle_frames: int = 200,
) -> SuperMetroidState:
    """Enter Pink PB via top door ledge (~532, 907). Alias of sill entry.

    Expects Big Pink on the solid top-door ledge (x≈520–548, y≈900–920).
    Lands ``0x9E11`` top spawn (~y=130). Prefer this over bottom place-bridge
    for the Mission Impossible crumble/collect path.
    """
    return play_big_pink_enter_pb_door_from_sill(
        session, settle_frames=settle_frames
    )



