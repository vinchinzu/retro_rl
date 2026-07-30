"""Big Pink main shaft → Green Hill Zone."""

from __future__ import annotations

from super_metroid.ram import SuperMetroidState
from super_metroid.routes.controller_common import (
    ensure_morph,
    hold,
    play_run_shoot_exit,
    require_room,
    select_weapon,
    unmorph,
    wait_ordinary_room,
)
from super_metroid.routes.kpdr.rooms import (
    ITEM_HI_JUMP,
    ROOM_BABY_KRAID,
    ROOM_BAT,
    ROOM_BELOW_SPAZER,
    ROOM_BIG_PINK,
    ROOM_BUSINESS,
    ROOM_EAST_TUNNEL,
    ROOM_GHZ,
    ROOM_GLASS,
    ROOM_HJ,
    ROOM_HJ_SHAFT,
    ROOM_KRAID,
    ROOM_KRAID_EYE,
    ROOM_NOOB,
    ROOM_RED_TOWER,
    ROOM_WAREHOUSE,
    ROOM_WAREHOUSE_KIHUNTER,
    ROOM_WEST_TUNNEL,
    ROOM_ZEELA,
)
from super_metroid.routes.runtime import ControllerSession

_hold = hold
_require_room = require_room
_select_weapon = select_weapon
_unmorph = unmorph
_wait_ordinary_room = wait_ordinary_room

def play_big_pink_to_ghz(session: ControllerSession) -> SuperMetroidState:
    """Natural Big Pink main-shaft anchor → Green Hill Zone.

    The GHZ door is the lower-right green door, not the upper-right wall beside
    the main-shaft anchor.  This descends through the lower winding morph
    tunnel, unmorphs to fire a Super from the left, and enters the door without
    placement or room/progression writes.

    Charge Beam is a separate side trip below the mass at x≈683. Its natural
    collect is known, but a conventional return is not yet route-ready, so this
    function takes the direct KPDR exit. The active route does not require an
    infinite bomb jump here.
    """
    _require_room(session, ROOM_BIG_PINK, "big_pink_to_ghz")
    ensure_morph(session)

    for _ in range(500):
        state = _hold(session, 1, "LEFT", reason="big_pink_lower_left")
        if state.samus_x <= 560 and state.samus_y >= 1540:
            break
    else:
        raise TimeoutError(f"big_pink_to_ghz: missed lower-left shelf: {state}")

    _unmorph(session)
    for _ in range(220):
        state = _hold(session, 1, "RIGHT", "B", "A", reason="big_pink_lower_drop")
        if state.samus_x >= 665 and state.samus_y >= 1660:
            break
    else:
        raise TimeoutError(f"big_pink_to_ghz: missed lower mass: {state}")

    _hold(session, 30, "RIGHT", "B", reason="big_pink_mass_run")
    _hold(session, 10, reason="big_pink_mass_settle")
    _hold(session, 12, "LEFT", reason="big_pink_mass_brake")
    _hold(session, 8, "A", reason="big_pink_mass_vertical")
    for _ in range(160):
        state = _hold(session, 1, "RIGHT", "A", reason="big_pink_tunnel_mount")
        if state.samus_x >= 705 and 1590 <= state.samus_y <= 1630:
            break
    else:
        raise TimeoutError(f"big_pink_to_ghz: missed morph-tunnel lip: {state}")
    _hold(session, 50, reason="big_pink_tunnel_lip_settle")
    ensure_morph(session)

    for frame in range(500):
        buttons = ("RIGHT", "X") if frame % 45 < 3 else ("RIGHT",)
        state = _hold(session, 1, *buttons, reason="big_pink_bomb_roll")
        if state.samus_x >= 900:
            break
    else:
        raise TimeoutError(f"big_pink_to_ghz: lower bomb-roll stalled: {state}")

    for _ in range(220):
        state = _hold(session, 1, "RIGHT", reason="big_pink_door_roll")
        if state.samus_x >= 970 and state.samus_y >= 1670:
            break
    else:
        raise TimeoutError(f"big_pink_to_ghz: missed green-door pocket: {state}")

    _unmorph(session)
    _select_weapon(session, 2)
    _hold(session, 25, "LEFT", reason="big_pink_super_standoff")
    _hold(session, 3, "RIGHT", reason="big_pink_face_door")
    _hold(session, 3, reason="big_pink_face_door_release")
    _hold(session, 2, "RIGHT", "X", reason="big_pink_green_door_super")
    _hold(session, 60, reason="big_pink_green_door_fuse")
    for _ in range(300):
        state = _hold(session, 1, "RIGHT", "B", reason="big_pink_enter_ghz")
        if state.room_id == ROOM_GHZ:
            break
    else:
        raise TimeoutError(f"big_pink_to_ghz: green door did not open: {state}")
    return _wait_ordinary_room(
        session, ROOM_GHZ, settle_frames=240, label="big_pink_to_ghz"
    )



