"""Wrecked Ship Basement → Phantoon's Room (rr-cjpp).

Unpowered hallway. Map station LEFT is dead — skip. Walk RIGHT. Bomb the
morph-tunnel obstruction (X while morph). Unmorph. Super the Gadora eye,
then the remaining blue shell. Enter ``0xCD13`` and settle. Do not fight.
"""

from __future__ import annotations

from super_metroid.ram import SuperMetroidState
from super_metroid.routes.controller_common import (
    ensure_morph,
    hold,
    is_morph,
    require_room,
    select_weapon,
    unmorph,
    wait_ordinary_room,
)
from super_metroid.routes.runtime import ControllerSession
from super_metroid.routes.skills.knockback import (
    escape_knockback_spin,
    is_knockback,
)

ROOM_WS_BASEMENT = 0xCC6F
ROOM_PHANTOON = 0xCD13
ROOM_WS_MAP = 0xCCCB
ROOM_WS_MAIN = 0xCAF6
WEAPON_SUPER = 2

# Pin lands on the center platform ~(657,91) p165 after a few idle frames.
# Floor standing ~y=187; morph ~y=201. Map door is far left.
# Bomb-block stall in the morph tunnel ~x=1051. Eye alcove x≳1160.
WS_BASEMENT_FLOOR_Y = 170
WS_BASEMENT_MORPH_X_MIN = 930
WS_BASEMENT_BOMB_X_MIN = 1000
WS_BASEMENT_ALCOVE_X = 1160
WS_BASEMENT_MAP_X = 80
_FALLING_POSES = frozenset({23, 24, 25, 26})
_STAND_POSES = frozenset({1, 2, 9, 10})
_RUN_BUDGET = 420
_BOMB_CYCLES = 6
_EYE_BUDGET = 900
_SETTLE = 200


def ws_basement_phantoon_settled(state: SuperMetroidState) -> bool:
    """Ordinary Phantoon-room handoff: ``0xCD13`` gs=8 door_transition=0."""
    return (
        int(state.room_id) == ROOM_PHANTOON
        and int(state.game_state) == 8
        and int(state.door_transition) == 0
    )


def at_ws_basement_bomb_blocks(state: SuperMetroidState) -> bool:
    """True in the morph-tunnel bomb-block stall of unpowered Basement."""
    return (
        int(state.room_id) == ROOM_WS_BASEMENT
        and int(state.samus_x) >= WS_BASEMENT_BOMB_X_MIN
        and int(state.samus_y) >= WS_BASEMENT_FLOOR_Y
    )


def at_ws_basement_eye_seat(state: SuperMetroidState) -> bool:
    """True in the Gadora alcove (past the morph tunnel) of Basement."""
    return (
        int(state.room_id) == ROOM_WS_BASEMENT
        and int(state.samus_x) >= WS_BASEMENT_ALCOVE_X
        and int(state.samus_y) >= WS_BASEMENT_FLOOR_Y - 40
    )


def _guard(session: ControllerSession, label: str) -> None:
    room = int(session.state.room_id)
    if room == ROOM_PHANTOON:
        return
    if room == ROOM_WS_MAP:
        raise TimeoutError(f"{label}: entered map 0xCCCB: {session.state}")
    if room == ROOM_WS_MAIN:
        raise TimeoutError(f"{label}: back to Main Shaft 0xCAF6: {session.state}")
    if room != ROOM_WS_BASEMENT:
        raise TimeoutError(
            f"{label}: left Basement into 0x{room:04X}: {session.state}"
        )
    if int(session.state.samus_x) < WS_BASEMENT_MAP_X:
        raise TimeoutError(f"{label}: walked into left map door: {session.state}")


def _land(session: ControllerSession, label: str) -> None:
    """Idle out of pose-24 fall onto the center platform (or floor)."""
    for _ in range(80):
        st = session.state
        _guard(session, label)
        pose = int(st.pose)
        if (
            pose not in _FALLING_POSES
            and int(st.velocity_y) == 0
            and not is_knockback(st)
        ):
            return
        hold(session, 1, reason=f"{label}_land")


def _kb(session: ControllerSession, label: str) -> None:
    escape_knockback_spin(
        session,
        prefer_dir="RIGHT",
        run_frames=6,
        spin_frames=24,
        label=label,
        stop_room_id=ROOM_PHANTOON,
    )


def _run_to_morph_seat(session: ControllerSession, label: str) -> None:
    """Turn RIGHT, drop off the platform, tank Coverns, stop on the floor."""
    hold(session, 12, "RIGHT", reason=f"{label}_turn")
    for _ in range(_RUN_BUDGET):
        st = session.state
        _guard(session, label)
        if int(st.room_id) == ROOM_PHANTOON:
            return
        if is_knockback(st):
            _kb(session, f"{label}_run_kb")
            continue
        x = int(st.samus_x)
        y = int(st.samus_y)
        if (
            y >= WS_BASEMENT_FLOOR_Y
            and x >= WS_BASEMENT_MORPH_X_MIN
            and int(st.velocity_y) == 0
            and not is_morph(int(st.pose))
        ):
            return
        hold(session, 1, "RIGHT", "B", reason=f"{label}_run")
    raise TimeoutError(f"{label}: did not reach morph-tunnel floor: {session.state}")


def _bomb_tunnel(session: ControllerSession, label: str) -> None:
    """Morph-roll RIGHT; bomb (X) at the tunnel stall; reach the eye alcove."""
    if int(session.state.room_id) == ROOM_PHANTOON:
        return
    ensure_morph(session)
    for _cycle in range(_BOMB_CYCLES):
        st = session.state
        _guard(session, label)
        if int(st.room_id) == ROOM_PHANTOON:
            return
        if int(st.samus_x) >= WS_BASEMENT_ALCOVE_X:
            return
        prev = int(st.samus_x)
        stall = 0
        rolled = False
        for _ in range(160):
            st = session.state
            _guard(session, label)
            if int(st.room_id) == ROOM_PHANTOON:
                return
            if int(st.samus_x) >= WS_BASEMENT_ALCOVE_X:
                return
            if is_knockback(st):
                _kb(session, f"{label}_roll_kb")
                stall = 0
                continue
            if not is_morph(int(st.pose)):
                ensure_morph(session)
            hold(session, 1, "RIGHT", reason=f"{label}_roll")
            rolled = True
            x = int(session.state.samus_x)
            stall = stall + 1 if abs(x - prev) < 2 else 0
            prev = x
            if stall >= 16:
                # Morph bombs are X, not A.
                hold(session, 3, "X", reason=f"{label}_bomb")
                hold(session, 80, reason=f"{label}_boom")
                break
        else:
            if rolled and int(session.state.samus_x) >= WS_BASEMENT_ALCOVE_X:
                return
            break
    if int(session.state.room_id) == ROOM_PHANTOON:
        return
    if int(session.state.samus_x) < WS_BASEMENT_ALCOVE_X:
        raise TimeoutError(
            f"{label}: morph tunnel did not reach alcove: {session.state}"
        )


def _super_eye_door(session: ControllerSession, label: str) -> None:
    """Super the Gadora (eye-open cycle), then the remaining blue shell."""
    if int(session.state.room_id) == ROOM_PHANTOON:
        return
    unmorph(session)
    select_weapon(session, WEAPON_SUPER)
    for frame in range(_EYE_BUDGET):
        st = session.state
        _guard(session, label)
        if int(st.room_id) == ROOM_PHANTOON:
            return
        if is_knockback(st):
            _kb(session, f"{label}_eye_kb")
            continue
        if is_morph(int(st.pose)):
            unmorph(session)
            continue
        phase = frame % 28
        if phase < 4:
            hold(session, 1, "RIGHT", "X", reason=f"{label}_super")
        elif phase >= 16:
            hold(session, 1, "RIGHT", "B", "A", reason=f"{label}_spin")
        else:
            hold(session, 1, "RIGHT", "B", reason=f"{label}_run")
    if int(session.state.room_id) != ROOM_PHANTOON:
        raise TimeoutError(f"{label}: Gadora / blue door missed: {session.state}")


def play_ws_basement_to_phantoon(session: ControllerSession) -> SuperMetroidState:
    """Unpowered basement hallway. Map station LEFT is dead — skip. Walk RIGHT.

    Bomb the morph-tunnel obstruction (X while morph; morph bombs are X, not A).
    Unmorph. Gadora eye door: Super Missile (already selected) or 3 missiles
    while the eye is open. The Gadora leaves a blue shell — shoot and walk
    through. Enter Phantoon ``0xCD13``. Coverns possible; tank. Do not fight
    Phantoon this card.

    https://wiki.supermetroid.run/Wrecked_Ship_Basement

    Source: ``scratch/post_ws_main_to_basement.state`` ``0xCC6F`` ~(657,92)
    p24 gs=8 (still falling, facing left). Lands ordinary ``gs=8``.
    """
    label = "ws_basement_to_phantoon"
    if ws_basement_phantoon_settled(session.state):
        return session.state
    require_room(session, ROOM_WS_BASEMENT, label)
    _land(session, label)
    _run_to_morph_seat(session, label)
    _bomb_tunnel(session, label)
    _super_eye_door(session, label)
    return wait_ordinary_room(
        session, ROOM_PHANTOON, settle_frames=_SETTLE, label=label
    )


__all__ = [
    "ROOM_PHANTOON",
    "ROOM_WS_BASEMENT",
    "ROOM_WS_MAP",
    "WEAPON_SUPER",
    "WS_BASEMENT_ALCOVE_X",
    "WS_BASEMENT_BOMB_X_MIN",
    "WS_BASEMENT_FLOOR_Y",
    "WS_BASEMENT_MORPH_X_MIN",
    "at_ws_basement_bomb_blocks",
    "at_ws_basement_eye_seat",
    "play_ws_basement_to_phantoon",
    "ws_basement_phantoon_settled",
]
