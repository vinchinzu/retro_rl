"""Spazer Room collect + return to Below Spazer top handoff."""

from __future__ import annotations

from super_metroid.ram import SuperMetroidState
from super_metroid.routes.controller_common import (
    hold,
    require_room,
    unmorph,
    wait_ordinary_room,
)
from super_metroid.routes.kpdr.rooms import ROOM_BELOW_SPAZER, ROOM_SPAZER
from super_metroid.routes.kpdr.spazer.geometry import (
    DOOR_TRAP_X_MAX,
    HANDOFF_X_MAX,
    SPAZER_BEAM_MASK,
    has_spazer,
    is_lag_pose,
    is_true_ground_pose,
)
from super_metroid.routes.kpdr.spazer.helpers import break_lag, try_select_weapon
from super_metroid.routes.runtime import ControllerSession
from super_metroid.routes.skills.geometry import POSE_KNOCKBACK


def _finish_spazer_fanfare(session: ControllerSession) -> SuperMetroidState:
    """Hold through item-grab poses until standing with Spazer bit set."""
    if not has_spazer(session.state):
        raise TimeoutError(f"spazer_collect: Spazer bit missing: {session.state}")
    for _ in range(500):
        state = hold(session, 1, reason="spazer_item_fanfare")
        if is_true_ground_pose(state) and has_spazer(state):
            hold(session, 16, reason="spazer_item_settle")
            return session.state
    for _ in range(40):
        hold(session, 1, "A", reason="spazer_item_unstick")
        if is_true_ground_pose(session.state):
            break
        hold(session, 1, "UP", reason="spazer_item_unstick")
    hold(session, 16, reason="spazer_item_settle")
    if not has_spazer(session.state):
        raise TimeoutError(
            f"spazer_collect: Spazer bit missing after fanfare: {session.state}"
        )
    return session.state


def play_spazer_collect(session: ControllerSession) -> SuperMetroidState:
    """Spazer Room pedestal collect.

    Requires: Spazer Room ordinary on entry. Floor path: RIGHT+B to x≈160,
    then Charge-cadence beam shots (X + long wait), collect bit 0x04.
    """
    require_room(session, ROOM_SPAZER, "spazer_collect")
    if has_spazer(session.state):
        return session.state

    unmorph(session)
    try_select_weapon(session, 0)
    hold(session, 8, reason="spazer_weapon_settle")

    for _ in range(200):
        state = hold(session, 1, "RIGHT", "B", reason="spazer_chozo_approach")
        if has_spazer(state):
            return _finish_spazer_fanfare(session)
        if state.samus_x >= 158:
            break
    hold(session, 10, reason="spazer_chozo_settle")

    for _cycle in range(20):
        if int(session.state.pose) in POSE_KNOCKBACK:
            hold(session, 6, "A", reason="spazer_lag_break")
            hold(session, 8, reason="spazer_lag_land")
        hold(session, 2, "RIGHT", reason="spazer_face_chozo")
        hold(session, 1, "X", reason="spazer_chozo_shot")
        for _ in range(20):
            state = hold(session, 1, "RIGHT", reason="spazer_chozo_wait")
            if has_spazer(state):
                return _finish_spazer_fanfare(session)
        if has_spazer(session.state):
            return _finish_spazer_fanfare(session)

    raise TimeoutError(f"spazer_collect: Spazer PLM not collected: {session.state}")


def _stand_after_collect(session: ControllerSession) -> None:
    """Clear residual grab / lag poses from collect handoff."""
    if is_true_ground_pose(session.state):
        return
    for _ in range(200):
        hold(session, 1, reason="spazer_return_clear")
        if is_true_ground_pose(session.state):
            return
    for _ in range(40):
        hold(session, 1, "A", reason="spazer_return_unstick")
        if is_true_ground_pose(session.state):
            return
        hold(session, 1, "UP", reason="spazer_return_unstick")


def play_spazer_return_to_below(
    session: ControllerSession,
) -> SuperMetroidState:
    """Spazer Room → Below Spazer top via left blue door (handoff clear of Super).

    Post-collect floor (~171,171) is below the door sill — double spin-jump
    LEFT onto the sill, shoot the blue door, exit, then nudge LEFT so the
    handoff pin is not door-trapped on the open Super door.

    Expected exit: Below Spazer ordinary on the top platform, x≲400.
    """
    require_room(session, ROOM_SPAZER, "spazer_return_to_below")
    if not has_spazer(session.state):
        raise RuntimeError(
            "spazer_return_to_below: Spazer not collected "
            f"(beams=0x{session.state.collected_beams:04X})"
        )

    unmorph(session)
    _stand_after_collect(session)
    hold(session, 12, reason="spazer_return_stand")

    # Floor → door sill: double spin-jump left over the mid ledge (x≈85 wall).
    hold(session, 5, "LEFT", "B", reason="spazer_return_runup")
    hold(session, 12, "LEFT", "B", "A", reason="spazer_return_spin1")
    hold(session, 40, "LEFT", "B", reason="spazer_return_land1")
    hold(session, 12, "LEFT", "B", "A", reason="spazer_return_spin2")
    for _ in range(100):
        state = hold(session, 1, "LEFT", "B", reason="spazer_return_to_sill")
        if state.room_id == ROOM_BELOW_SPAZER:
            break
        if state.samus_x <= 50 and state.samus_y <= 155:
            break

    if session.state.room_id != ROOM_BELOW_SPAZER:
        for _ in range(30):
            hold(session, 1, reason="spazer_return_stop")
            if int(session.state.pose) in (1, 2):
                break
        for _ in range(25):
            if is_true_ground_pose(session.state):
                break
            hold(session, 1, "A", reason="spazer_return_lag_break")
            hold(session, 2, reason="spazer_return_lag_break")
        try_select_weapon(session, 0)
        hold(session, 4, "LEFT", reason="spazer_return_face")
        hold(session, 3, reason="spazer_return_face_rel")
        hold(session, 8, "X", reason="spazer_return_blue_shot")
        hold(session, 45, reason="spazer_return_blue_fuse")
        for _ in range(200):
            state = hold(session, 1, "LEFT", "B", reason="spazer_return_enter")
            if state.room_id == ROOM_BELOW_SPAZER:
                break
            if is_lag_pose(state):
                hold(session, 8, "A", reason="spazer_return_unstick")
        else:
            raise TimeoutError(
                f"spazer_return_to_below: failed to leave Spazer: {session.state}"
            )

    wait_ordinary_room(
        session,
        ROOM_BELOW_SPAZER,
        settle_frames=160,
        label="spazer_return_to_below",
    )
    for _ in range(50):
        state = hold(session, 1, "LEFT", "B", reason="spazer_return_clear_door")
        if state.samus_x <= HANDOFF_X_MAX and state.room_id == ROOM_BELOW_SPAZER:
            break
        if state.room_id != ROOM_BELOW_SPAZER:
            raise TimeoutError(
                f"spazer_return_to_below: left Below Spazer during clear: {state}"
            )
    hold(session, 16, reason="spazer_return_handoff_settle")
    if session.state.room_id != ROOM_BELOW_SPAZER:
        raise TimeoutError(
            f"spazer_return_to_below: bad handoff room: {session.state}"
        )
    if session.state.samus_x > DOOR_TRAP_X_MAX:
        raise TimeoutError(
            f"spazer_return_to_below: still door-trapped "
            f"x={session.state.samus_x}: {session.state}"
        )
    return session.state


__all__ = [
    "SPAZER_BEAM_MASK",
    "play_spazer_collect",
    "play_spazer_return_to_below",
]
