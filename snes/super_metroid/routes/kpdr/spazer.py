"""Early Spazer Beam detour (Below Spazer climb → collect → return).

Spazer Room ``0xA447`` is KPDR K2.2. Continuous power-on reaches Below Spazer
``0xA408`` with **Charge** on the main K1 path (``play_big_pink_to_ghz``).

Geometry status
---------------
* **Green-door entry** (top ledge y≲150 → Super → Spazer): pure-green from
  ``scratch/pre_spazer_door_with_charge`` (continuous-with-Charge + place).
* **Collect** (pedestal → beams ``0x1004``): pure-green from
  ``scratch/post_spazer_entry_pure``.
* **Return** (Spazer → Below Spazer top, clear of open Super door): pure-green
  from ``scratch/post_spazer_collect_pure`` → handoff ~``(380, 155)``.
* **Floor→mid spin / mid→top double WJ**: still residual (human guide path;
  mid→top probe-proven from y≈260). Full continuous fold waits on climb pure.
* **Top→floor→West** after return: residual (bomb-gap / shaft drop); do not
  call ``play_below_spazer_to_west`` from the top handoff — RIGHT re-enters
  Spazer while the Super door is still open.
"""

from __future__ import annotations

from super_metroid.ram import SuperMetroidState
from super_metroid.routes.controller_common import (
    hold,
    require_room,
    select_weapon,
    unmorph,
    wait_ordinary_room,
)
from super_metroid.routes.kpdr.rooms import ROOM_BELOW_SPAZER, ROOM_SPAZER
from super_metroid.routes.runtime import ControllerSession

# collected_beams bit for Spazer.
SPAZER_BEAM_MASK = 0x0004

# Top ledge / green-door band (guide green-door ~480,120).
_TOP_Y_MAX = 160
_DOOR_X_MIN = 450


def play_below_spazer_to_spazer(
    session: ControllerSession,
) -> SuperMetroidState:
    """Below Spazer top ledge → Spazer Room via Super green door.

    Requires: Below Spazer ordinary on the **top ledge** (y≲160, x≳400) facing
    the green Super door. Floor-entry sources must climb first (residual).

    Expected exit: Spazer Room ordinary, just inside left door.
    """
    require_room(session, ROOM_BELOW_SPAZER, "below_spazer_to_spazer")
    unmorph(session)

    if session.state.samus_y > _TOP_Y_MAX:
        raise NotImplementedError(
            "below_spazer_to_spazer: floor/mid climb residual "
            f"(need y≤{_TOP_Y_MAX} on top ledge; got y={session.state.samus_y}). "
            "Use scratch/pre_spazer_door_with_charge or complete WJ climb pure."
        )

    # Approach green Super door on top-right.
    for _ in range(120):
        state = hold(session, 1, "RIGHT", "B", reason="spazer_door_approach")
        if state.samus_x >= _DOOR_X_MIN and state.samus_y <= _TOP_Y_MAX:
            break
    else:
        raise TimeoutError(
            f"below_spazer_to_spazer: missed green-door lip: {session.state}"
        )

    hold(session, 10, reason="spazer_door_settle")
    select_weapon(session, 2)
    hold(session, 6, reason="spazer_super_ready")
    hold(session, 3, "RIGHT", reason="spazer_face_door")
    hold(session, 3, reason="spazer_face_door_release")
    hold(session, 8, "X", reason="spazer_green_door_super")
    hold(session, 50, reason="spazer_green_door_fuse")
    for _ in range(250):
        state = hold(session, 1, "RIGHT", "B", reason="spazer_enter")
        if state.room_id == ROOM_SPAZER:
            break
    else:
        raise TimeoutError(
            f"below_spazer_to_spazer: green door did not open: {session.state}"
        )
    return wait_ordinary_room(
        session, ROOM_SPAZER, settle_frames=120, label="below_spazer_to_spazer"
    )


def play_spazer_collect(session: ControllerSession) -> SuperMetroidState:
    """Spazer Room pedestal collect.

    Requires: Spazer Room ordinary on entry. Hop onto Chozo platform
    (~160–180, ~140), open with **Charge-cadence** beam shots (X + long
    wait — short taps miss with Charge equipped), collect bit 0x04.
    """
    require_room(session, ROOM_SPAZER, "spazer_collect")
    if session.state.collected_beams & SPAZER_BEAM_MASK:
        return session.state

    unmorph(session)
    try:
        select_weapon(session, 0)
    except RuntimeError:
        pass
    hold(session, 8, reason="spazer_weapon_settle")

    # Floor path (probe-green): RIGHT+B to x≈160, then Charge-cadence X while
    # easing right into the orb (~171,171). No pedestal hop required.
    for _ in range(200):
        state = hold(session, 1, "RIGHT", "B", reason="spazer_chozo_approach")
        if state.collected_beams & SPAZER_BEAM_MASK:
            return _finish_spazer_fanfare(session)
        if state.samus_x >= 158:
            break
    hold(session, 10, reason="spazer_chozo_settle")

    for _cycle in range(20):
        if int(session.state.pose) in (137, 138):
            hold(session, 6, "A", reason="spazer_lag_break")
            hold(session, 8, reason="spazer_lag_land")
        hold(session, 2, "RIGHT", reason="spazer_face_chozo")
        hold(session, 1, "X", reason="spazer_chozo_shot")
        for _ in range(20):
            state = hold(session, 1, "RIGHT", reason="spazer_chozo_wait")
            if state.collected_beams & SPAZER_BEAM_MASK:
                return _finish_spazer_fanfare(session)
        if session.state.collected_beams & SPAZER_BEAM_MASK:
            return _finish_spazer_fanfare(session)

    raise TimeoutError(
        f"spazer_collect: Spazer PLM not collected: {session.state}"
    )


def _finish_spazer_fanfare(session: ControllerSession) -> SuperMetroidState:
    """Hold through item-grab poses until standing with Spazer bit set."""
    if not (session.state.collected_beams & SPAZER_BEAM_MASK):
        raise TimeoutError(
            f"spazer_collect: Spazer bit missing: {session.state}"
        )
    # Pose 164 = item grab; 138 = fanfare/knockback lag.
    for _ in range(500):
        state = hold(session, 1, reason="spazer_item_fanfare")
        if int(state.pose) in (1, 2, 9, 10) and (
            state.collected_beams & SPAZER_BEAM_MASK
        ):
            hold(session, 16, reason="spazer_item_settle")
            return session.state
    # Nudge free of stuck grab poses.
    for _ in range(40):
        hold(session, 1, "A", reason="spazer_item_unstick")
        if int(session.state.pose) in (1, 2, 9, 10):
            break
        hold(session, 1, "UP", reason="spazer_item_unstick")
    hold(session, 16, reason="spazer_item_settle")
    if not (session.state.collected_beams & SPAZER_BEAM_MASK):
        raise TimeoutError(
            f"spazer_collect: Spazer bit missing after fanfare: {session.state}"
        )
    return session.state


def play_spazer_return_to_below(
    session: ControllerSession,
) -> SuperMetroidState:
    """Spazer Room → Below Spazer top via left blue door (handoff clear of Super).

    Requires: Spazer held and free of item-grab pose. Post-collect floor
    (~171,171) is below the door sill — double spin-jump LEFT onto the sill,
    shoot the blue door, exit, then nudge LEFT so the handoff pin is not
    door-trapped on the open Super door (RIGHT re-enters Spazer).

    Expected exit: Below Spazer ordinary on the top platform, x≲400
    (safe of green Super door ~480). Top→floor→West remains residual.
    """
    require_room(session, ROOM_SPAZER, "spazer_return_to_below")
    if not (session.state.collected_beams & SPAZER_BEAM_MASK):
        raise RuntimeError(
            f"spazer_return_to_below: Spazer not collected "
            f"(beams=0x{session.state.collected_beams:04X})"
        )

    unmorph(session)
    # Clear residual grab / lag poses from collect handoff.
    if int(session.state.pose) not in (1, 2, 9, 10):
        for _ in range(200):
            hold(session, 1, reason="spazer_return_clear")
            if int(session.state.pose) in (1, 2, 9, 10):
                break
        if int(session.state.pose) not in (1, 2, 9, 10):
            for _ in range(40):
                hold(session, 1, "A", reason="spazer_return_unstick")
                if int(session.state.pose) in (1, 2, 9, 10):
                    break
                hold(session, 1, "UP", reason="spazer_return_unstick")
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
        # On sill: stop, free lag poses, shoot blue door, walk out.
        for _ in range(30):
            hold(session, 1, reason="spazer_return_stop")
            if int(session.state.pose) in (1, 2):
                break
        for _ in range(25):
            if int(session.state.pose) in (1, 2, 9, 10):
                break
            hold(session, 1, "A", reason="spazer_return_lag_break")
            hold(session, 2, reason="spazer_return_lag_break")
        try:
            select_weapon(session, 0)
        except RuntimeError:
            pass
        hold(session, 4, "LEFT", reason="spazer_return_face")
        hold(session, 3, reason="spazer_return_face_rel")
        hold(session, 8, "X", reason="spazer_return_blue_shot")
        hold(session, 45, reason="spazer_return_blue_fuse")
        for _ in range(200):
            state = hold(session, 1, "LEFT", "B", reason="spazer_return_enter")
            if state.room_id == ROOM_BELOW_SPAZER:
                break
            if int(state.pose) in (137, 138, 164):
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
    # Nudge left off the open Super door so RIGHT is not an instant re-entry.
    for _ in range(50):
        state = hold(session, 1, "LEFT", "B", reason="spazer_return_clear_door")
        if state.samus_x <= 400 and state.room_id == ROOM_BELOW_SPAZER:
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
    if session.state.samus_x > 430:
        raise TimeoutError(
            f"spazer_return_to_below: still door-trapped "
            f"x={session.state.samus_x}: {session.state}"
        )
    return session.state


__all__ = [
    "ROOM_SPAZER",
    "SPAZER_BEAM_MASK",
    "play_below_spazer_to_spazer",
    "play_spazer_collect",
    "play_spazer_return_to_below",
]
