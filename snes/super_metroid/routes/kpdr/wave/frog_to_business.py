"""K4 Frog Save → Business pure return (rr-vsjy / rr-vqv3 hop 7).

Source: ``post_speedway_to_frog_save_pure`` ~(216,122) Frog Save **right**
entry after Speedway left leave. Human tape Phase B hop 13 (f9708–9987):

1. Transition settle on right sill ~(200–240, y floor)
2. LEFT past the central save tube (Hi-Jump pulses — tube blocks flat run)
3. Near left blue door: LEFT / LEFT+X / B+LEFT push into Business ``0xA7DE``
4. Settle Business **right** entry (Frog Save is east of Business mid-shaft)

Mirrors ``play_frog_save_to_speedway`` direction (RIGHT tube clear) for the
return stack. Tape recon: ``docs/tasks/SM-SPEED-ICE-MOAT-HUMAN.md`` Phase B.
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
from super_metroid.routes.kpdr.norfair.common import _STANDING_POSES
from super_metroid.routes.kpdr.rooms import ROOM_BUSINESS, ROOM_FROG_SAVE
from super_metroid.routes.kpdr.wave.geometry import (
    FTB_BUSINESS_SETTLE,
    FTB_DOOR_FRAMES,
    FTB_DOOR_X,
    FTB_DOOR_Y,
    FTB_LEAVE_FRAMES,
    FTB_TUBE_X,
)
from super_metroid.routes.runtime import ControllerSession
from super_metroid.routes.skills.knockback import escape_kb, is_knockback

_LEDGE = _STANDING_POSES | frozenset({1, 2, 9, 10, 11, 12, 37, 38, 82})


def _y_band(state: SuperMetroidState, band: tuple[int, int]) -> bool:
    return band[0] <= int(state.samus_y) <= band[1]


def _on_door_sill(state: SuperMetroidState) -> bool:
    return (
        state.room_id == ROOM_FROG_SAVE
        and int(state.samus_x) <= FTB_DOOR_X + 12
        and _y_band(state, FTB_DOOR_Y)
        and int(state.velocity_y) == 0
    )


def _play_rle(
    session: ControllerSession,
    label: str,
    steps: list[tuple[int, tuple[str, ...]]],
    *,
    stop_when=None,
) -> None:
    for n, buttons in steps:
        for _ in range(n):
            st = session.state
            if st.room_id == ROOM_BUSINESS:
                return
            if st.room_id != ROOM_FROG_SAVE:
                return
            if stop_when is not None and stop_when(st):
                return
            if is_knockback(st):
                escape_kb(session, label, "LEFT", stop_room_id=ROOM_BUSINESS)
                continue
            hold(session, 1, *buttons, reason=label)


def play_frog_save_to_business(session: ControllerSession) -> SuperMetroidState:
    """Frog Save right pin → ordinary Business via left blue door.

    Expects post-Speedway→Frog pure ~(216,122). Clears the central save tube
    with Hi-Jump pulses and leaves into Business Center ``0xA7DE``.
    """
    label = "frog_save_to_business"
    require_room(session, ROOM_FROG_SAVE, label)

    start = session.frame
    unmorph(session)
    select_weapon(session, 0)
    for _ in range(24):
        st = session.state
        if int(st.velocity_y) == 0 and int(st.pose) in _LEDGE:
            break
        hold(session, 1, reason=f"{label}_settle")

    # Compact open-loop from right pin: walk → tube-clear hops → door approach.
    # Mirror of frog_save_to_speedway Hi-Jump pulse spacing (tube both sides).
    _play_rle(
        session,
        f"{label}_rle",
        [
            (8, ("LEFT",)),
            (20, ("B", "LEFT")),
            (12, ("B", "LEFT", "A")),
            (8, ("B", "LEFT")),
            (12, ("B", "LEFT", "A")),
            (16, ("B", "LEFT")),
            (12, ("LEFT", "X")),
            (8, ("LEFT",)),
            (24, ("B", "LEFT")),
            (16, ("B", "LEFT", "X")),
        ],
        stop_when=lambda st: st.room_id == ROOM_BUSINESS,
    )

    min_x = int(session.state.samus_x)
    for frame in range(FTB_LEAVE_FRAMES):
        st = session.state
        if st.room_id == ROOM_BUSINESS:
            break
        if st.room_id != ROOM_FROG_SAVE:
            break
        if is_knockback(st):
            escape_kb(session, label, "LEFT", stop_room_id=ROOM_BUSINESS)
            continue

        x = int(st.samus_x)
        y = int(st.samus_y)
        if x < min_x:
            min_x = x

        if _on_door_sill(st) or (x <= FTB_DOOR_X + 20 and _y_band(st, FTB_DOOR_Y)):
            phase = frame % 14
            if phase < 4:
                hold(session, 1, "LEFT", "X", reason=f"{label}_door_shot")
            elif phase < 11:
                hold(session, 1, "B", "LEFT", reason=f"{label}_door_push")
            else:
                hold(session, 1, "LEFT", reason=f"{label}_door_walk")
            continue

        # Tube band: keep hopping so the save capsule does not pin the run.
        if FTB_TUBE_X[0] <= x <= FTB_TUBE_X[1] + 40:
            phase = frame % 16
            if phase < 6:
                hold(session, 1, "B", "LEFT", "A", reason=f"{label}_rx_tube_hop")
            elif phase < 12:
                hold(session, 1, "B", "LEFT", reason=f"{label}_rx_tube_run")
            else:
                hold(session, 1, "LEFT", "X", reason=f"{label}_rx_tube_shot")
            continue

        if int(st.velocity_y) != 0:
            hold(session, 1, "B", "LEFT", reason=f"{label}_rx_air")
            continue
        if y > FTB_DOOR_Y[1] + 20:
            hold(session, 1, "LEFT", "A", reason=f"{label}_rx_up")
            continue
        phase = frame % 16
        if phase < 10:
            hold(session, 1, "B", "LEFT", reason=f"{label}_rx_run")
        elif phase < 13:
            hold(session, 1, "LEFT", "X", reason=f"{label}_rx_shot")
        else:
            hold(session, 1, "B", "LEFT", "A", reason=f"{label}_rx_hop")
    else:
        st = session.state
        if st.room_id != ROOM_BUSINESS:
            for frame in range(FTB_DOOR_FRAMES):
                st = session.state
                if st.room_id == ROOM_BUSINESS:
                    break
                if st.room_id != ROOM_FROG_SAVE:
                    break
                if is_knockback(st):
                    escape_kb(session, label, "LEFT", stop_room_id=ROOM_BUSINESS)
                    continue
                phase = frame % 12
                if phase < 4:
                    hold(session, 1, "LEFT", "X", reason=f"{label}_final_shot")
                elif phase < 10:
                    hold(session, 1, "B", "LEFT", reason=f"{label}_final_push")
                else:
                    hold(session, 1, "LEFT", reason=f"{label}_final_walk")

    if session.state.room_id != ROOM_BUSINESS:
        state = session.state
        tube = (
            " (save-tube stall?)"
            if FTB_TUBE_X[0] - 10 <= int(state.samus_x) <= FTB_TUBE_X[1] + 20
            else ""
        )
        raise TimeoutError(
            f"{label}: Business door missed; room=0x{state.room_id:04X} "
            f"pose={state.pose} xy=({state.samus_x},{state.samus_y}) "
            f"door_transition={state.door_transition} "
            f"frames={session.frame - start} min_x={min_x}{tube}"
        )

    state = wait_ordinary_room(
        session,
        ROOM_BUSINESS,
        settle_frames=FTB_BUSINESS_SETTLE,
        label=label,
    )
    # Continuous natural entry often lands floor-left (x≈20) near HJ door.
    # Pure dual pin is ~(216,1419); re-center right so Ice Super climb setup
    # does not LEFT-run into HJ shaft (rr-kxge continuous residual).
    unmorph(session)
    for _ in range(160):
        st = session.state
        if st.room_id != ROOM_BUSINESS:
            break
        x = int(st.samus_x)
        y = int(st.samus_y)
        # Floor band only — do not walk if still mid-shaft.
        if y < 1350:
            break
        if 200 <= x <= 240 and int(st.velocity_y) == 0:
            hold(session, 8, reason=f"{label}_business_floor_pin")
            return session.state
        if x < 200:
            hold(session, 1, "RIGHT", "B", reason=f"{label}_business_floor_r")
        elif x > 240:
            hold(session, 1, "LEFT", reason=f"{label}_business_floor_l")
        else:
            hold(session, 1, reason=f"{label}_business_floor_idle")
    return session.state


__all__ = ["play_frog_save_to_business"]
