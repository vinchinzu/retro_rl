"""K4 Frog Speedway → Frog Save pure return (rr-05dp / rr-vqv3 hop 6).

Source: ``post_farm_to_speedway_pure`` ~(2008,139) Speedway **right** entry
after Farm left leave (8-screen tunnel). Human tape Phase B hop 12
(f9283–9707):

1. Transition settle on right sill ~(2000–2040,139)
2. Continuous B+LEFT dash across the full tunnel (Speed Booster breaks
   mid-room Boost Blocks — without Speed max progress stalls ~x795 from left)
3. Near left blue door: LEFT / LEFT+X / B+LEFT push into Frog Save ``0xB167``
4. Settle Frog Save **right** entry ~(200–240,139) (Speedway is east)

Tape recon: ``docs/tasks/SM-SPEED-ICE-MOAT-HUMAN.md`` Phase B hop 12.
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
from super_metroid.routes.kpdr.k4_common import _STANDING_POSES
from super_metroid.routes.kpdr.rooms import ROOM_FROG_SAVE, ROOM_FROG_SPEEDWAY
from super_metroid.routes.kpdr.wave.geometry import (
    SPEED_BOOSTER_MASK,
    STF_DOOR_FRAMES,
    STF_DOOR_X,
    STF_DOOR_Y,
    STF_FROG_SETTLE,
    STF_LEAVE_FRAMES,
    has_speed,
)
from super_metroid.routes.runtime import ControllerSession
from super_metroid.routes.skills.knockback import escape_kb, is_knockback

_LEDGE = _STANDING_POSES | frozenset({1, 2, 9, 10, 11, 12, 37, 38})


def _y_band(state: SuperMetroidState, band: tuple[int, int]) -> bool:
    return band[0] <= int(state.samus_y) <= band[1]


def _on_door_sill(state: SuperMetroidState) -> bool:
    return (
        state.room_id == ROOM_FROG_SPEEDWAY
        and int(state.samus_x) <= STF_DOOR_X + 12
        and _y_band(state, STF_DOOR_Y)
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
            if st.room_id == ROOM_FROG_SAVE:
                return
            if st.room_id != ROOM_FROG_SPEEDWAY:
                return
            if stop_when is not None and stop_when(st):
                return
            if is_knockback(st):
                escape_kb(session, label, "LEFT", stop_room_id=ROOM_FROG_SAVE)
                continue
            hold(session, 1, *buttons, reason=label)


def play_speedway_to_frog_save(session: ControllerSession) -> SuperMetroidState:
    """Frog Speedway right pin → ordinary Frog Save via left blue door.

    Expects post-Farm→Speedway pure ~(2008,139) with Speed collected. Runs
    LEFT through mid-room Boost Blocks and leaves into Frog Save ``0xB167``.
    """
    label = "speedway_to_frog_save"
    require_room(session, ROOM_FROG_SPEEDWAY, label)
    if not has_speed(session.state):
        raise RuntimeError(
            f"{label}: Speed Booster not collected "
            f"(items=0x{int(session.state.collected_items):04X}; "
            f"need bit 0x{SPEED_BOOSTER_MASK:04X})"
        )

    start = session.frame
    unmorph(session)
    select_weapon(session, 0)
    for _ in range(20):
        st = session.state
        if int(st.velocity_y) == 0 and int(st.pose) in _LEDGE:
            break
        hold(session, 1, reason=f"{label}_settle")

    # Human ordinary leave from right pin: long B+LEFT dash, then door pulse.
    # Transition idle stripped by post_farm_to_speedway_pure ordinary pin.
    _play_rle(
        session,
        f"{label}_rle",
        [
            (12, ("LEFT",)),
            (280, ("B", "LEFT")),
            (20, ("LEFT",)),
            (16, ("LEFT", "X")),
            (8, ("LEFT",)),
            (40, ("B", "LEFT")),
            (20, ("LEFT",)),
            (12, ("B", "LEFT", "X")),
        ],
        stop_when=lambda st: st.room_id == ROOM_FROG_SAVE,
    )

    min_x = int(session.state.samus_x)
    for frame in range(STF_LEAVE_FRAMES):
        st = session.state
        if st.room_id == ROOM_FROG_SAVE:
            break
        if st.room_id != ROOM_FROG_SPEEDWAY:
            break
        if is_knockback(st):
            escape_kb(session, label, "LEFT", stop_room_id=ROOM_FROG_SAVE)
            continue

        x = int(st.samus_x)
        y = int(st.samus_y)
        if x < min_x:
            min_x = x

        if _on_door_sill(st) or (x <= STF_DOOR_X + 20 and _y_band(st, STF_DOOR_Y)):
            phase = frame % 14
            if phase < 4:
                hold(session, 1, "LEFT", "X", reason=f"{label}_door_shot")
            elif phase < 11:
                hold(session, 1, "B", "LEFT", reason=f"{label}_door_push")
            else:
                hold(session, 1, "LEFT", reason=f"{label}_door_walk")
            continue

        # Mid-tunnel: keep Speed charge (B held) while running left.
        if int(st.velocity_y) != 0:
            hold(session, 1, "B", "LEFT", reason=f"{label}_rx_air")
            continue
        if y > STF_DOOR_Y[1] + 20:
            hold(session, 1, "LEFT", "A", reason=f"{label}_rx_up")
            continue
        phase = frame % 16
        if phase < 12:
            hold(session, 1, "B", "LEFT", reason=f"{label}_rx_run")
        elif phase < 14:
            hold(session, 1, "LEFT", "X", reason=f"{label}_rx_shot")
        else:
            hold(session, 1, "B", "LEFT", "X", reason=f"{label}_rx_dash_shot")
    else:
        st = session.state
        if st.room_id != ROOM_FROG_SAVE:
            for frame in range(STF_DOOR_FRAMES):
                st = session.state
                if st.room_id == ROOM_FROG_SAVE:
                    break
                if st.room_id != ROOM_FROG_SPEEDWAY:
                    break
                if is_knockback(st):
                    escape_kb(session, label, "LEFT", stop_room_id=ROOM_FROG_SAVE)
                    continue
                phase = frame % 12
                if phase < 4:
                    hold(session, 1, "LEFT", "X", reason=f"{label}_final_shot")
                elif phase < 10:
                    hold(session, 1, "B", "LEFT", reason=f"{label}_final_push")
                else:
                    hold(session, 1, "LEFT", reason=f"{label}_final_walk")

    if session.state.room_id != ROOM_FROG_SAVE:
        state = session.state
        stall = (
            " (boost-block stall; no Speed charge?)"
            if min_x >= 780 and int(state.samus_x) >= 780
            else ""
        )
        raise TimeoutError(
            f"{label}: Frog Save door missed; room=0x{state.room_id:04X} "
            f"pose={state.pose} xy=({state.samus_x},{state.samus_y}) "
            f"door_transition={state.door_transition} "
            f"frames={session.frame - start} min_x={min_x}{stall}"
        )

    return wait_ordinary_room(
        session,
        ROOM_FROG_SAVE,
        settle_frames=STF_FROG_SETTLE,
        label=label,
    )


__all__ = ["play_speedway_to_frog_save"]
