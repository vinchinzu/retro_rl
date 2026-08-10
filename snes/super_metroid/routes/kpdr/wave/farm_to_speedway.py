"""K4 Upper Norfair Farm → Frog Speedway pure return (rr-z13h / rr-vqv3).

Source: ``post_bubble_to_farm_pure`` ~(472,139) Farm right-top after Bubble
bottom-left leave. Human tape Phase B hop 12 (f9081–9282):

1. LEFT walk from right-top pin
2. B+LEFT run across the short farm floor
3. Mid-room hop (B+LEFT+A) over the central gap / enemies
4. Near-door shot pulses + B+LEFT push through left blue door
5. Settle Frog Speedway ``0xB106`` **right** entry ~(2000–2040,139)
   (8-screen tunnel; Farm is east — next hop runs LEFT through Boost Blocks)

Product loadout needs **Speed** for the Wave→Business return stack (Speedway
Boost Blocks on hop 6). This hop itself is a short room leave; soft-check
Speed so a no-Speed source fails with a clear residual.

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
from super_metroid.routes.kpdr.rooms import ROOM_FROG_SPEEDWAY, ROOM_UPPER_NORFAIR_FARM
from super_metroid.routes.kpdr.wave.geometry import (
    FTS_DOOR_FRAMES,
    FTS_DOOR_X,
    FTS_DOOR_Y,
    FTS_LEAVE_FRAMES,
    FTS_MID_HOP_X,
    FTS_SPEEDWAY_SETTLE,
    SPEED_BOOSTER_MASK,
    has_speed,
)
from super_metroid.routes.runtime import ControllerSession
from super_metroid.routes.skills.knockback import escape_kb, is_knockback

_LEDGE = _STANDING_POSES | frozenset({1, 2, 9, 10, 11, 12, 37, 38})


def _y_band(state: SuperMetroidState, band: tuple[int, int]) -> bool:
    return band[0] <= int(state.samus_y) <= band[1]


def _on_door_sill(state: SuperMetroidState) -> bool:
    return (
        state.room_id == ROOM_UPPER_NORFAIR_FARM
        and int(state.samus_x) <= FTS_DOOR_X + 12
        and _y_band(state, FTS_DOOR_Y)
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
            if st.room_id == ROOM_FROG_SPEEDWAY:
                return
            if st.room_id != ROOM_UPPER_NORFAIR_FARM:
                return
            if stop_when is not None and stop_when(st):
                return
            if is_knockback(st):
                escape_kb(session, label, "LEFT", stop_room_id=ROOM_FROG_SPEEDWAY)
                continue
            hold(session, 1, *buttons, reason=label)


def play_farm_to_speedway(session: ControllerSession) -> SuperMetroidState:
    """Upper Norfair Farm right-top pin → ordinary Frog Speedway left entry.

    Expects post-Bubble→Farm pure ~(472,139) with Speed collected. Crosses
    the short farm floor LEFT and leaves via the left blue door into
    Speedway ``0xB106``.
    """
    label = "farm_to_speedway"
    require_room(session, ROOM_UPPER_NORFAIR_FARM, label)
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

    # Human f9081–9240 compressed from pin ~(472,139): walk → dash → hop → door.
    # Thrash idle after Bubble settle already stripped by post_bubble_to_farm pin.
    _play_rle(
        session,
        f"{label}_rle",
        [
            (10, ("LEFT",)),
            (35, ("B", "LEFT")),
            (44, ("B", "LEFT", "A")),
            (5, ("LEFT", "A")),
            (10, ("LEFT", "A", "X")),
            (4, ("LEFT", "X")),
            (1, ("LEFT", "A", "X")),
            (1, ("B", "LEFT", "A", "X")),
            (18, ("B", "LEFT", "A")),
            (70, ("B", "LEFT")),
        ],
        stop_when=lambda st: st.room_id == ROOM_FROG_SPEEDWAY,
    )

    # Reactive residual: keep LEFT progress / door push if open-loop stalls.
    for frame in range(FTS_LEAVE_FRAMES):
        st = session.state
        if st.room_id == ROOM_FROG_SPEEDWAY:
            break
        if st.room_id != ROOM_UPPER_NORFAIR_FARM:
            break
        if is_knockback(st):
            escape_kb(session, label, "LEFT", stop_room_id=ROOM_FROG_SPEEDWAY)
            continue

        x = int(st.samus_x)
        y = int(st.samus_y)
        vy = int(st.velocity_y)

        if _on_door_sill(st) or (x <= FTS_DOOR_X + 15 and _y_band(st, FTS_DOOR_Y)):
            phase = frame % 14
            if phase < 4:
                hold(session, 1, "LEFT", "X", reason=f"{label}_door_shot")
            elif phase < 11:
                hold(session, 1, "B", "LEFT", reason=f"{label}_door_push")
            else:
                hold(session, 1, "LEFT", reason=f"{label}_door_walk")
            continue

        # Mid-room: re-apply hop if still past hop window and grounded high.
        if x > FTS_MID_HOP_X:
            phase = frame % 18
            if phase < 10:
                hold(session, 1, "B", "LEFT", reason=f"{label}_rx_run")
            elif phase < 15:
                hold(session, 1, "B", "LEFT", "A", reason=f"{label}_rx_hop")
            else:
                hold(session, 1, "LEFT", "X", reason=f"{label}_rx_shot")
            continue

        # Approach door from mid-left: hop if airborne lag, else run+shot.
        if vy != 0:
            hold(session, 1, "B", "LEFT", reason=f"{label}_rx_air")
            continue
        if y > FTS_DOOR_Y[1] + 20:
            hold(session, 1, "LEFT", "A", reason=f"{label}_rx_up")
            continue
        phase = frame % 16
        if phase < 8:
            hold(session, 1, "B", "LEFT", reason=f"{label}_rx_lrun")
        elif phase < 12:
            hold(session, 1, "LEFT", "X", reason=f"{label}_rx_lshot")
        else:
            hold(session, 1, "B", "LEFT", "A", reason=f"{label}_rx_lhop")
    else:
        st = session.state
        if st.room_id != ROOM_FROG_SPEEDWAY:
            # Final short door budget if sill contact.
            for frame in range(FTS_DOOR_FRAMES):
                st = session.state
                if st.room_id == ROOM_FROG_SPEEDWAY:
                    break
                if st.room_id != ROOM_UPPER_NORFAIR_FARM:
                    break
                if is_knockback(st):
                    escape_kb(session, label, "LEFT", stop_room_id=ROOM_FROG_SPEEDWAY)
                    continue
                phase = frame % 12
                if phase < 4:
                    hold(session, 1, "LEFT", "X", reason=f"{label}_final_shot")
                elif phase < 10:
                    hold(session, 1, "B", "LEFT", reason=f"{label}_final_push")
                else:
                    hold(session, 1, "LEFT", reason=f"{label}_final_walk")

    if session.state.room_id != ROOM_FROG_SPEEDWAY:
        state = session.state
        raise TimeoutError(
            f"{label}: Speedway door missed; room=0x{state.room_id:04X} "
            f"pose={state.pose} xy=({state.samus_x},{state.samus_y}) "
            f"door_transition={state.door_transition} "
            f"frames={session.frame - start}"
        )

    return wait_ordinary_room(
        session,
        ROOM_FROG_SPEEDWAY,
        settle_frames=FTS_SPEEDWAY_SETTLE,
        label=label,
    )


__all__ = ["play_farm_to_speedway"]
