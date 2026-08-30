"""K4 Double Chamber → Single Chamber pure return (rr-qpkd / rr-vqv3 stack).

Source: ``post_wave_to_double_chamber_pure`` ~(984,139) Super door ledge after
Wave→Double. Human tape Phase B hop 9 (f6052–6752):

1. LEFT on high ledge + spin hop into Super column
2. Morph/fall mid platforms → floor y≈450 at x≈773
3. Floor LEFT with morph tunnel ~x450–550
4. Spin hop gap → bottom-left blue door y395 → Single ``0xAD5E`` ~(20,395)

Tape recon: ``docs/tasks/SM-SPEED-ICE-MOAT-HUMAN.md`` Phase B hop 9.
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
from super_metroid.routes.kpdr.norfair.common import _STANDING_POSES
from super_metroid.routes.kpdr.rooms import ROOM_DOUBLE_CHAMBER, ROOM_SINGLE_CHAMBER
from super_metroid.routes.kpdr.wave.geometry import (
    DTS_DOOR_FRAMES,
    DTS_DOOR_X,
    DTS_DOOR_Y,
    DTS_DROP_FRAMES,
    DTS_FLOOR_FRAMES,
    DTS_FLOOR_Y,
    DTS_FLOOR_Y_MIN,
    DTS_GAP_LAUNCH_X,
    DTS_HOP_LAUNCH_X,
    DTS_LEDGE_FRAMES,
    DTS_LEDGE_Y_MAX,
    DTS_MID_X,
    DTS_MID_Y,
    DTS_MORPH_TUNNEL_X,
    DTS_SINGLE_SETTLE,
)
from super_metroid.routes.runtime import ControllerSession
from super_metroid.routes.skills.knockback import escape_kb, is_knockback


def _on_floor(state: SuperMetroidState) -> bool:
    return (
        state.room_id == ROOM_DOUBLE_CHAMBER
        and state.samus_y >= DTS_FLOOR_Y_MIN
        and state.velocity_y == 0
    )


def _on_door_sill(state: SuperMetroidState) -> bool:
    return (
        state.room_id == ROOM_DOUBLE_CHAMBER
        and state.samus_x <= DTS_DOOR_X + 15
        and DTS_DOOR_Y[0] <= state.samus_y <= DTS_DOOR_Y[1]
        and state.velocity_y == 0
    )


def _ledge_left_and_hop(session: ControllerSession, label: str) -> None:
    """Super ledge ~(984,139) → LEFT run → spin hop into Super column mid."""
    unmorph(session)
    select_weapon(session, 0)
    for _ in range(25):
        state = hold(session, 1, reason=f"{label}_ledge_stand")
        if state.velocity_y == 0 and state.pose in _STANDING_POSES:
            break

    hopping = False
    for frame in range(DTS_LEDGE_FRAMES):
        state = session.state
        if state.room_id != ROOM_DOUBLE_CHAMBER:
            return
        if state.samus_y > DTS_LEDGE_Y_MAX + 10:
            return  # already dropping into column
        if is_knockback(state):
            escape_kb(session, label, "LEFT", stop_room_id=ROOM_SINGLE_CHAMBER)
            continue
        if is_morph(state.pose) or state.pose in (137, 138, 39, 40):
            hold(session, 4, "UP", reason=f"{label}_ledge_unmorph")
            continue

        x = int(state.samus_x)
        y = int(state.samus_y)
        grounded = int(state.velocity_y) == 0 and y <= DTS_LEDGE_Y_MAX

        # Commit spin hop once near launch band or already airborne from hop.
        if hopping or (grounded and x <= DTS_HOP_LAUNCH_X):
            hopping = True
            hold(session, 1, "LEFT", "B", "A", reason=f"{label}_ledge_hop")
            continue

        # Walk/run LEFT along ledge toward hop edge.
        phase = frame % 16
        if phase < 10:
            hold(session, 1, "LEFT", "B", reason=f"{label}_ledge_run")
        elif phase < 13:
            hold(session, 1, "LEFT", reason=f"{label}_ledge_walk")
        else:
            hold(session, 1, "LEFT", "X", reason=f"{label}_ledge_shot")


def _super_column_drop(session: ControllerSession, label: str) -> None:
    """Air/mid Super column → morph drop to floor y≥420."""
    for frame in range(DTS_DROP_FRAMES):
        state = session.state
        if state.room_id != ROOM_DOUBLE_CHAMBER:
            return
        if _on_floor(state):
            return
        if is_knockback(state):
            escape_kb(session, label, "LEFT", stop_room_id=ROOM_SINGLE_CHAMBER)
            continue

        x = int(state.samus_x)
        y = int(state.samus_y)
        vy = int(state.velocity_y)

        # Still on high ledge — re-hop left.
        if y <= DTS_LEDGE_Y_MAX and vy == 0:
            if x > DTS_HOP_LAUNCH_X:
                hold(session, 1, "LEFT", "B", reason=f"{label}_reledge")
            else:
                hold(session, 1, "LEFT", "B", "A", reason=f"{label}_rehop")
            continue

        # Mid platforms y∈[220,360]: morph and drop LEFT bias into column.
        if DTS_MID_Y[0] <= y <= DTS_MID_Y[1]:
            if vy == 0 and not is_morph(state.pose):
                # Landed mid — morph then roll/drop.
                try:
                    ensure_morph(session, max_attempts=3)
                except TimeoutError:
                    hold(session, 1, "DOWN", reason=f"{label}_mid_crouch")
                continue
            if is_morph(state.pose):
                # Morph fall / roll toward column center ~x788.
                if x > DTS_MID_X[1]:
                    hold(session, 1, "LEFT", reason=f"{label}_mid_morph_l")
                elif x < DTS_MID_X[0]:
                    hold(session, 1, "RIGHT", reason=f"{label}_mid_morph_r")
                else:
                    phase = frame % 10
                    if phase < 4:
                        hold(session, 1, "DOWN", reason=f"{label}_mid_morph_d")
                    elif phase < 7:
                        hold(session, 1, "LEFT", reason=f"{label}_mid_morph_nudge")
                    else:
                        hold(session, 1, reason=f"{label}_mid_morph_fall")
                continue
            # Air mid: LEFT bias + optional DOWN for morph entry.
            if x > 820:
                hold(session, 1, "LEFT", reason=f"{label}_mid_air_l")
            elif frame % 8 < 3:
                hold(session, 1, "DOWN", reason=f"{label}_mid_air_d")
            else:
                hold(session, 1, "LEFT", reason=f"{label}_mid_air")
            continue

        # Between ledge and mid (y 170–220) or deep fall: LEFT bias, try morph.
        if y < DTS_FLOOR_Y_MIN:
            if not is_morph(state.pose) and vy != 0 and frame % 12 < 3:
                hold(session, 1, "DOWN", reason=f"{label}_fall_morph")
            elif x > 800:
                hold(session, 1, "LEFT", reason=f"{label}_fall_l")
            elif x < 740:
                hold(session, 1, "RIGHT", reason=f"{label}_fall_r")
            else:
                hold(session, 1, reason=f"{label}_fall")
            continue

        # Near floor but still moving: let land.
        hold(session, 1, reason=f"{label}_land_wait")


def _floor_left_to_gap(session: ControllerSession, label: str) -> None:
    """Floor y≈450 LEFT through morph tunnel → pre-gap ~x190."""
    for frame in range(DTS_FLOOR_FRAMES):
        state = session.state
        if state.room_id == ROOM_SINGLE_CHAMBER:
            return
        if state.room_id != ROOM_DOUBLE_CHAMBER:
            return
        if is_knockback(state):
            escape_kb(session, label, "LEFT", stop_room_id=ROOM_SINGLE_CHAMBER)
            continue

        x = int(state.samus_x)
        y = int(state.samus_y)

        # Already at/past gap launch or on door sill.
        if x <= DTS_GAP_LAUNCH_X and y >= DTS_FLOOR_Y_MIN - 40:
            return
        if _on_door_sill(state) or x <= DTS_DOOR_X + 5:
            return

        # Fell into Super column still high — keep dropping.
        if y < DTS_FLOOR_Y_MIN - 20 and y > DTS_LEDGE_Y_MAX:
            if not is_morph(state.pose) and int(state.velocity_y) == 0:
                try:
                    ensure_morph(session, max_attempts=2)
                except TimeoutError:
                    hold(session, 1, "DOWN", reason=f"{label}_refall_d")
            else:
                hold(session, 1, "LEFT", reason=f"{label}_refall")
            continue

        # Morph tunnel band: must be morph to clear low ceiling.
        in_tunnel = DTS_MORPH_TUNNEL_X[0] <= x <= DTS_MORPH_TUNNEL_X[1]
        if in_tunnel or (
            x > DTS_MORPH_TUNNEL_X[0]
            and x < DTS_MORPH_TUNNEL_X[1] + 40
            and y >= DTS_FLOOR_Y[0] - 10
        ):
            if not is_morph(state.pose) and int(state.velocity_y) == 0:
                try:
                    ensure_morph(session, max_attempts=3)
                except TimeoutError:
                    hold(session, 1, "DOWN", reason=f"{label}_tunnel_d")
                continue
            hold(session, 1, "LEFT", reason=f"{label}_tunnel_roll")
            continue

        # Past tunnel (x < tunnel left): unmorph and run LEFT.
        if x < DTS_MORPH_TUNNEL_X[0] and is_morph(state.pose):
            unmorph(session)
            continue

        # Floor run LEFT; hop small steps; stay unmorphed outside tunnel.
        if is_morph(state.pose) and x > DTS_MORPH_TUNNEL_X[1]:
            # Pre-tunnel: unmorph for faster run unless already in low area.
            unmorph(session)
            continue

        if y < DTS_FLOOR_Y_MIN - 30 and int(state.velocity_y) != 0:
            hold(session, 1, "LEFT", reason=f"{label}_floor_air")
            continue

        phase = frame % 20
        if phase < 8:
            hold(session, 1, "LEFT", "B", reason=f"{label}_floor_run")
        elif phase < 12:
            hold(session, 1, "LEFT", "B", "A", reason=f"{label}_floor_hop")
        elif phase < 16:
            hold(session, 1, "LEFT", reason=f"{label}_floor_walk")
        else:
            hold(session, 1, "LEFT", "X", reason=f"{label}_floor_shot")


def _gap_and_left_door(session: ControllerSession, label: str) -> None:
    """Spin hop over left gap → sill y395 → LEFT blue door into Single."""
    unmorph(session)
    select_weapon(session, 0)

    for frame in range(DTS_DOOR_FRAMES):
        state = session.state
        if state.room_id == ROOM_SINGLE_CHAMBER:
            return
        if state.room_id != ROOM_DOUBLE_CHAMBER:
            return
        if is_knockback(state):
            escape_kb(session, label, "LEFT", stop_room_id=ROOM_SINGLE_CHAMBER)
            continue

        x = int(state.samus_x)
        y = int(state.samus_y)
        vy = int(state.velocity_y)

        if is_morph(state.pose) or state.pose in (137, 138, 39, 40):
            # Spring/morph under door column — hop unmorph up to sill.
            if y > DTS_DOOR_Y[1]:
                hold(session, 1, "LEFT", "A", reason=f"{label}_col_up")
            else:
                hold(session, 4, "UP", reason=f"{label}_door_unmorph")
            continue

        # On sill: push LEFT through blue door (shot pulses).
        if _on_door_sill(state) or (x <= DTS_DOOR_X + 8 and abs(y - 395) < 30 and vy == 0):
            phase = frame % 14
            if phase < 4:
                hold(session, 1, "LEFT", "X", reason=f"{label}_door_shot")
            elif phase < 10:
                hold(session, 1, "LEFT", "B", reason=f"{label}_door_push")
            else:
                hold(session, 1, "LEFT", "B", "A", reason=f"{label}_door_spin")
            continue

        # Left wall column x≈37 but low (y>415): climb with LEFT+A.
        if x <= 50 and y > DTS_DOOR_Y[1]:
            hold(session, 1, "LEFT", "A", reason=f"{label}_col_climb")
            continue

        # Approaching gap from floor: spin hop LEFT.
        if x > DTS_DOOR_X + 20:
            if y >= DTS_FLOOR_Y_MIN - 50 and vy == 0 and x <= DTS_GAP_LAUNCH_X + 40:
                hold(session, 1, "LEFT", "B", "A", reason=f"{label}_gap_hop")
            elif vy != 0:
                hold(session, 1, "LEFT", reason=f"{label}_gap_air")
            else:
                phase = frame % 16
                if phase < 8:
                    hold(session, 1, "LEFT", "B", reason=f"{label}_gap_run")
                elif phase < 12:
                    hold(session, 1, "LEFT", "B", "A", reason=f"{label}_gap_spin")
                else:
                    hold(session, 1, "LEFT", reason=f"{label}_gap_walk")
            continue

        # Default LEFT pressure.
        hold(session, 1, "LEFT", reason=f"{label}_door_nudge")


def play_double_to_single_chamber(session: ControllerSession) -> SuperMetroidState:
    """Double Chamber Super ledge → ordinary Single Chamber bottom-left.

    Expects post-Wave→Double pure pin ~(984,139). Leaves via bottom-left blue
    door into Single ``0xAD5E`` ~(20,395).
    """
    label = "double_to_single_chamber"
    require_room(session, ROOM_DOUBLE_CHAMBER, label)
    start = session.frame

    if session.state.samus_y <= DTS_LEDGE_Y_MAX + 20:
        _ledge_left_and_hop(session, label)

    if (
        session.state.room_id == ROOM_DOUBLE_CHAMBER
        and not _on_floor(session.state)
        and session.state.samus_y > DTS_LEDGE_Y_MAX
    ):
        _super_column_drop(session, label)
    elif (
        session.state.room_id == ROOM_DOUBLE_CHAMBER
        and session.state.samus_y > DTS_LEDGE_Y_MAX
        and not _on_floor(session.state)
    ):
        _super_column_drop(session, label)

    # If still high after hop (air), keep dropping.
    if (
        session.state.room_id == ROOM_DOUBLE_CHAMBER
        and session.state.samus_y < DTS_FLOOR_Y_MIN
    ):
        _super_column_drop(session, label)

    if (
        session.state.room_id == ROOM_DOUBLE_CHAMBER
        and session.state.samus_x > DTS_GAP_LAUNCH_X
    ):
        _floor_left_to_gap(session, label)

    if session.state.room_id == ROOM_DOUBLE_CHAMBER:
        _gap_and_left_door(session, label)

    if session.state.room_id != ROOM_SINGLE_CHAMBER:
        state = session.state
        raise TimeoutError(
            f"{label}: Single Chamber door missed; room=0x{state.room_id:04X} "
            f"pose={state.pose} xy=({state.samus_x},{state.samus_y}) "
            f"door_transition={state.door_transition} "
            f"frames={session.frame - start}"
        )

    return wait_ordinary_room(
        session,
        ROOM_SINGLE_CHAMBER,
        settle_frames=DTS_SINGLE_SETTLE,
        label=label,
    )


__all__ = ["play_double_to_single_chamber"]
