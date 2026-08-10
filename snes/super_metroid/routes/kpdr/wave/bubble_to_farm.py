"""K4 Bubble Mountain → Upper Norfair Farm pure return (rr-czg9 / rr-vqv3).

Source: ``post_single_to_bubble_pure`` ~(472,395) right mid sill after
Single→Bubble. Human tape Phase B hop 11 (f7624–8952):

1. LEFT walk + LEFT+A spin climb from y395 → upper seat ~y155 @ x≈311
2. LEFT hop / fall with morph into mid-low platform ~y531
3. Morph tunnels: RIGHT + DOWN drops through bottom chambers y745–905
4. LEFT morph roll → unmorph hop → bottom-most left blue door (node 4)
5. LEFT+B push → Farm ``0xAF72`` settle right top ~(472–523,139)

Reverse of scaffold ``play_farm_to_bubble`` (Farm right door ↔ Bubble
bottom-left). Open-loop human RLE (thrash windows trimmed) with y-band
stop gates + short reactive budgets.

Tape recon: ``docs/tasks/SM-SPEED-ICE-MOAT-HUMAN.md`` Phase B hop 11.
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
from super_metroid.routes.kpdr.k4_common import _STANDING_POSES
from super_metroid.routes.kpdr.rooms import ROOM_BUBBLE, ROOM_UPPER_NORFAIR_FARM
from super_metroid.routes.kpdr.wave.geometry import (
    BTF_BOTTOM_FLOOR_Y,
    BTF_BOTTOM_FRAMES,
    BTF_BOTTOM_Y_MIN,
    BTF_CLIMB_FRAMES,
    BTF_DOOR_FRAMES,
    BTF_DOOR_X,
    BTF_DOOR_Y,
    BTF_DROP_FRAMES,
    BTF_FARM_SETTLE,
    BTF_MID_LOW_Y,
    BTF_MID_Y,
    BTF_UPPER_Y,
)
from super_metroid.routes.runtime import ControllerSession
from super_metroid.routes.skills.knockback import escape_kb, is_knockback

_LEDGE = _STANDING_POSES | frozenset({1, 2, 9, 10, 37, 38})


def _y_band(state: SuperMetroidState, band: tuple[int, int]) -> bool:
    return band[0] <= int(state.samus_y) <= band[1]


def _on_door_sill(state: SuperMetroidState) -> bool:
    return (
        state.room_id == ROOM_BUBBLE
        and int(state.samus_x) <= BTF_DOOR_X + 12
        and _y_band(state, BTF_DOOR_Y)
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
            if st.room_id == ROOM_UPPER_NORFAIR_FARM:
                return
            if st.room_id != ROOM_BUBBLE:
                return
            if stop_when is not None and stop_when(st):
                return
            if is_knockback(st):
                escape_kb(session, label, "LEFT", stop_room_id=ROOM_UPPER_NORFAIR_FARM)
                continue
            hold(session, 1, *buttons, reason=label)


def _mid_to_upper(session: ControllerSession, label: str) -> None:
    """Pin y395 ~(472,395) → upper seat ~y155 (human f7624–7808)."""
    if int(session.state.samus_y) <= BTF_UPPER_Y[1] + 20:
        return
    if int(session.state.samus_y) > BTF_MID_Y[1] + 50:
        return

    unmorph(session)
    select_weapon(session, 0)
    for _ in range(20):
        st = session.state
        if int(st.velocity_y) == 0 and int(st.pose) in _LEDGE:
            break
        hold(session, 1, reason=f"{label}_mid_settle")

    _play_rle(
        session,
        f"{label}_climb",
        [
            (23, ("LEFT",)),
            (40, ("LEFT", "A")),
            (52, ("LEFT", "A", "X")),
            (3, ("LEFT",)),
            (5, ()),
            (3, ("A",)),
            (31, ("LEFT", "A")),
            (2, ("A",)),
            (8, ()),
            (8, ("X",)),
            (6, ()),
            (6, ("X",)),
            (4, ()),
        ],
        stop_when=lambda st: (
            _y_band(st, BTF_UPPER_Y)
            and int(st.velocity_y) == 0
            and int(st.pose) in _LEDGE
        ),
    )

    # Short reactive if still mid.
    for frame in range(BTF_CLIMB_FRAMES // 3):
        st = session.state
        if st.room_id != ROOM_BUBBLE:
            return
        if _y_band(st, BTF_UPPER_Y) and int(st.velocity_y) == 0:
            return
        if int(st.samus_y) <= BTF_UPPER_Y[1]:
            return
        if is_knockback(st):
            escape_kb(session, label, "LEFT", stop_room_id=ROOM_UPPER_NORFAIR_FARM)
            continue
        if is_morph(st.pose):
            hold(session, 3, "UP", reason=f"{label}_climb_unmorph")
            continue
        if int(st.velocity_y) != 0:
            hold(session, 1, "LEFT", "A", reason=f"{label}_climb_air")
            continue
        if int(st.samus_x) > 420:
            hold(session, 1, "LEFT", reason=f"{label}_climb_approach")
        else:
            phase = frame % 16
            if phase < 10:
                hold(session, 1, "LEFT", "A", reason=f"{label}_climb_spin")
            else:
                hold(session, 1, "LEFT", "A", "X", reason=f"{label}_climb_shot")


def _upper_to_mid_low(session: ControllerSession, label: str) -> None:
    """Upper ~y155 → mid-low ~y531 (human f7839–7985)."""
    y = int(session.state.samus_y)
    if y >= BTF_MID_LOW_Y[0] - 5:
        return
    if y > BTF_UPPER_Y[1] + 100:
        return

    unmorph(session)
    for _ in range(15):
        st = session.state
        if int(st.velocity_y) == 0 and int(st.pose) in _LEDGE:
            break
        hold(session, 1, reason=f"{label}_up_settle")

    _play_rle(
        session,
        f"{label}_drop",
        [
            (6, ("LEFT",)),
            (5, ("X",)),
            (4, ()),
            (1, ("LEFT",)),
            (9, ("LEFT", "A")),
            (54, ("LEFT",)),
            (14, ("LEFT", "A")),
            (6, ("LEFT",)),
            (3, ("DOWN",)),
            (7, ("DOWN", "X")),
            (5, ("DOWN",)),
            (6, ("DOWN", "X")),
            (4, ("DOWN",)),
            (5, ("DOWN", "X")),
            (5, ("DOWN", "LEFT")),
            (5, ("LEFT",)),
            (7, ("DOWN", "LEFT")),
            (20, ("DOWN",)),
        ],
        stop_when=lambda st: int(st.samus_y) >= BTF_MID_LOW_Y[0] - 5,
    )

    for frame in range(BTF_DROP_FRAMES // 3):
        st = session.state
        if st.room_id != ROOM_BUBBLE:
            return
        if int(st.samus_y) >= BTF_MID_LOW_Y[0] - 5:
            return
        if is_knockback(st):
            escape_kb(session, label, "LEFT", stop_room_id=ROOM_UPPER_NORFAIR_FARM)
            continue
        if int(st.samus_x) > 200:
            hold(session, 1, "LEFT", reason=f"{label}_drop_l")
        elif not is_morph(st.pose) and int(st.velocity_y) != 0:
            hold(session, 1, "DOWN", reason=f"{label}_drop_morph")
        else:
            hold(session, 1, "DOWN", "LEFT", reason=f"{label}_drop_dl")


def _mid_low_through_bottom_to_door(session: ControllerSession, label: str) -> None:
    """Mid-low y531 → bottom morph tunnels → Farm door (human f7988–8952).

    Full open-loop from human tape with thrash idle/X windows trimmed where
    geometry still advances. Reactive budgets fill residual stalls.
    """
    if session.state.room_id == ROOM_UPPER_NORFAIR_FARM:
        return

    # Prefer morph before tunnel RLE.
    if int(session.state.samus_y) < BTF_BOTTOM_Y_MIN and not is_morph(session.state.pose):
        try:
            ensure_morph(session, max_attempts=3)
        except TimeoutError:
            hold(session, 4, "DOWN", reason=f"{label}_ml_crouch")

    # Human f7988–8951 compressed (idle thrash shortened; keep RIGHT→LEFT path).
    _play_rle(
        session,
        f"{label}_bot_rle",
        [
            # mid-low RIGHT into drop shaft
            (15, ("RIGHT",)),
            (9, ("RIGHT", "A")),
            (12, ("RIGHT",)),
            (7, ("RIGHT", "X")),
            (12, ("RIGHT",)),
            (6, ("DOWN",)),
            (9, ("DOWN", "A")),
            (10, ("DOWN",)),
            (8, ("DOWN", "X")),
            (16, ("DOWN",)),
            (2, ()),
            (11, ("DOWN",)),
            (9, ("DOWN", "X")),
            (8, ("DOWN",)),
            (2, ()),
            (9, ("RIGHT",)),
            (12, ()),
            (7, ("DOWN",)),
            (5, ()),
            (5, ("DOWN",)),
            (11, ()),
            # land morph shelf ~x251 y745
            (40, ("RIGHT",)),
            (6, ()),
            (8, ("X",)),
            (20, ()),
            # human: long RIGHT to ~x363 (keep full length — geometry needs it)
            (130, ("RIGHT",)),
            (24, ()),
            # LEFT morph tunnels
            (32, ("LEFT",)),
            (10, ("LEFT", "X")),
            (108, ("LEFT",)),
            (11, ("LEFT", "X")),
            (42, ("LEFT",)),
            (18, ()),
            (15, ("LEFT",)),
            (11, ("LEFT", "X")),
            (154, ("LEFT",)),
            (1, ()),
            (10, ("UP",)),
            (1, ("UP", "LEFT")),
            (15, ("LEFT",)),
            (20, ("LEFT", "A")),
            (12, ("LEFT",)),
            (6, ("LEFT", "X")),
            (31, ("LEFT",)),
            (40, ("B", "LEFT")),
            (15, ("LEFT",)),
            (20, ()),
        ],
        stop_when=lambda st: st.room_id == ROOM_UPPER_NORFAIR_FARM,
    )

    if session.state.room_id == ROOM_UPPER_NORFAIR_FARM:
        return

    # Reactive bottom / door budget.
    for frame in range(BTF_BOTTOM_FRAMES):
        st = session.state
        if st.room_id == ROOM_UPPER_NORFAIR_FARM:
            return
        if st.room_id != ROOM_BUBBLE:
            return
        if is_knockback(st):
            escape_kb(session, label, "LEFT", stop_room_id=ROOM_UPPER_NORFAIR_FARM)
            continue

        x = int(st.samus_x)
        y = int(st.samus_y)
        vy = int(st.velocity_y)
        pose = int(st.pose)

        if _on_door_sill(st) or (
            x <= BTF_DOOR_X + 15 and _y_band(st, BTF_DOOR_Y) and vy == 0
        ):
            if is_morph(pose):
                hold(session, 4, "UP", reason=f"{label}_door_unmorph")
                continue
            phase = frame % 14
            if phase < 7:
                hold(session, 1, "B", "LEFT", reason=f"{label}_door_push")
            elif phase < 11:
                hold(session, 1, "LEFT", "X", reason=f"{label}_door_shot")
            else:
                hold(session, 1, "LEFT", reason=f"{label}_door_walk")
            continue

        # Still mid-low: keep right/down progress.
        if y < BTF_BOTTOM_Y_MIN:
            if not is_morph(pose) and vy == 0:
                try:
                    ensure_morph(session, max_attempts=2)
                except TimeoutError:
                    hold(session, 2, "DOWN", reason=f"{label}_rx_morph")
                continue
            if y < 620:
                hold(session, 1, "RIGHT", reason=f"{label}_rx_ml_r")
            elif y < 700:
                hold(session, 1, "DOWN" if frame % 3 else "RIGHT", reason=f"{label}_rx_ml_d")
            else:
                hold(session, 1, "RIGHT", "DOWN", reason=f"{label}_rx_ml_br")
            continue

        # Bottom shelves: human path RIGHT to ~x350 then LEFT morph.
        if y < BTF_BOTTOM_FLOOR_Y[0] - 30:
            if not is_morph(pose) and vy == 0 and x > 120:
                try:
                    ensure_morph(session, max_attempts=2)
                except TimeoutError:
                    hold(session, 2, "DOWN", reason=f"{label}_rx_bot_morph")
                continue
            if x < 340:
                # Alternate RIGHT with brief DOWN to seat into tube.
                if frame % 20 < 14:
                    hold(session, 1, "RIGHT", reason=f"{label}_rx_shelf_r")
                else:
                    hold(session, 1, "DOWN", reason=f"{label}_rx_shelf_d")
            else:
                # Past shelf: drop + LEFT.
                if y < BTF_BOTTOM_FLOOR_Y[0] - 10:
                    hold(session, 1, "DOWN", reason=f"{label}_rx_shelf_drop")
                else:
                    hold(session, 1, "LEFT", reason=f"{label}_rx_shelf_l")
            continue

        # Floor band y≥880: LEFT to door; hop if standing near sill.
        if x <= 100:
            if is_morph(pose):
                hold(session, 4, "UP", reason=f"{label}_rx_floor_up")
            elif y > BTF_DOOR_Y[1]:
                hold(session, 1, "LEFT", "A", reason=f"{label}_rx_floor_hop")
            else:
                hold(session, 1, "B", "LEFT", reason=f"{label}_rx_floor_push")
            continue

        if not is_morph(pose) and vy == 0:
            try:
                ensure_morph(session, max_attempts=2)
            except TimeoutError:
                hold(session, 1, "LEFT", reason=f"{label}_rx_floor_walk")
            continue
        hold(session, 1, "LEFT", reason=f"{label}_rx_floor_l")


def _door_budget(session: ControllerSession, label: str) -> None:
    for frame in range(BTF_DOOR_FRAMES):
        st = session.state
        if st.room_id == ROOM_UPPER_NORFAIR_FARM:
            return
        if st.room_id != ROOM_BUBBLE:
            return
        if is_knockback(st):
            escape_kb(session, label, "LEFT", stop_room_id=ROOM_UPPER_NORFAIR_FARM)
            continue
        if is_morph(st.pose):
            hold(session, 3, "UP", reason=f"{label}_final_unmorph")
            continue
        x = int(st.samus_x)
        if x > BTF_DOOR_X + 20:
            hold(session, 1, "LEFT", "B", reason=f"{label}_final_run")
            continue
        phase = frame % 12
        if phase < 7:
            hold(session, 1, "B", "LEFT", reason=f"{label}_final_push")
        elif phase < 10:
            hold(session, 1, "LEFT", "X", reason=f"{label}_final_shot")
        else:
            hold(session, 1, "LEFT", reason=f"{label}_final_walk")


def play_bubble_to_farm(session: ControllerSession) -> SuperMetroidState:
    """Bubble Mountain right mid pin → ordinary Upper Norfair Farm.

    Expects post-Single→Bubble pure ~(472,395). Climbs upper, morphs down the
    left shaft/bottom tunnels, leaves via bottom-most left blue door into
    Farm ``0xAF72``.
    """
    label = "bubble_to_farm"
    require_room(session, ROOM_BUBBLE, label)
    start = session.frame

    unmorph(session)
    select_weapon(session, 0)

    y = int(session.state.samus_y)
    if BTF_MID_Y[0] - 40 <= y <= BTF_MID_Y[1] + 50:
        _mid_to_upper(session, label)

    if session.state.room_id == ROOM_BUBBLE:
        y = int(session.state.samus_y)
        if y < BTF_MID_LOW_Y[0] - 5:
            _upper_to_mid_low(session, label)

    if session.state.room_id == ROOM_BUBBLE:
        _mid_low_through_bottom_to_door(session, label)

    if session.state.room_id == ROOM_BUBBLE and (
        _on_door_sill(session.state) or int(session.state.samus_x) <= 60
    ):
        _door_budget(session, label)

    if session.state.room_id != ROOM_UPPER_NORFAIR_FARM:
        state = session.state
        raise TimeoutError(
            f"{label}: Farm door missed; room=0x{state.room_id:04X} "
            f"pose={state.pose} xy=({state.samus_x},{state.samus_y}) "
            f"door_transition={state.door_transition} "
            f"frames={session.frame - start}"
        )

    return wait_ordinary_room(
        session,
        ROOM_UPPER_NORFAIR_FARM,
        settle_frames=BTF_FARM_SETTLE,
        label=label,
    )


__all__ = ["play_bubble_to_farm"]
