"""Ice Tutorial → Ice Gate pure return (K5 stack hop 2).

Source: ``post_ice_snake_to_tutorial_pure`` ~(39, 139) in ``0xA865`` after
Snake→Tutorial dual 2386f. Tape Phase B return hop 21 (Gate entry ~(494, 139)).

Hybrid pure::

  1. Partial cleaned human RLE left shelf → first gap → lower → mid hop
     onto structure lip ~(210–220, 140)
  2. Open-loop morph tunnel (double-DOWN midair → roll RIGHT to ~x300)
  3. Ice freeze pulses + long spin gap + door pressure into ``0xA815``

Do not clone boyon angled thrash RLE — freeze + gap spin only after tunnel.
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
from super_metroid.routes.kpdr.ice.geometry import (
    GATE_RETURN_SETTLE,
    TUTORIAL_DOOR_X,
    TUTORIAL_MID_RLE,
    TUTORIAL_TO_GATE_FRAMES,
)
from super_metroid.routes.kpdr.rooms import ROOM_ICE_GATE, ROOM_ICE_TUTORIAL
from super_metroid.routes.rle import play_script
from super_metroid.routes.runtime import ControllerSession
from super_metroid.routes.skills.knockback import (
    escape_knockback_spin,
    is_knockback,
)

_CROUCH_MORPH = frozenset(
    {31, 37, 38, 39, 40, 41, 42, 49, 50, 55, 61, 65, 137, 138}
)
_STAND = frozenset({1, 2, 9, 10, 11})


def _ensure_beam(session: ControllerSession) -> None:
    unmorph(session)
    if int(session.state.selected_item) != 0:
        select_weapon(session, 0)


def _kb(session: ControllerSession, label: str) -> bool:
    if not is_knockback(session.state):
        return False
    escape_knockback_spin(
        session,
        prefer_dir="RIGHT",
        run_frames=2,
        spin_frames=10,
        label=label,
        ensure_beam=True,
        break_on_motion_clear=True,
    )
    return True


def _stand_up(session: ControllerSession, label: str, frames: int = 28) -> None:
    for _ in range(frames):
        st = session.state
        if int(st.room_id) != ROOM_ICE_TUTORIAL:
            return
        pose = int(st.pose)
        if pose in _STAND and int(st.velocity_y) == 0:
            return
        if pose in _CROUCH_MORPH:
            hold(session, 1, "UP", reason=f"{label}_up")
        else:
            hold(session, 1, reason=f"{label}_wait")


def _land_pin(session: ControllerSession, label: str) -> None:
    for _ in range(40):
        st = session.state
        if int(st.room_id) != ROOM_ICE_TUTORIAL:
            return
        if _kb(session, f"{label}_kb"):
            continue
        pose = int(st.pose)
        if pose in _CROUCH_MORPH:
            hold(session, 1, "UP", reason=f"{label}_up")
            continue
        if int(st.velocity_y) == 0 and pose in _STAND | frozenset({164, 166, 75, 77}):
            break
        hold(session, 1, reason=f"{label}_land")
    _stand_up(session, f"{label}_stand")
    _ensure_beam(session)


def _rle_to_mid(session: ControllerSession, label: str) -> None:
    """Left pin → mid structure lip via partial cleaned human RLE."""

    def _stop(state: SuperMetroidState) -> bool:
        if int(state.room_id) == ROOM_ICE_GATE:
            return True
        x, y = int(state.samus_x), int(state.samus_y)
        return (
            x >= 208
            and 128 <= y <= 152
            and int(state.velocity_y) == 0
        )

    play_script(
        session,
        TUTORIAL_MID_RLE,
        reason=f"{label}_mid_rle",
        room_id=ROOM_ICE_TUTORIAL,
        stop_when=_stop,
        on_lag="break",
    )
    _stand_up(session, f"{label}_mid_stand")


def _morph_tunnel(session: ControllerSession, label: str) -> None:
    """Mid lip ~(210,140) → double-DOWN morph midair → roll to boyon shelf ~x300.

    Human f18619–18740: vertical A jump, DOWN+A / DOWN / release / DOWN morph,
    RIGHT roll on pipe y≈120 into open shelf.
    """
    if int(session.state.room_id) != ROOM_ICE_TUTORIAL:
        return
    _stand_up(session, f"{label}_pre")
    hold(session, 13, "A", reason=f"{label}_jump")
    hold(session, 3, "DOWN", "A", reason=f"{label}_da")
    hold(session, 5, "DOWN", reason=f"{label}_d1")
    hold(session, 3, reason=f"{label}_release")
    hold(session, 7, "DOWN", reason=f"{label}_d2")
    for _ in range(55):
        st = session.state
        if int(st.room_id) != ROOM_ICE_TUTORIAL:
            return
        if int(st.samus_x) >= 295:
            break
        hold(session, 1, "RIGHT", reason=f"{label}_roll")
    _stand_up(session, f"{label}_unmorph")


def _gap_and_door(session: ControllerSession, label: str) -> None:
    """Boyon shelf → Ice freeze → long spin gap → right blue door pressure."""
    if int(session.state.room_id) != ROOM_ICE_TUTORIAL:
        return

    # Freeze pulses (no angled thrash).
    for _ in range(4):
        if int(session.state.room_id) != ROOM_ICE_TUTORIAL:
            return
        hold(session, 3, "X", reason=f"{label}_freeze")
        hold(session, 4, reason=f"{label}_freeze_wait")
    hold(session, 6, reason=f"{label}_freeze_settle")
    hold(session, 8, "RIGHT", "B", reason=f"{label}_align")

    # Long spin gap (tuned dual-green: A×3 + B+RIGHT+A ×60).
    hold(session, 3, "A", reason=f"{label}_gap_a")
    hold(session, 60, "RIGHT", "B", "A", reason=f"{label}_gap_spin")

    # Multi-attempt land recovery + door pressure.
    for frame in range(360):
        st = session.state
        if int(st.room_id) == ROOM_ICE_GATE:
            return
        if int(st.room_id) != ROOM_ICE_TUTORIAL:
            return
        if _kb(session, f"{label}_door_kb"):
            continue

        pose = int(st.pose)
        if pose in _CROUCH_MORPH:
            hold(session, 1, "UP", reason=f"{label}_door_up")
            continue

        x, y = int(st.samus_x), int(st.samus_y)
        # Lower floor under door column — hop up to shelf.
        if y > 160:
            hold(session, 2, "A", reason=f"{label}_shelf_a")
            hold(session, 16, "RIGHT", "B", "A", reason=f"{label}_shelf_hop")
            continue

        if x < TUTORIAL_DOOR_X - 30:
            phase = frame % 18
            if phase < 10:
                hold(session, 1, "RIGHT", "B", reason=f"{label}_approach")
            elif phase < 14:
                hold(session, 1, "RIGHT", "B", "A", reason=f"{label}_approach_hop")
            else:
                hold(session, 1, "RIGHT", "X", reason=f"{label}_approach_shot")
            continue

        phase = frame % 16
        if phase < 4:
            hold(session, 1, "RIGHT", "X", reason=f"{label}_door_shot")
        elif phase < 12:
            hold(session, 1, "RIGHT", "B", reason=f"{label}_door_push")
        else:
            hold(session, 1, "RIGHT", "B", "A", reason=f"{label}_door_spin")


def play_ice_tutorial_to_gate(session: ControllerSession) -> SuperMetroidState:
    """Tutorial left-entry pin → ordinary Ice Gate via right blue door."""
    label = "ice_tutorial_to_gate"
    require_room(session, ROOM_ICE_TUTORIAL, label)
    start = session.frame
    _ensure_beam(session)
    _land_pin(session, f"{label}_pin")

    if int(session.state.room_id) == ROOM_ICE_GATE:
        return wait_ordinary_room(
            session, ROOM_ICE_GATE, settle_frames=GATE_RETURN_SETTLE, label=label
        )

    # One primary pass + one retry from current band if short.
    for attempt in range(2):
        if int(session.state.room_id) == ROOM_ICE_GATE:
            break
        if int(session.state.room_id) != ROOM_ICE_TUTORIAL:
            break
        if session.frame - start > TUTORIAL_TO_GATE_FRAMES:
            break

        x = int(session.state.samus_x)
        y = int(session.state.samus_y)

        if x < 200:
            _rle_to_mid(session, f"{label}_a{attempt}")
            x = int(session.state.samus_x)
            y = int(session.state.samus_y)

        if (
            int(session.state.room_id) == ROOM_ICE_TUTORIAL
            and 200 <= x < 290
            and y < 160
        ):
            _morph_tunnel(session, f"{label}_a{attempt}")

        if int(session.state.room_id) == ROOM_ICE_TUTORIAL and int(session.state.samus_x) >= 270:
            _gap_and_door(session, f"{label}_a{attempt}")
        elif int(session.state.room_id) == ROOM_ICE_TUTORIAL and int(session.state.samus_x) >= 200:
            # Tunnel short — retry morph once more then gap.
            _morph_tunnel(session, f"{label}_a{attempt}_retry")
            if int(session.state.room_id) == ROOM_ICE_TUTORIAL:
                _gap_and_door(session, f"{label}_a{attempt}_gap")

    if int(session.state.room_id) != ROOM_ICE_GATE:
        st = session.state
        raise TimeoutError(
            f"{label}: Gate door missed; room=0x{int(st.room_id):04X} "
            f"pose={st.pose} xy=({st.samus_x},{st.samus_y}) "
            f"door_transition={st.door_transition} "
            f"frames={session.frame - start}"
        )

    return wait_ordinary_room(
        session,
        ROOM_ICE_GATE,
        settle_frames=GATE_RETURN_SETTLE,
        label=label,
    )


__all__ = ["play_ice_tutorial_to_gate"]
