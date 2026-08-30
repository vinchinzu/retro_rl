"""Pure Ice Beam Acid Room → Ice Beam Snake Room (left blue door).

Tape: ``speed_to_wave_ice_moat_human.json`` f11231→11964 entry Snake ~(20,139).
Verified room-clear open-loop (``policies/room_clears/room_a75d_…``) dual-greens
from the Gate→Acid pure handoff ``post_ice_gate_to_acid_pure`` ~(470,139).

Acid is a short **horizontal** 2-screen hop (run/jump cadence + door open), not
a freeze-climb. Operator 2WJ note applies to **Ice Snake vertical** (next hop
``rr-5if``) — do not invent freeze platforms here.
"""

from __future__ import annotations

from pathlib import Path

from super_metroid.ram import SuperMetroidState
from super_metroid.routes.controller_common import (
    hold,
    require_room,
    select_weapon,
    unmorph,
    wait_ordinary_room,
)
from super_metroid.routes.kpdr.ice.geometry import (
    ACID_FLOOR_Y_MAX,
    ACID_SNAKE_SETTLE_FRAMES,
    ACID_TO_SNAKE_RLE,
)
from super_metroid.routes.kpdr.norfair.common import _STANDING_POSES
from super_metroid.routes.kpdr.rooms import ROOM_ICE_ACID, ROOM_ICE_SNAKE
from super_metroid.routes.rle import play_script
from super_metroid.routes.runtime import ControllerSession
from super_metroid.routes.skills.knockback import (
    escape_knockback_spin,
    is_knockback,
)

# Door pressure budget if RLE ends still in Acid (RNG / lag).
_DOOR_PUSH_FRAMES = 280
_LEFT_DOOR_X = 70


def _acid_to_snake_rle(session: ControllerSession, label: str) -> None:
    """Play verified Acid traverse RLE; stop early on Snake entry."""

    def _stop(state: SuperMetroidState) -> bool:
        return int(state.room_id) == ROOM_ICE_SNAKE

    play_script(
        session,
        ACID_TO_SNAKE_RLE,
        reason=f"{label}_rle",
        room_id=ROOM_ICE_ACID,
        stop_when=_stop,
        on_lag="break",
    )


def _door_push_if_needed(session: ControllerSession, label: str) -> None:
    """Fallback left blue door pressure if RLE left us short of Snake."""
    for frame in range(_DOOR_PUSH_FRAMES):
        state = session.state
        if state.room_id == ROOM_ICE_SNAKE:
            return
        if state.room_id != ROOM_ICE_ACID:
            return

        if is_knockback(state):
            escape_knockback_spin(
                session,
                prefer_dir="LEFT",
                run_frames=3,
                spin_frames=12,
                label=f"{label}_kb",
                run_with=("B", "X"),
                spin_with=("B", "A"),
                ensure_beam=True,
                break_on_motion_clear=True,
            )
            continue

        x = int(state.samus_x)
        y = int(state.samus_y)
        # Fallen below floor band: climb back toward door height.
        if y > ACID_FLOOR_Y_MAX + 40:
            hold(session, 1, "LEFT", "B", "A", reason=f"{label}_recover")
            continue

        if x <= _LEFT_DOOR_X:
            phase = frame % 16
            if phase < 4:
                hold(session, 1, "LEFT", "X", reason=f"{label}_door_shot")
            elif phase < 10:
                hold(session, 1, "LEFT", "B", reason=f"{label}_door_run")
            else:
                hold(session, 1, "LEFT", "B", "A", reason=f"{label}_door_hop")
            continue

        # Still mid-room: keep left pressure (should be rare after RLE).
        phase = frame % 20
        if phase < 10:
            hold(session, 1, "LEFT", "B", "A", reason=f"{label}_push_hop")
        elif phase < 13:
            hold(session, 1, "LEFT", "B", "X", reason=f"{label}_push_shot")
        else:
            hold(session, 1, "LEFT", "B", reason=f"{label}_push_run")


def play_ice_acid_to_snake(session: ControllerSession) -> SuperMetroidState:
    """Acid Room floor pin → ordinary Ice Snake via left blue door.

    Source: pure Gate→Acid handoff (``post_ice_gate_to_acid_pure`` ~(470,139)).
    Exit: Ice Snake ``0xA8B9`` ordinary (room-clear settle ~y650 mid shaft).
    """
    label = "ice_acid_to_snake"
    require_room(session, ROOM_ICE_ACID, label)

    unmorph(session)
    if int(session.state.selected_item) != 0:
        select_weapon(session, 0)

    if is_knockback(session.state):
        escape_knockback_spin(
            session,
            prefer_dir="LEFT",
            run_frames=3,
            spin_frames=12,
            label=f"{label}_kb0",
            ensure_beam=True,
            break_on_motion_clear=True,
        )

    _acid_to_snake_rle(session, label)

    if session.state.room_id != ROOM_ICE_SNAKE:
        _door_push_if_needed(session, label)

    if session.state.room_id != ROOM_ICE_SNAKE:
        raise TimeoutError(
            f"{label}: Ice Snake missed: {session.state} "
            f"(source should be Acid floor ~y139 x~470; RLE {Path(__file__).name})"
        )

    state = wait_ordinary_room(
        session,
        ROOM_ICE_SNAKE,
        settle_frames=ACID_SNAKE_SETTLE_FRAMES,
        label=label,
    )
    unmorph(session)
    for _ in range(40):
        st = hold(session, 1, reason=f"{label}_stand")
        if (
            int(st.velocity_y) == 0
            and int(st.pose) in _STANDING_POSES
            and int(st.door_transition) == 0
        ):
            return st
    return state


__all__ = ["play_ice_acid_to_snake"]
