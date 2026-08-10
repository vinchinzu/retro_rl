"""Ice Gate → Business pure return (K5 stack hop 3).

Source: ``post_ice_tutorial_to_gate_pure`` ~(807, 131) mid-top Gate after
Tutorial→Gate dual 969f. Tape Phase B return hop 22 (Business Super entry
~(39, 907)).

Hybrid pure::

  1. Accept mid-top settle band ~(450–900, 100–200) — not door lip
  2. Cleaned human RLE: morph drop shaft → tunnel mouth → roll RIGHT
     through pipe y≈569 to Super door column (thrash stuck trimmed)
  3. Unmorph floor + RIGHT door pressure into Business Super
  4. Settle ordinary Business Super lip ~(40–90, 880–960)

Tape: ``tasks/speed_to_wave_ice_moat_human.json`` f19152–20145.
Do not clone full thrash RLE at tunnel mouth — use cleaned data JSON.
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
    BUSINESS_RETURN_SETTLE,
    GATE_SUPER_DOOR_X,
    GATE_TO_BUSINESS_FRAMES,
    GATE_TO_BUSINESS_RLE,
    GATE_TUNNEL_Y,
    ICE_SUPER_Y_MAX,
    ICE_SUPER_Y_MIN,
    LEDGE_POSES,
)
from super_metroid.routes.kpdr.rooms import ROOM_BUSINESS, ROOM_ICE_GATE
from super_metroid.routes.rle import play_script
from super_metroid.routes.runtime import ControllerSession
from super_metroid.routes.skills.knockback import (
    escape_knockback_spin,
    is_knockback,
)

_MORPH = frozenset(
    {27, 28, 29, 30, 31, 37, 38, 39, 40, 41, 42, 43, 45, 49, 50, 55, 65, 137, 138}
)
_STAND = frozenset({1, 2, 9, 10, 11})


def _kb(session: ControllerSession, label: str, prefer: str = "RIGHT") -> bool:
    if not is_knockback(session.state):
        return False
    escape_knockback_spin(
        session,
        prefer_dir=prefer,
        run_frames=2,
        spin_frames=10,
        label=label,
        ensure_beam=True,
        break_on_motion_clear=True,
    )
    return True


def _ensure_beam(session: ControllerSession) -> None:
    if int(session.state.selected_item) != 0:
        select_weapon(session, 0)


def _land_mid_top(session: ControllerSession, label: str) -> None:
    """Stabilize mid-top pin (air residual pose 81 OK after brief land)."""
    for _ in range(48):
        st = session.state
        if int(st.room_id) != ROOM_ICE_GATE:
            return
        if _kb(session, f"{label}_kb"):
            continue
        pose = int(st.pose)
        if pose in _MORPH and pose not in (41, 45, 49, 55):
            hold(session, 1, "UP", reason=f"{label}_up")
            continue
        if int(st.velocity_y) == 0 and pose in _STAND | LEDGE_POSES | frozenset(
            {81, 164, 166}
        ):
            break
        hold(session, 1, reason=f"{label}_land")
    _ensure_beam(session)


def _rle_drop_and_roll(session: ControllerSession, label: str) -> None:
    """Mid-top → morph shaft drop → tunnel roll via cleaned human RLE."""

    def _stop(state: SuperMetroidState) -> bool:
        if int(state.room_id) == ROOM_BUSINESS:
            return True
        x, y = int(state.samus_x), int(state.samus_y)
        # Right Super floor / door column — hand off to door pressure.
        if x >= GATE_SUPER_DOOR_X - 20 and y >= 620:
            return True
        if x >= 1720 and y >= 640 and int(state.velocity_y) == 0:
            return True
        return False

    play_script(
        session,
        GATE_TO_BUSINESS_RLE,
        reason=f"{label}_rle",
        room_id=ROOM_ICE_GATE,
        stop_when=_stop,
        on_lag="break",
    )


def _closed_loop_tunnel_roll(session: ControllerSession, label: str) -> None:
    """Fallback if RLE leaves mid-tunnel: remorph + roll RIGHT to door column."""
    if int(session.state.room_id) != ROOM_ICE_GATE:
        return

    for frame in range(700):
        st = session.state
        if int(st.room_id) == ROOM_BUSINESS:
            return
        if int(st.room_id) != ROOM_ICE_GATE:
            return
        if _kb(session, f"{label}_roll_kb"):
            continue

        x, y = int(st.samus_x), int(st.samus_y)
        pose = int(st.pose)

        if x >= GATE_SUPER_DOOR_X - 30 and y >= 600:
            return
        if x >= 1720 and y >= 620:
            return

        # On pipe top (y~555 standing) — morph + drop into tunnel y≈569.
        if (
            pose in _STAND | LEDGE_POSES
            and GATE_TUNNEL_Y[0] - 30 <= y <= GATE_TUNNEL_Y[0] + 5
            and x < 1600
        ):
            hold(session, 3, "DOWN", reason=f"{label}_pipe_morph")
            hold(session, 8, "RIGHT", reason=f"{label}_pipe_into")
            continue

        if pose not in _MORPH and y < 650 and x < 1700:
            hold(session, 2, "DOWN", reason=f"{label}_remorph")
            continue

        if y > GATE_TUNNEL_Y[1] + 50 and x < 1600:
            # Below tunnel — hop back up-right morph.
            if frame % 20 < 6:
                hold(session, 1, "RIGHT", "A", reason=f"{label}_climb_hop")
            else:
                hold(session, 1, "RIGHT", reason=f"{label}_climb_r")
            continue

        hold(session, 1, "RIGHT", reason=f"{label}_roll")


def _door_to_business(session: ControllerSession, label: str) -> None:
    """Right Super door column → Business Center."""
    if int(session.state.room_id) != ROOM_ICE_GATE:
        return

    for frame in range(480):
        st = session.state
        if int(st.room_id) == ROOM_BUSINESS:
            return
        if int(st.room_id) != ROOM_ICE_GATE:
            return
        if _kb(session, f"{label}_door_kb"):
            continue

        pose = int(st.pose)
        x, y = int(st.samus_x), int(st.samus_y)

        if pose in _MORPH and y >= 600:
            hold(session, 1, "UP", reason=f"{label}_door_up")
            continue

        if x < GATE_SUPER_DOOR_X - 100:
            hold(session, 1, "RIGHT", "B", reason=f"{label}_approach")
            continue

        # Drop from tunnel y to door floor y≈651 if needed.
        if y < 620 and x >= GATE_SUPER_DOOR_X - 80:
            hold(session, 1, "RIGHT", reason=f"{label}_floor_drop")
            continue

        phase = frame % 16
        if phase < 4:
            hold(session, 1, "RIGHT", "X", reason=f"{label}_door_shot")
        elif phase < 12:
            hold(session, 1, "RIGHT", "B", reason=f"{label}_door_push")
        else:
            hold(session, 1, "RIGHT", "B", "A", reason=f"{label}_door_spin")


def play_ice_gate_to_business(session: ControllerSession) -> SuperMetroidState:
    """Ice Gate mid-top return pin → ordinary Business Super lip.

    Source: ``post_ice_tutorial_to_gate_pure`` (accept mid-top band, not
    Tutorial door lip). Exit: Business ``0xA7DE`` Super door height.
    """
    label = "ice_gate_to_business"
    require_room(session, ROOM_ICE_GATE, label)
    start = session.frame
    _ensure_beam(session)
    _land_mid_top(session, f"{label}_pin")

    if int(session.state.room_id) == ROOM_BUSINESS:
        return wait_ordinary_room(
            session, ROOM_BUSINESS, settle_frames=BUSINESS_RETURN_SETTLE, label=label
        )

    for attempt in range(2):
        if int(session.state.room_id) == ROOM_BUSINESS:
            break
        if int(session.state.room_id) != ROOM_ICE_GATE:
            break
        if session.frame - start > GATE_TO_BUSINESS_FRAMES:
            break

        x = int(session.state.samus_x)
        y = int(session.state.samus_y)

        # Primary: RLE from mid-top / upper / mid shaft.
        if x < GATE_SUPER_DOOR_X - 50 and y < 700:
            if attempt == 0 or y < GATE_TUNNEL_Y[0] + 40:
                _rle_drop_and_roll(session, f"{label}_a{attempt}")
            else:
                _closed_loop_tunnel_roll(session, f"{label}_a{attempt}")

        if int(session.state.room_id) == ROOM_ICE_GATE and int(session.state.samus_x) < 1720:
            _closed_loop_tunnel_roll(session, f"{label}_a{attempt}_fb")

        if int(session.state.room_id) == ROOM_ICE_GATE:
            _door_to_business(session, f"{label}_a{attempt}")

    if int(session.state.room_id) != ROOM_BUSINESS:
        st = session.state
        raise TimeoutError(
            f"{label}: Business door missed; room=0x{int(st.room_id):04X} "
            f"pose={st.pose} xy=({st.samus_x},{st.samus_y}) "
            f"door_transition={st.door_transition} "
            f"frames={session.frame - start}"
        )

    state = wait_ordinary_room(
        session,
        ROOM_BUSINESS,
        settle_frames=BUSINESS_RETURN_SETTLE,
        label=label,
    )
    unmorph(session)
    for _ in range(60):
        st = hold(session, 1, reason=f"{label}_stand")
        if int(st.room_id) != ROOM_BUSINESS:
            break
        y = int(st.samus_y)
        if (
            int(st.velocity_y) == 0
            and int(st.pose) in _STAND | LEDGE_POSES
            and ICE_SUPER_Y_MIN - 40 <= y <= ICE_SUPER_Y_MAX + 40
            and int(st.door_transition) == 0
        ):
            return st
        if y > ICE_SUPER_Y_MAX + 80 and int(st.velocity_y) == 0:
            return st
    return state


__all__ = ["play_ice_gate_to_business"]
