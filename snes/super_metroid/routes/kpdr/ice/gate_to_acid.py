"""Pure Ice Beam Gate Room → Ice Beam Acid Room (left blue door).

Tape: ``speed_to_wave_ice_moat_human.json`` f10817→11231 entry Acid ~(786,651).
Human path: floor run LEFT along y≈651 from Business entry ~(1752,651), mid
crouch/hop around x≈880–900, enter left door. Skips Ice Tutorial on entry.

**Loadout:** needs **Speed** (Boost Blocks / solid around x≈1045 without it)
and preferably Wave (product continuous tip). Pure sources from pre-Speed
Business continuous will stick mid-room — use Wave tip handoff / grant Speed
for pure probes until continuous Wave→Business exists.
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
from super_metroid.routes.kpdr.rooms import ROOM_ICE_ACID, ROOM_ICE_GATE
from super_metroid.routes.runtime import ControllerSession
from super_metroid.routes.skills.knockback import (
    escape_knockback_spin,
    is_knockback,
)

# Floor band after Business Super entry (human ordinary ~(1752, 651)).
_GATE_FLOOR_Y = (600, 720)
# Mid crouch/hop (human pose 42 sink then A around x 885–920).
_MID_HOP_X = (820, 940)
# Left blue door to Acid (place-green ~780; tape entry 786).
_LEFT_DOOR_X = 50
_RUN_FRAMES = 1200
_ACID_SETTLE = 280
# collected_items bit for Speed Booster.
_SPEED_MASK = 0x2000


def play_ice_gate_to_acid(session: ControllerSession) -> SuperMetroidState:
    """Ice Gate right-lip entry → ordinary Acid Room via left blue door.

    Source: pure Business→Gate handoff with **Speed** loadout (Wave tip
    handoff preferred). Exit: Acid ``0xA75D`` ordinary.
    """
    label = "ice_gate_to_acid"
    require_room(session, ROOM_ICE_GATE, label)

    items = int(getattr(session.state, "collected_items", 0) or 0)
    if items & _SPEED_MASK == 0:
        # Soft warning path: still try (may clear if blocks already open).
        # Product pure expects Speed from Wave continuous spine.
        pass

    unmorph(session)
    if int(session.state.selected_item) != 0:
        select_weapon(session, 0)

    mid_hop_done = False
    for frame in range(_RUN_FRAMES):
        state = session.state
        if state.room_id == ROOM_ICE_ACID:
            break

        if is_knockback(state):
            escape_knockback_spin(
                session,
                prefer_dir="LEFT",
                run_frames=3,
                spin_frames=14,
                label=f"{label}_kb",
                run_with=("B", "X"),
                spin_with=("B", "A"),
                ensure_beam=True,
                break_on_motion_clear=True,
            )
            continue

        x = int(state.samus_x)
        y = int(state.samus_y)
        pose = int(state.pose)
        grounded = int(state.velocity_y) == 0 and pose in _STANDING_POSES | frozenset(
            {37, 38, 9, 10, 1, 2}
        )

        # Morph lag — unmorph.
        if pose in (31, 39, 40, 41, 42, 65):
            # Human uses brief crouch (42) before mid hop — only force unmorph
            # outside the hop band or after hop budget.
            if not (_MID_HOP_X[0] <= x <= _MID_HOP_X[1]) or mid_hop_done:
                unmorph(session)
                continue

        # Mid obstacle: crouch sink then short LEFT hop (tape ~f11104–11130).
        if (
            not mid_hop_done
            and _MID_HOP_X[0] <= x <= _MID_HOP_X[1]
            and _GATE_FLOOR_Y[0] <= y <= _GATE_FLOOR_Y[1]
        ):
            if grounded and pose not in (42, 40):
                hold(session, 4, "DOWN", reason=f"{label}_mid_crouch")
            # Hop pulse
            for _ in range(16):
                st = hold(session, 1, "LEFT", "B", "A", reason=f"{label}_mid_hop")
                if st.room_id == ROOM_ICE_ACID:
                    break
                if int(st.samus_x) < _MID_HOP_X[0] - 10:
                    break
            # Coast
            for _ in range(24):
                st = hold(session, 1, "LEFT", "B", reason=f"{label}_mid_coast")
                if st.room_id == ROOM_ICE_ACID or int(st.samus_x) < 780:
                    break
            mid_hop_done = True
            continue

        # Near left door: beam + walk in.
        if x <= 100:
            phase = frame % 14
            if phase < 3:
                hold(session, 1, "LEFT", "X", reason=f"{label}_door_shot")
            else:
                hold(session, 1, "LEFT", "B", reason=f"{label}_door_run")
            continue

        # Default floor dash: shoot while running (enemies on shelves).
        # No open-loop A — jumping into spin sticks on early pillars without Speed.
        phase = frame % 20
        if phase < 3:
            hold(session, 1, "LEFT", "B", "X", reason=f"{label}_run_shot")
        else:
            hold(session, 1, "LEFT", "B", reason=f"{label}_run")
    else:
        raise TimeoutError(
            f"{label}: Acid Room missed: {session.state} "
            f"(need Speed loadout if stuck near x≈1045; mid hop ~x885)"
        )

    state = wait_ordinary_room(
        session,
        ROOM_ICE_ACID,
        settle_frames=_ACID_SETTLE,
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


__all__ = ["play_ice_gate_to_acid"]
