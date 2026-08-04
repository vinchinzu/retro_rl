"""K4 Rising Tide → Bubble Mountain pure controller.

Cross the 5-screen heated lava corridor with charged Hi-Jumps when low,
knockback spin-escapes, then continuous beam-shot pressure on the right blue
door. Caps: Morph, Bombs, Missiles, Supers, Hi-Jump, Varia — no Speed.
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
from super_metroid.routes.kpdr.rooms import (
    ROOM_BUBBLE,
    ROOM_RISING_TIDE,
)
from super_metroid.routes.runtime import ControllerSession
from super_metroid.routes.skills.knockback import (
    escape_knockback_spin,
    is_knockback,
)

# Rising Tide (0xAFA3) is 5×1 screens (80×16 blocks). Left lip spawn ≈ (39, 139).
# Right blue door is block [63, 7] → pixel ≈ (1008, 112). Platforming cross with
# charged Hi-Jumps when low; continuous RIGHT+B+X door pressure from x≥930
# while staying on the door ledge (y≤170). No Super required.
_RISING_CROSS_FRAMES = 5000
_RISING_TO_BUBBLE_SETTLE_FRAMES = 320


def _rising_land_and_arm(session: ControllerSession, label: str) -> None:
    """Entry settle on left lip + beam select (no cross inputs)."""
    for _ in range(40):
        state = hold(session, 1, reason=f"{label}_land")
        if state.velocity_y == 0 and state.pose in _STANDING_POSES:
            break
    unmorph(session)
    select_weapon(session, 0)


def _rising_cross_to_bubble(session: ControllerSession, label: str) -> None:
    """Open-loop corridor cross + right blue door (frame budgets unchanged)."""
    max_x = session.state.samus_x
    min_y = session.state.samus_y
    door_reached = False
    stuck_frames = 0
    last_x = session.state.samus_x

    for frame in range(_RISING_CROSS_FRAMES):
        state = session.state
        if state.room_id == ROOM_BUBBLE:
            break

        max_x = max(max_x, state.samus_x)
        min_y = min(min_y, state.samus_y)
        if abs(state.samus_x - last_x) <= 1:
            stuck_frames += 1
        else:
            stuck_frames = 0
            last_x = state.samus_x

        # Knockback / contact: spin-escape right (assist energy can stunlock).
        if is_knockback(state):
            escape_knockback_spin(
                session,
                prefer_dir="RIGHT",
                run_frames=6,
                spin_frames=20,
                label=label,
                stop_room_id=ROOM_BUBBLE,
            )
            stuck_frames = 0
            last_x = session.state.samus_x
            continue

        # Door approach band (x≥930): continuous shoot-run; keep door altitude.
        # Falling under the door platform (y>170) and walking past the shell
        # without transition is the common miss — climb back, then re-pressure.
        if state.samus_x >= 930:
            door_reached = True
            if state.selected_item != 0:
                select_weapon(session, 0)
            if state.samus_y > 170:
                if state.samus_x > 1040:
                    hold(session, 1, "LEFT", "B", reason=f"{label}_under_back")
                elif (
                    state.velocity_y == 0
                    and state.pose in _STANDING_POSES
                ):
                    for _ in range(14):
                        hold(session, 1, "A", reason=f"{label}_door_charge")
                    for _ in range(40):
                        st = hold(
                            session, 1, "RIGHT", "B", "A", reason=f"{label}_door_up"
                        )
                        if st.room_id == ROOM_BUBBLE:
                            break
                else:
                    hold(session, 1, "RIGHT", "B", "A", reason=f"{label}_under_hop")
                continue
            phase = frame % 16
            if phase < 8:
                inputs = ("RIGHT", "B", "X")
            elif phase < 12:
                inputs = ("RIGHT", "B", "A")
            else:
                inputs = ("RIGHT", "B")
            state = hold(session, 1, *inputs, reason=f"{label}_door")
            if state.room_id == ROOM_BUBBLE:
                break
            continue

        # Mid-room low: charged Hi-Jump to stay on platforms above lava.
        if (
            state.velocity_y == 0
            and state.samus_y > 150
            and state.pose in _STANDING_POSES
        ):
            for _ in range(12):
                hold(session, 1, "A", reason=f"{label}_charge")
            for _ in range(32):
                st = hold(session, 1, "RIGHT", "B", "A", reason=f"{label}_hj")
                if st.room_id == ROOM_BUBBLE:
                    break
            continue

        # Stuck on a ledge / enemy body: short reverse then re-commit right.
        if stuck_frames > 40:
            for _ in range(10):
                hold(session, 1, "LEFT", "B", reason=f"{label}_unstick_back")
            for _ in range(10):
                hold(session, 1, "A", reason=f"{label}_unstick_charge")
            for _ in range(35):
                hold(session, 1, "RIGHT", "B", "A", reason=f"{label}_unstick_jump")
            stuck_frames = 0
            last_x = session.state.samus_x
            continue

        # Default cross: run-jump cadence; occasional beam for Sovas.
        if state.selected_item != 0:
            select_weapon(session, 0)
        phase = frame % 32
        if phase < 3:
            inputs = ("RIGHT", "B", "X")
        elif phase < 22:
            inputs = ("RIGHT", "B", "A")
        else:
            inputs = ("RIGHT", "B")
        state = hold(session, 1, *inputs, reason=f"{label}_cross")
        if state.room_id == ROOM_BUBBLE:
            break
    else:
        state = session.state
        raise TimeoutError(
            f"{label}: right blue door missed before room "
            f"0x{ROOM_BUBBLE:04X}; room=0x{state.room_id:04X} "
            f"pose={state.pose} xy=({state.samus_x},{state.samus_y}) "
            f"door_transition={state.door_transition} max_x={max_x} "
            f"min_y={min_y} door_reached={door_reached} "
            f"selected={state.selected_item}"
        )


def play_rising_tide_to_bubble(session: ControllerSession) -> SuperMetroidState:
    """Rising Tide left lip → ordinary Bubble Mountain via right blue door.

    CATH-03 pure successor lands near the left blue lip (x≈39 / y≈139).  Cross
    the 5-screen heated lava corridor with charged Hi-Jumps when low, knockback
    spin-escapes, then continuous beam-shot pressure on the right blue door
    (block ``[63, 7]`` / ≈x1008 y112) — plant-and-shoot is unreliable here;
    frog-style RIGHT+B+X while keeping door-ledge altitude works.  Settle
    ordinary ``0xACB3`` (Bubble node 3, mid-left).  Caps: Morph, Bombs,
    Missiles, Supers, Hi-Jump, Varia — **no Speed**.

    Open-loop phases are named helpers only — frame budgets unchanged.
    """
    label = "rising_tide_to_bubble"
    require_room(session, ROOM_RISING_TIDE, label)

    _rising_land_and_arm(session, label)
    _rising_cross_to_bubble(session, label)

    return wait_ordinary_room(
        session,
        ROOM_BUBBLE,
        settle_frames=_RISING_TO_BUBBLE_SETTLE_FRAMES,
        label=label,
    )


__all__ = [
    "play_rising_tide_to_bubble",
]
