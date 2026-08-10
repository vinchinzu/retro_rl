"""Ice Snake mid-right → Ice Tutorial pure return (K5 stack hop 1).

Source: ``post_ice_to_snake_pure`` ~(472, 395) pose 10 in ``0xA8B9`` after
Ice→Snake dual 538f. Tape Phase B return hop 20: drop right column to floor,
2WJ left-shaft climb (reuse ``_snake_platform_climb``; multi-attempt — L3 is
pin-sensitive post-Ice), top cross RIGHT into Tutorial blue door ``0xA865``.

Do not clone freeze thrash RLE — open-loop only for the morph-tunnel drop
geometry; climb is the proven platform-band helper from ``snake_to_ice``.
"""

from __future__ import annotations

from super_metroid.ram import SuperMetroidState
from super_metroid.routes.controller_common import (
    hold,
    is_morph,
    require_room,
    select_weapon,
    unmorph,
    wait_ordinary_room,
)
from super_metroid.routes.kpdr.ice.geometry import (
    SNAKE_HANDOFF_Y,
    SNAKE_TOP_TO_TUTORIAL_FRAMES,
    SNAKE_TOP_Y,
    SNAKE_TO_TUTORIAL_DROP_FRAMES,
    SNAKE_TUTORIAL_DOOR_X,
    SNAKE_TUTORIAL_DOOR_Y,
    TUTORIAL_RETURN_SETTLE,
    in_ice_snake,
    on_snake_top,
)
from super_metroid.routes.kpdr.ice.snake_to_ice import (
    _settle_ground,
    _snake_platform_climb,
)
from super_metroid.routes.kpdr.rooms import ROOM_ICE_SNAKE, ROOM_ICE_TUTORIAL
from super_metroid.routes.runtime import ControllerSession
from super_metroid.routes.skills.knockback import (
    escape_knockback_spin,
    is_knockback,
)

# Floor handoff align (acid-like pin works best with platform climb bands).
_FLOOR_ALIGN_X = 216
_CLIMB_ATTEMPTS = 4


def _on_snake_floor_band(state: SuperMetroidState) -> bool:
    if not in_ice_snake(state):
        return False
    y = int(state.samus_y)
    return SNAKE_HANDOFF_Y[0] <= y <= SNAKE_HANDOFF_Y[1] + 40


def _snake_mid_right_to_floor(session: ControllerSession, label: str) -> SuperMetroidState:
    """Ice-door mid-right ~(472, 395) → floor band ~(210, 651).

    Live dual-green sequence (not freeze thrash)::

        LEFT off Ice alcove hole → y427 ledge
          → LEFT+A hop onto tunnel roof ~y351–363
            → double DOWN morph drop into tunnel floor y377
              → morph roll LEFT → freefall past mid shelves → y651
    """
    unmorph(session)
    if int(session.state.selected_item) != 0:
        select_weapon(session, 0)

    # --- Ice shelf → y427 ledge (hole ~x440) ---
    for _ in range(120):
        st = session.state
        if not in_ice_snake(st):
            return st
        if int(st.samus_y) >= 420 and int(st.samus_x) <= 335:
            break
        if is_knockback(st):
            escape_knockback_spin(
                session,
                prefer_dir="LEFT",
                run_frames=2,
                spin_frames=8,
                label=f"{label}_shelf_kb",
                ensure_beam=True,
                break_on_motion_clear=True,
            )
            continue
        hold(session, 1, "LEFT", reason=f"{label}_shelf_left")

    # --- Hop onto tunnel roof shelf ---
    hold(session, 3, "A", reason=f"{label}_hop_a")
    hold(session, 22, "LEFT", "A", reason=f"{label}_hop_la")
    for _ in range(40):
        st = session.state
        if int(st.velocity_y) == 0 and int(st.samus_y) < 400:
            break
        hold(session, 1, "LEFT", reason=f"{label}_hop_coast")

    # --- Double DOWN morph into tunnel floor y377 ---
    for _ in range(6):
        hold(session, 1, "DOWN", reason=f"{label}_morph1")
    for _ in range(4):
        hold(session, 1, reason=f"{label}_morph_pause")
    for _ in range(6):
        hold(session, 1, "DOWN", reason=f"{label}_morph2")
    for _ in range(8):
        hold(session, 1, reason=f"{label}_morph_settle")

    # --- Roll LEFT through tunnel ---
    for _ in range(80):
        st = session.state
        if not in_ice_snake(st):
            return st
        if int(st.samus_x) < 200 or int(st.samus_y) > 450:
            break
        hold(session, 1, "LEFT", reason=f"{label}_tunnel_roll")

    # --- Freefall past mid shelves to floor ---
    for frame in range(SNAKE_TO_TUTORIAL_DROP_FRAMES):
        st = session.state
        if not in_ice_snake(st):
            return st
        if _on_snake_floor_band(st) and int(st.velocity_y) == 0:
            break

        if is_knockback(st):
            escape_knockback_spin(
                session,
                prefer_dir="LEFT",
                run_frames=2,
                spin_frames=8,
                label=f"{label}_fall_kb",
                ensure_beam=True,
                break_on_motion_clear=True,
            )
            continue

        y = int(st.samus_y)
        pose = int(st.pose)
        # Mid shelf ~y521: morph nudge off platform (tape RIGHT/LEFT wobble).
        if 500 <= y <= 560 and int(st.velocity_y) == 0:
            if not (is_morph(pose) or pose in (39, 40, 41, 42, 49, 50)):
                hold(session, 1, "DOWN", reason=f"{label}_midshelf_morph")
            elif (frame // 8) % 2 == 0:
                hold(session, 1, "RIGHT", reason=f"{label}_midshelf_r")
            else:
                hold(session, 1, "LEFT", reason=f"{label}_midshelf_l")
            continue
        if 400 <= y <= 500 and int(st.velocity_y) == 0:
            hold(session, 1, "DOWN", reason=f"{label}_mid_morph")
            continue
        hold(session, 1, reason=f"{label}_fall")
    else:
        st = session.state
        if not _on_snake_floor_band(st):
            raise TimeoutError(
                f"{label}: floor drop missed; room=0x{int(st.room_id):04X} "
                f"pose={st.pose} xy=({st.samus_x},{st.samus_y}) "
                f"door_transition={st.door_transition}"
            )

    unmorph(session)
    for _ in range(20):
        hold(session, 1, "UP", reason=f"{label}_floor_unmorph")
    _settle_ground(session, f"{label}_floor")

    # Align toward acid-like floor pin — climb bands are x-sensitive at L2/L3.
    for _ in range(30):
        st = session.state
        if abs(int(st.samus_x) - _FLOOR_ALIGN_X) < 10:
            break
        if int(st.samus_x) < _FLOOR_ALIGN_X:
            hold(session, 1, "RIGHT", reason=f"{label}_floor_align")
        else:
            hold(session, 1, "LEFT", reason=f"{label}_floor_align")
    _settle_ground(session, f"{label}_floor_align")
    return session.state


def _snake_climb_to_top(session: ControllerSession, label: str) -> SuperMetroidState:
    """Multi-attempt 2WJ platform climb — L3 is pin-sensitive after Ice return.

    Important: do **not** thrash-recover between attempts (LEFT+A spam drops
    the retry pin). Just re-enter ``_snake_platform_climb`` from wherever we
    landed — bands skip by y, so mid-height retries continue upward.
    """
    if int(session.state.selected_item) != 0:
        select_weapon(session, 0)

    for attempt in range(_CLIMB_ATTEMPTS):
        st = session.state
        if not in_ice_snake(st):
            return st
        if on_snake_top(st):
            return st
        # Near-top settle (door height) — treat as climb done for door leave.
        if int(st.samus_y) <= SNAKE_TOP_Y[1] + 10 and int(st.velocity_y) == 0:
            _settle_ground(session, f"{label}_near_top")
            if on_snake_top(session.state) or int(session.state.samus_y) <= SNAKE_TOP_Y[1] + 20:
                return session.state

        # Only re-climb while still in Snake and below top band.
        if int(session.state.samus_y) > SNAKE_TOP_Y[1]:
            _snake_platform_climb(session, f"{label}_a{attempt}")

        if on_snake_top(session.state):
            return session.state

    if not on_snake_top(session.state) and int(session.state.samus_y) > SNAKE_TOP_Y[1]:
        st = session.state
        raise TimeoutError(
            f"{label}: climb missed top; room=0x{int(st.room_id):04X} "
            f"pose={st.pose} xy=({st.samus_x},{st.samus_y}) "
            f"attempts={_CLIMB_ATTEMPTS}"
        )
    return session.state


def _snake_top_to_tutorial(session: ControllerSession, label: str) -> SuperMetroidState:
    """Top shelf → right blue door → ordinary Ice Tutorial."""
    unmorph(session)
    if int(session.state.selected_item) != 0:
        select_weapon(session, 0)

    for frame in range(SNAKE_TOP_TO_TUTORIAL_FRAMES):
        st = session.state
        if int(st.room_id) == ROOM_ICE_TUTORIAL:
            break
        if not in_ice_snake(st):
            break

        if is_knockback(st):
            escape_knockback_spin(
                session,
                prefer_dir="RIGHT",
                run_frames=2,
                spin_frames=10,
                label=f"{label}_door_kb",
                ensure_beam=True,
                break_on_motion_clear=True,
            )
            continue

        if is_morph(st.pose) or int(st.pose) in (39, 40, 41, 42):
            hold(session, 1, "UP", reason=f"{label}_door_unmorph")
            continue

        x = int(st.samus_x)
        y = int(st.samus_y)

        if y > SNAKE_TOP_Y[1] + 30:
            hold(session, 1, "RIGHT", "A", reason=f"{label}_reclimb")
            continue

        if x >= SNAKE_TUTORIAL_DOOR_X and SNAKE_TUTORIAL_DOOR_Y[0] <= y <= SNAKE_TUTORIAL_DOOR_Y[1]:
            phase = frame % 16
            if phase < 4:
                hold(session, 1, "RIGHT", "X", reason=f"{label}_door_shot")
            elif phase < 11:
                hold(session, 1, "RIGHT", "B", reason=f"{label}_door_push")
            else:
                hold(session, 1, "RIGHT", "B", "A", reason=f"{label}_door_spin")
            continue

        if x < 100:
            phase = frame % 18
            if phase < 10:
                hold(session, 1, "RIGHT", "B", reason=f"{label}_top_run")
            elif phase < 14:
                hold(session, 1, "RIGHT", "B", "A", reason=f"{label}_top_hop")
            else:
                hold(session, 1, "RIGHT", reason=f"{label}_top_walk")
            continue

        if x < SNAKE_TUTORIAL_DOOR_X:
            phase = frame % 18
            if phase < 8:
                hold(session, 1, "RIGHT", "B", reason=f"{label}_approach")
            elif phase < 12:
                hold(session, 1, "RIGHT", "A", reason=f"{label}_approach_jump")
            elif phase < 15:
                hold(session, 1, "RIGHT", "X", reason=f"{label}_approach_shot")
            else:
                hold(session, 1, "RIGHT", reason=f"{label}_approach_walk")
            continue

        hold(session, 1, "RIGHT", reason=f"{label}_door_nudge")
    else:
        st = session.state
        raise TimeoutError(
            f"{label}: Tutorial door missed; room=0x{int(st.room_id):04X} "
            f"pose={st.pose} xy=({st.samus_x},{st.samus_y}) "
            f"door_transition={st.door_transition}"
        )

    return wait_ordinary_room(
        session,
        ROOM_ICE_TUTORIAL,
        settle_frames=TUTORIAL_RETURN_SETTLE,
        label=label,
    )


def play_ice_snake_to_tutorial(session: ControllerSession) -> SuperMetroidState:
    """Snake mid-right Ice-door pin → ordinary Ice Tutorial.

    Expects post-Ice return handoff ``post_ice_to_snake_pure`` in Snake
    ``0xA8B9`` near the center-right Ice door. Drops to floor, 2WJ climbs
    (multi-attempt), exits top-right into Tutorial ``0xA865``.
    """
    label = "ice_snake_to_tutorial"
    require_room(session, ROOM_ICE_SNAKE, label)
    start = session.frame

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

    if on_snake_top(session.state):
        return _snake_top_to_tutorial(session, label)

    if int(session.state.samus_y) < SNAKE_HANDOFF_Y[0] - 20:
        _snake_mid_right_to_floor(session, label)

    if session.state.room_id == ROOM_ICE_SNAKE and not on_snake_top(session.state):
        _snake_climb_to_top(session, label)

    if int(session.state.room_id) == ROOM_ICE_TUTORIAL:
        return wait_ordinary_room(
            session,
            ROOM_ICE_TUTORIAL,
            settle_frames=TUTORIAL_RETURN_SETTLE,
            label=label,
        )

    if session.state.room_id != ROOM_ICE_SNAKE:
        st = session.state
        raise TimeoutError(
            f"{label}: left Snake without Tutorial; room=0x{int(st.room_id):04X} "
            f"pose={st.pose} xy=({st.samus_x},{st.samus_y}) "
            f"frames={session.frame - start}"
        )

    state = _snake_top_to_tutorial(session, label)
    if int(state.room_id) != ROOM_ICE_TUTORIAL:
        raise TimeoutError(
            f"{label}: finished without Tutorial; "
            f"room=0x{int(state.room_id):04X} xy=({state.samus_x},{state.samus_y}) "
            f"frames={session.frame - start}"
        )
    return state


__all__ = ["play_ice_snake_to_tutorial"]
