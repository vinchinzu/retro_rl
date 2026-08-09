"""Pure Ice Beam Snake Room → Ice Beam PLM collect.

Source: ``post_ice_acid_to_snake_pure`` ~(216, 651) in ``0xA8B9``.
Technique: platform-hop / 2WJ climb bands (not freeze ladder). Tape thrash
f12664–15400 is non-product.

Path sketch (tape + live pure probe)::

    floor y651 → L1..L8 platform hops → top y139
      → right cross past center wall (x>171 only open at top)
        → right column → morph tunnel y~377 → Ice 0xA890 → PLM bit 0x0002

**Residual (rr-5if):** climb bands dual-greenable; morph-tunnel entry from the
right column after top cross is the open knob. Left-wall morph at x=171 is
solid at mid height — do not thrash freeze platforms.
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
from super_metroid.routes.kpdr.ice.geometry import (
    ICE_BEAM_MASK,
    ICE_PLM_X,
    ICE_ROOM_SETTLE,
    SNAKE_CLIMB_FRAMES,
    SNAKE_ICE_COLLECT_FRAMES,
    SNAKE_L1_Y,
    SNAKE_L2_Y,
    SNAKE_L3_Y,
    SNAKE_L4_Y,
    SNAKE_L5_Y,
    SNAKE_L6_Y,
    SNAKE_L7_Y,
    SNAKE_TOP_Y,
    SNAKE_TUNNEL_FRAMES,
    SNAKE_TUNNEL_X_MIN,
    SNAKE_TUNNEL_Y,
    has_ice,
    in_ice_snake,
    on_snake_top,
    on_snake_tunnel_band,
)
from super_metroid.routes.kpdr.k4_common import _STANDING_POSES
from super_metroid.routes.kpdr.rooms import ROOM_ICE, ROOM_ICE_SNAKE
from super_metroid.routes.runtime import ControllerSession
from super_metroid.routes.skills.knockback import (
    escape_knockback_spin,
    is_knockback,
)

_LEDGE = _STANDING_POSES | frozenset({1, 2, 9, 10, 37, 38})


def _settle_ground(session: ControllerSession, label: str, *, max_frames: int = 40) -> None:
    for _ in range(max_frames):
        st = session.state
        if (
            int(st.velocity_y) == 0
            and int(st.pose) in _LEDGE
            and int(st.door_transition) == 0
        ):
            return
        if is_knockback(st):
            escape_knockback_spin(
                session,
                prefer_dir="LEFT",
                run_frames=2,
                spin_frames=10,
                label=f"{label}_kb",
                ensure_beam=True,
                break_on_motion_clear=True,
            )
            continue
        if int(st.pose) in (31, 39, 40, 41, 42, 65):
            hold(session, 1, "UP", reason=f"{label}_unmorph")
        else:
            hold(session, 1, reason=f"{label}_settle")


def _y_band(state: SuperMetroidState, band: tuple[int, int]) -> bool:
    return band[0] <= int(state.samus_y) <= band[1]


def _snake_platform_climb(session: ControllerSession, label: str) -> SuperMetroidState:
    """Floor handoff → top shelf via alternating platform hops (2WJ-style).

    Live pure pin sequence (post_ice_acid_to_snake_pure)::

        L1 y587 ← left run+jump
        L2 y523 ← right spin
        L3 y459 ← left
        L4 y395 ← A then RIGHT+A from left wall
        L5 y331 ← RIGHT then LEFT+A
        L6 y267 ← left wall, A, RIGHT+A
        L7 y203 ← RIGHT then LEFT+A
        top y139 ← left wall, A, RIGHT+A
    """
    require_room(session, ROOM_ICE_SNAKE, label)
    unmorph(session)
    if int(session.state.selected_item) != 0:
        select_weapon(session, 0)

    # --- L1: left run + left jump ---
    if int(session.state.samus_y) > SNAKE_L1_Y[1]:
        hold(session, 8, reason=f"{label}_l1_idle")
        hold(session, 11, "LEFT", reason=f"{label}_l1_walk")
        hold(session, 11, "LEFT", "B", reason=f"{label}_l1_run")
        hold(session, 34, "LEFT", "B", "A", reason=f"{label}_l1_jump")
        hold(session, 12, "LEFT", "B", reason=f"{label}_l1_coast")
        _settle_ground(session, f"{label}_l1")

    # --- L2: right spin hop ---
    if int(session.state.samus_y) > SNAKE_L2_Y[1]:
        hold(session, 3, "B", reason=f"{label}_l2_b")
        hold(session, 4, "B", "A", reason=f"{label}_l2_ba")
        hold(session, 23, "RIGHT", "B", "A", reason=f"{label}_l2_jump")
        hold(session, 27, "RIGHT", "B", reason=f"{label}_l2_coast")
        _settle_ground(session, f"{label}_l2")

    # --- L3: left hop ---
    if int(session.state.samus_y) > SNAKE_L3_Y[1]:
        hold(session, 3, "LEFT", "A", reason=f"{label}_l3_a")
        hold(session, 21, "LEFT", "A", reason=f"{label}_l3_jump")
        hold(session, 12, "LEFT", reason=f"{label}_l3_coast")
        hold(session, 8, "LEFT", "B", reason=f"{label}_l3_run")
        hold(session, 12, "LEFT", reason=f"{label}_l3_walk")
        _settle_ground(session, f"{label}_l3")

    # --- L4: left wall, A, RIGHT+A → mid door height ---
    if int(session.state.samus_y) > SNAKE_L4_Y[1]:
        hold(session, 5, "A", reason=f"{label}_l4_a")
        hold(session, 12, "RIGHT", "A", reason=f"{label}_l4_ra")
        hold(session, 12, "RIGHT", "B", "A", reason=f"{label}_l4_spin")
        hold(session, 25, "RIGHT", reason=f"{label}_l4_coast")
        for _ in range(50):
            st = session.state
            if (
                _y_band(st, SNAKE_L4_Y)
                and int(st.velocity_y) == 0
                and int(st.pose) in _LEDGE
            ):
                break
            hold(session, 1, reason=f"{label}_l4_land")
        _settle_ground(session, f"{label}_l4")

    # --- L5: RIGHT then LEFT+A ---
    if int(session.state.samus_y) > SNAKE_L5_Y[1]:
        hold(session, 12, "RIGHT", reason=f"{label}_l5_right")
        hold(session, 28, "LEFT", "A", reason=f"{label}_l5_jump")
        hold(session, 15, "LEFT", reason=f"{label}_l5_coast")
        _settle_ground(session, f"{label}_l5")

    # --- L6: left wall, A, RIGHT+A ---
    if int(session.state.samus_y) > SNAKE_L6_Y[1]:
        hold(session, 18, "LEFT", reason=f"{label}_l6_wall")
        hold(session, 4, "A", reason=f"{label}_l6_a")
        hold(session, 18, "RIGHT", "A", reason=f"{label}_l6_ra")
        hold(session, 10, "RIGHT", "B", "A", reason=f"{label}_l6_spin")
        hold(session, 15, "RIGHT", reason=f"{label}_l6_coast")
        _settle_ground(session, f"{label}_l6")

    # --- L7: RIGHT then LEFT+A ---
    if int(session.state.samus_y) > SNAKE_L7_Y[1]:
        hold(session, 20, "RIGHT", reason=f"{label}_l7_right")
        hold(session, 24, "LEFT", "A", reason=f"{label}_l7_jump")
        hold(session, 14, "LEFT", reason=f"{label}_l7_coast")
        _settle_ground(session, f"{label}_l7")

    # --- top: left wall, A, RIGHT+A ---
    if int(session.state.samus_y) > SNAKE_TOP_Y[1]:
        for _ in range(30):
            if int(session.state.samus_x) <= 68:
                break
            hold(session, 1, "LEFT", reason=f"{label}_top_wall")
        hold(session, 3, reason=f"{label}_top_pause")
        hold(session, 5, "A", reason=f"{label}_top_a")
        hold(session, 20, "RIGHT", "A", reason=f"{label}_top_ra")
        hold(session, 12, "RIGHT", "B", "A", reason=f"{label}_top_spin")
        hold(session, 25, "RIGHT", reason=f"{label}_top_coast")
        for _ in range(50):
            st = session.state
            if on_snake_top(st):
                break
            hold(session, 1, reason=f"{label}_top_land")
        _settle_ground(session, f"{label}_top")

    return session.state


def _snake_top_to_tunnel(session: ControllerSession, label: str) -> SuperMetroidState:
    """Top shelf → right column → morph tunnel band.

    Top cross past x=171 (only open at top). Shoot-down opens the right-column
    shelf (~y155 → y~270). Tunnel morph from the right column is residual.
    """
    # Cross right on top (avoid Tutorial door x≳230 y<160).
    for _ in range(50):
        st = session.state
        if not in_ice_snake(st):
            return st
        if int(st.samus_x) >= 200:
            break
        if int(st.samus_y) > 180:
            break
        hold(session, 1, "RIGHT", "B", reason=f"{label}_top_cross")

    # Jump right + morph fall; shoot-down if shelf traps us.
    hold(session, 10, "RIGHT", "A", reason=f"{label}_top_jump")
    for i in range(80):
        st = session.state
        if not in_ice_snake(st):
            return st
        y = int(st.samus_y)
        if y >= 250:
            break
        if 145 <= y <= 175 and int(st.velocity_y) == 0:
            # Shelf trap: shoot down (opens right column).
            hold(session, 1, "DOWN", "X", reason=f"{label}_shelf_shot")
            hold(session, 1, "X", reason=f"{label}_shelf_shot")
            hold(session, 2, reason=f"{label}_shelf_shot")
            continue
        hold(session, 1, "DOWN", reason=f"{label}_morph_fall")

    # Right platform ~y267: unmorph, walk left edge, drop toward tunnel band.
    unmorph(session)
    _settle_ground(session, f"{label}_right_plat")
    for _ in range(40):
        st = session.state
        if int(st.samus_x) <= 200 or int(st.samus_y) > 290:
            break
        hold(session, 1, "LEFT", reason=f"{label}_edge")

    # Drop / spin toward tunnel y band on right side.
    hold(session, 12, "LEFT", "A", reason=f"{label}_drop_jump")
    for i in range(SNAKE_TUNNEL_FRAMES):
        st = session.state
        if st.room_id == ROOM_ICE:
            return st
        if not in_ice_snake(st):
            return st
        if on_snake_tunnel_band(st) or (
            int(st.samus_x) >= SNAKE_TUNNEL_X_MIN
            and SNAKE_TUNNEL_Y[0] <= int(st.samus_y) <= SNAKE_TUNNEL_Y[1]
        ):
            break
        y = int(st.samus_y)
        if y > 520:
            # Too low — short left/right hop recovery toward mid.
            hold(session, 1, "LEFT", "A", reason=f"{label}_recover")
            continue
        if y < SNAKE_TUNNEL_Y[0]:
            hold(session, 1, "RIGHT", reason=f"{label}_drop_right")
        else:
            hold(session, 1, "RIGHT", reason=f"{label}_seek_tunnel")

    # Morph + roll right into tunnel / Ice door.
    if session.state.room_id == ROOM_ICE:
        return session.state
    try:
        ensure_morph(session)
    except TimeoutError:
        # Air morph: held DOWN while falling.
        for _ in range(12):
            hold(session, 1, "DOWN", reason=f"{label}_air_morph")
            if is_morph(session.state.pose):
                break

    for i in range(SNAKE_TUNNEL_FRAMES):
        st = session.state
        if st.room_id == ROOM_ICE:
            return st
        if not in_ice_snake(st):
            return st
        if is_knockback(st):
            hold(session, 1, "DOWN", "RIGHT", reason=f"{label}_kb_morph")
            continue
        if not is_morph(st.pose):
            hold(session, 1, "DOWN", reason=f"{label}_remorph")
            continue
        hold(session, 1, "RIGHT", reason=f"{label}_tunnel_roll")
        if int(st.samus_x) >= 320:
            unmorph(session)
            for j in range(120):
                st = session.state
                if st.room_id == ROOM_ICE:
                    return st
                phase = j % 14
                if phase < 3:
                    hold(session, 1, "RIGHT", "X", reason=f"{label}_door_shot")
                elif phase < 9:
                    hold(session, 1, "RIGHT", "B", reason=f"{label}_door_run")
                else:
                    hold(session, 1, "RIGHT", "B", "A", reason=f"{label}_door_hop")
            break

    return session.state


def _ice_collect_plm(session: ControllerSession, label: str) -> SuperMetroidState:
    """Ice Beam room left entry → chozo PLM (beam bit 0x0002)."""
    require_room(session, ROOM_ICE, label)
    if has_ice(session.state):
        return session.state

    unmorph(session)
    if int(session.state.selected_item) != 0:
        select_weapon(session, 0)
    for _ in range(30):
        st = hold(session, 1, reason=f"{label}_stand")
        if int(st.velocity_y) == 0 and int(st.pose) in _STANDING_POSES:
            break

    for frame in range(SNAKE_ICE_COLLECT_FRAMES):
        st = session.state
        if has_ice(st):
            break
        if st.room_id != ROOM_ICE:
            raise TimeoutError(
                f"{label}: left Ice during collect; "
                f"room=0x{int(st.room_id):04X} xy=({st.samus_x},{st.samus_y})"
            )
        if int(st.pose) in (137, 138, 39, 40, 41, 42):
            hold(session, 1, "UP", reason=f"{label}_unmorph")
            continue
        if int(st.samus_x) < ICE_PLM_X - 10:
            phase = frame % 20
            if phase < 8:
                hold(session, 1, "RIGHT", "B", "A", reason=f"{label}_chozo_hop")
            elif phase < 14:
                hold(session, 1, "RIGHT", "B", reason=f"{label}_chozo_run")
            else:
                hold(session, 1, "RIGHT", "X", reason=f"{label}_chozo_shot")
        else:
            if frame % 10 == 0:
                hold(session, 1, "X", reason=f"{label}_plm_shot")
            else:
                hold(session, 1, "RIGHT", reason=f"{label}_plm_walk")
    else:
        st = session.state
        raise TimeoutError(
            f"{label}: Ice PLM not collected; beams=0x{int(st.collected_beams):04X} "
            f"pose={st.pose} xy=({st.samus_x},{st.samus_y})"
        )

    hold(session, 80, reason=f"{label}_fanfare")
    unmorph(session)
    for _ in range(40):
        st = hold(session, 1, reason=f"{label}_post_stand")
        if int(st.velocity_y) == 0 and int(st.pose) in _STANDING_POSES:
            break
    return session.state


def play_ice_snake_to_ice(session: ControllerSession) -> SuperMetroidState:
    """Snake floor pin → Ice Beam PLM (beam bit 0x0002).

    Source: pure Acid→Snake handoff ``post_ice_acid_to_snake_pure`` ~(216, 651).
    """
    label = "ice_snake_to_ice"
    require_room(session, ROOM_ICE_SNAKE, label)
    start = session.frame

    if has_ice(session.state) and session.state.room_id == ROOM_ICE:
        return session.state

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

    if session.state.room_id == ROOM_ICE_SNAKE:
        _snake_platform_climb(session, label)

    if session.state.room_id == ROOM_ICE_SNAKE and not has_ice(session.state):
        if not on_snake_top(session.state) and int(session.state.samus_y) > SNAKE_TOP_Y[1]:
            # Climb budget exceeded — one more open attempt window.
            for _ in range(min(400, SNAKE_CLIMB_FRAMES // 4)):
                if on_snake_top(session.state) or session.state.room_id != ROOM_ICE_SNAKE:
                    break
                hold(session, 1, "LEFT", "A", reason=f"{label}_climb_push")
        _snake_top_to_tunnel(session, label)

    if session.state.room_id != ROOM_ICE:
        st = session.state
        raise TimeoutError(
            f"{label}: Ice door missed; room=0x{int(st.room_id):04X} "
            f"pose={st.pose} xy=({st.samus_x},{st.samus_y}) "
            f"door_transition={st.door_transition} "
            f"beams=0x{int(st.collected_beams):04X} "
            f"frames={session.frame - start} "
            f"(prefer 2WJ climb + right-column tunnel; not freeze ladder)"
        )

    wait_ordinary_room(
        session, ROOM_ICE, settle_frames=ICE_ROOM_SETTLE, label=label
    )
    state = _ice_collect_plm(session, label)

    if not has_ice(state):
        raise TimeoutError(
            f"{label}: finished without Ice bit; "
            f"beams=0x{int(state.collected_beams):04X} room=0x{int(state.room_id):04X} "
            f"xy=({state.samus_x},{state.samus_y}) "
            f"frames={session.frame - start}"
        )
    return state


__all__ = ["play_ice_snake_to_ice", "ICE_BEAM_MASK"]
