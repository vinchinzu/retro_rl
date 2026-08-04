"""Big Pink main shaft → Green Hill Zone (with Charge Beam detour)."""

from __future__ import annotations

from super_metroid.ram import SuperMetroidState
from super_metroid.routes.controller_common import (
    ensure_morph,
    hold,
    require_room,
    select_weapon,
    unmorph,
    wait_ordinary_room,
)
from super_metroid.routes.kpdr.charge_return import (
    CHARGE_BEAM_MASK,
    play_charge_beam_collect,
    play_charge_beam_return,
)
from super_metroid.routes.kpdr.rooms import (
    ROOM_BIG_PINK,
    ROOM_GHZ,
)
from super_metroid.routes.runtime import ControllerSession
from super_metroid.routes.skills.geometry import POSE_KNOCKBACK
from super_metroid.routes.skills.knockback import escape_knockback_spin

# Tunnel lip band for morph roll into lower corridor (human ~715,1590).
_LIP_Y_MAX = 1635
_LIP_Y_MIN = 1585
_LIP_X_MIN = 690


def _main_shaft_to_mass(session: ControllerSession) -> SuperMetroidState:
    """Main shaft → mass face when Charge is already held (skip Chozo)."""
    from super_metroid.routes.kpdr.charge_return import _descend_main_to_mass

    return _descend_main_to_mass(session)


def _climb_charge_pit_to_mass(session: ControllerSession) -> SuperMetroidState:
    """If still deep in the Charge drop shaft, staircase up to mass band."""
    unmorph(session)
    best_y = session.state.samus_y
    for _cycle in range(30):
        if session.state.samus_y <= 1700 and session.state.samus_x >= 700:
            hold(session, 10, reason="big_pink_mass_band_settle")
            return session.state
        hold(session, 6, "RIGHT", "B", reason="big_pink_pit_runup")
        hold(session, 1, "RIGHT", "A", reason="big_pink_pit_jump")
        for _ in range(28):
            state = hold(session, 1, "RIGHT", "A", reason="big_pink_pit_air")
            if state.samus_y < best_y:
                best_y = state.samus_y
            if state.samus_y <= 1700:
                break
        for _ in range(16):
            hold(session, 1, reason="big_pink_pit_land")
    raise TimeoutError(
        f"big_pink_to_ghz: charge-pit climb stalled best_y={best_y}: {session.state}"
    )


def _on_tunnel_lip(state: SuperMetroidState) -> bool:
    return (
        state.samus_x >= _LIP_X_MIN
        and _LIP_Y_MIN <= state.samus_y <= _LIP_Y_MAX
    )


def _break_pose_lag(session: ControllerSession, *, label: str) -> SuperMetroidState:
    """Exit brief pose-137/138 lag on the mass shelf.

    Human ``charge_human`` f2003–2020: pose 138 lasts ~16f then **A** jumps
    out — idle-holding never clears it.
    """
    unmorph(session)
    if int(session.state.pose) not in POSE_KNOCKBACK:
        hold(session, 8, reason=f"{label}_stand_settle")
        return session.state
    hold(session, 12, "A", reason=f"{label}_lag_a")
    for _ in range(24):
        state = hold(session, 1, reason=f"{label}_lag_land")
        if int(state.pose) not in POSE_KNOCKBACK:
            break
    if int(session.state.pose) in POSE_KNOCKBACK:
        escape_knockback_spin(
            session,
            prefer_dir="LEFT",
            run_frames=3,
            spin_frames=12,
            label=label,
        )
        for _ in range(20):
            hold(session, 1, reason=f"{label}_kb_idle")
            if int(session.state.pose) not in POSE_KNOCKBACK:
                break
    hold(session, 10, reason=f"{label}_stand_settle")
    return session.state


def _mass_to_tunnel_lip(session: ControllerSession) -> SuperMetroidState:
    """Mass platform → morph-tunnel lip (human ``charge_human`` f1950–2155).

    From the grounded right ledge (~743,1755 pose 1):

    1. A then LEFT+A onto upper mass shelf (~709,1675).
    2. A-break pose-138 lag; crouch; walk left to ~675.
    3. UP+RIGHT aim, A vertical, RIGHT+A onto lip (~715,1590).
    """
    _break_pose_lag(session, label="big_pink_lip")

    if session.state.samus_y > 1780:
        _climb_charge_pit_to_mass(session)
        _break_pose_lag(session, label="big_pink_lip_post_pit")

    # Shelf hop (human f1950–2000).
    for _attempt in range(4):
        if session.state.samus_y <= 1690 and 690 <= session.state.samus_x <= 725:
            break
        hold(session, 12, "A", reason="big_pink_shelf_prep")
        for _ in range(45):
            state = hold(session, 1, "LEFT", "A", reason="big_pink_shelf_hop")
            if state.samus_y <= 1690 and state.samus_x <= 720:
                break
        for _ in range(20):
            hold(session, 1, reason="big_pink_shelf_land")
        _break_pose_lag(session, label="big_pink_shelf")

    # Human f2065–2105: crouch then walk left to ~675.
    hold(session, 6, "DOWN", reason="big_pink_shelf_crouch")
    hold(session, 4, reason="big_pink_shelf_crouch")
    hold(session, 6, "DOWN", reason="big_pink_shelf_crouch")
    for _ in range(50):
        if session.state.samus_x <= 680:
            break
        hold(session, 1, "LEFT", reason="big_pink_mass_left")
    hold(session, 10, reason="big_pink_mass_left_settle")

    # Human f2110–2155: UP+RIGHT aim → A → RIGHT+A to lip.
    for _attempt in range(6):
        if _on_tunnel_lip(session.state):
            hold(session, 16, reason="big_pink_tunnel_lip_settle")
            return session.state
        _break_pose_lag(session, label="big_pink_lip_try")
        hold(session, 8, "UP", "RIGHT", reason="big_pink_lip_aim")
        hold(session, 10, "A", reason="big_pink_lip_prep")
        for _ in range(40):
            state = hold(session, 1, "RIGHT", "A", reason="big_pink_lip_jump")
            if _on_tunnel_lip(state):
                hold(session, 16, reason="big_pink_tunnel_lip_settle")
                return session.state
        for _ in range(16):
            hold(session, 1, reason="big_pink_lip_land")
        if session.state.samus_x > 730:
            for _ in range(25):
                hold(session, 1, "LEFT", reason="big_pink_lip_recenter")
            hold(session, 6, "DOWN", reason="big_pink_lip_recrouch")

    # Fallback: pre-Charge mass face + continuous RIGHT+A mount.
    hold(session, 20, "RIGHT", "B", reason="big_pink_mass_run")
    hold(session, 10, reason="big_pink_mass_settle")
    hold(session, 12, "LEFT", reason="big_pink_mass_brake")
    hold(session, 8, "A", reason="big_pink_mass_vertical")
    for _ in range(200):
        state = hold(session, 1, "RIGHT", "A", reason="big_pink_tunnel_mount")
        if _on_tunnel_lip(state):
            hold(session, 40, reason="big_pink_tunnel_lip_settle")
            return session.state
    raise TimeoutError(f"big_pink_to_ghz: missed morph-tunnel lip: {session.state}")


def _mass_to_ghz(session: ControllerSession) -> SuperMetroidState:
    """Mass / post-Charge return → corridor → green Super door → GHZ.

    Geometry: mass → tunnel lip → morph bomb-roll (human drops to y≈1685 in
    the lower corridor) → Super door pocket → GHZ.
    """
    _mass_to_tunnel_lip(session)

    state = session.state
    if not (state.samus_x >= 680 and state.samus_y <= 1650):
        raise TimeoutError(
            f"big_pink_to_ghz: not on tunnel lip: {session.state}"
        )

    # Morph roll through lower corridor (human f2230–2500: x715→936).
    ensure_morph(session)
    for frame in range(600):
        buttons = ("RIGHT", "X") if frame % 45 < 3 else ("RIGHT",)
        state = hold(session, 1, *buttons, reason="big_pink_bomb_roll")
        if state.samus_x >= 900:
            break
    else:
        raise TimeoutError(
            f"big_pink_to_ghz: lower bomb-roll stalled: {session.state}"
        )

    unmorph(session)
    for _ in range(160):
        state = hold(session, 1, "RIGHT", reason="big_pink_door_approach")
        if state.samus_x >= 930 and state.samus_y >= 1660:
            break
    else:
        raise TimeoutError(
            f"big_pink_to_ghz: missed green-door pocket: {session.state}"
        )

    hold(session, 12, reason="big_pink_door_settle")
    select_weapon(session, 2)
    # Human: Super from x936, walk right into GHZ.
    hold(session, 6, reason="big_pink_super_ready")
    hold(session, 3, "RIGHT", reason="big_pink_face_door")
    hold(session, 3, reason="big_pink_face_door_release")
    hold(session, 8, "X", reason="big_pink_green_door_super")
    hold(session, 50, reason="big_pink_green_door_fuse")
    for _ in range(300):
        state = hold(session, 1, "RIGHT", "B", reason="big_pink_enter_ghz")
        if state.room_id == ROOM_GHZ:
            break
    else:
        raise TimeoutError(
            f"big_pink_to_ghz: green door did not open: {session.state}"
        )
    return wait_ordinary_room(
        session, ROOM_GHZ, settle_frames=240, label="big_pink_to_ghz"
    )


def play_big_pink_to_ghz(session: ControllerSession) -> SuperMetroidState:
    """Natural Big Pink main-shaft anchor → Charge Beam → Green Hill Zone.

    Route (human-aligned where pure-stable):

    1. Descend to mass (optional missile pack brush).
    2. Bomb-drop to Charge Chozo, **R-angle** open, collect.
    3. Ordinary jump staircase return to grounded mass ledge.
    4. Lip climb + corridor / Super into GHZ.

    If Charge is already held, skips the Chozo detour.
    """
    require_room(session, ROOM_BIG_PINK, "big_pink_to_ghz")

    if session.state.collected_beams & CHARGE_BEAM_MASK:
        _main_shaft_to_mass(session)
        return _mass_to_ghz(session)

    play_charge_beam_collect(session)
    play_charge_beam_return(session)
    return _mass_to_ghz(session)
