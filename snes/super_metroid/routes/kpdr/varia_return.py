"""Post-Varia return hops (K3 exit → Kraid → … → Business for K4).

First natural door after Varia PLM: Varia Suit Room left blue door back into
Kraid's Room. Continuous tip ends mid-item pose (~81); this hop recovers to
standing, opens the blue door, and spin-pushes left through the transition.
"""

from __future__ import annotations

from super_metroid.policy import StateRequirement
from super_metroid.ram import SuperMetroidState
from super_metroid.routes.controller_common import (
    hold,
    require_room,
    require_state,
    select_weapon,
    unmorph,
    wait_ordinary_room,
)
from super_metroid.routes.kpdr.rooms import (
    ITEM_VARIA,
    ROOM_KRAID,
    ROOM_KRAID_EYE,
    ROOM_VARIA,
)
from super_metroid.routes.runtime import ControllerSession

_STANDING = frozenset({1, 2, 5, 6, 9, 10})
_BAD_AIR = frozenset({81, 164, 0, 155})


def _recover_standing(session: ControllerSession, *, timeout: int = 400) -> None:
    """Leave post-item / mid-air lock poses and land on the Varia floor."""
    unmorph(session)
    for index in range(timeout):
        st = session.state
        if (
            st.pose in _STANDING
            and st.velocity_y == 0
            and st.samus_y >= 130
        ):
            return
        phase = index % 24
        if st.pose in _BAD_AIR or st.pose in (137, 138):
            if phase < 4:
                hold(session, 1, "A", reason="varia_recover_jump")
            elif phase < 12:
                # Nudge right so we do not pin against the left door while locked.
                hold(session, 1, "RIGHT", reason="varia_recover_nudge")
            else:
                hold(session, 1, reason="varia_recover_idle")
            continue
        if phase < 3:
            hold(session, 1, "A", reason="varia_recover_jump")
        elif phase < 10:
            hold(session, 1, "RIGHT", reason="varia_recover_nudge")
        else:
            hold(session, 1, reason="varia_recover_idle")
    raise TimeoutError(
        f"varia_to_kraid: recover standing failed: {session.state}"
    )


def play_varia_to_kraid(session: ControllerSession) -> SuperMetroidState:
    """Exit Varia Suit Room left into Kraid's Room (post-collect return).

    Expects ordinary gameplay in ``0xA6E2`` with Varia already collected
    (continuous tip endpoint or pure post-collect state). Controller-dev until
    composed on continuous ``--to varia`` prefix as a SpineHop.
    """
    require_state(
        session,
        StateRequirement(
            room_id=ROOM_VARIA,
            game_states=frozenset({8}),
            collected_items_mask=ITEM_VARIA,
        ),
        "varia_to_kraid",
    )
    _recover_standing(session)

    # Stay mid-room so door shots clear the left shell without pin-knockback.
    for _ in range(50):
        x = session.state.samus_x
        if 95 <= x <= 130:
            break
        hold(
            session,
            1,
            "RIGHT" if x < 95 else "LEFT",
            reason="varia_center",
        )
    hold(session, 8, reason="varia_center_settle")

    try:
        select_weapon(session, 0)
    except RuntimeError:
        pass

    # Face left, open the blue door with standing beam shots.
    hold(session, 8, "LEFT", reason="varia_face_left")
    hold(session, 6, reason="varia_face_release")
    for _ in range(4):
        hold(session, 4, "X", reason="varia_door_shot")
        hold(session, 18, reason="varia_door_fuse")

    # Spin-push through. Walk-only often pin-locks pose 138 on the door lip.
    for _ in range(360):
        state = hold(session, 1, "LEFT", "B", "A", reason="varia_exit_spin")
        if state.room_id == ROOM_KRAID:
            break
        # Knockback against the lip: brief recover then resume spin.
        if state.pose in (137, 138) and state.samus_x <= 60:
            hold(session, 4, reason="varia_lip_release")
            hold(session, 3, "RIGHT", reason="varia_lip_backoff")
            hold(session, 4, "X", reason="varia_lip_reshot")
            hold(session, 12, reason="varia_lip_fuse")
    else:
        raise TimeoutError(
            f"varia_to_kraid: did not reach 0x{ROOM_KRAID:04X}: {session.state}"
        )

    # Varia left door lands on the right side of Kraid's Room (~x 450+).
    return wait_ordinary_room(
        session,
        ROOM_KRAID,
        settle_frames=240,
        label="varia_to_kraid",
        x_range=(350, 560),
        y_range=(300, 450),
        min_settle_frame=12,
    )


def play_kraid_to_eye_return(session: ControllerSession) -> SuperMetroidState:
    """Left-door return from Kraid's Room to the Eye Door Room (post-Varia).

    Controller-dev reverse hop from a continuous-like post-Varia re-entry on
    the right side of Kraid's Room. Not continuous evidence until composed
    after pure green + planner graph promote.

    Geometry note (SM-K4-06E): this left hatch is a gray/blue shell that opens
    with standing beams, but the transition trigger does **not** fire on the
    natural floor band (y≈400–427). Prior 06B–06D pins (pose 82/138, x≈37,
    door_transition=0) were floor-lip stalls. After opening the door, the exit
    must **jump-enter** through the elevated Y band (~140–380) — floor spin
    alone never transitions.
    """
    require_state(
        session,
        StateRequirement(room_id=ROOM_KRAID, game_states=frozenset({8})),
        "kraid_to_eye_return",
    )
    # Keep both the explicit room guard and the ordinary-gameplay requirement
    # visible at this boundary for pure probes and future entry tightening.
    require_room(session, ROOM_KRAID, "kraid_to_eye_return")
    select_weapon(session, 0)

    # Stage mid-left of the arena (not on the lip) for standing door shots.
    for _ in range(150):
        state = hold(session, 1, "LEFT", reason="kraid_return_approach")
        if state.samus_x <= 160:
            break
    else:
        raise TimeoutError(
            f"kraid_to_eye_return: left door approach timed out: {session.state}"
        )
    hold(session, 12, reason="kraid_return_approach_settle")
    # Small right backoff so shots clear the shell without walking into it.
    hold(session, 10, "RIGHT", reason="kraid_return_lip_backoff")
    unmorph(session)
    hold(session, 8, "LEFT", reason="kraid_return_face_left")
    hold(session, 6, reason="kraid_return_release")
    # Standing beam only (mirror varia_to_kraid) — do not hold LEFT while
    # firing or Samus walks into the closed shell.
    for _ in range(6):
        hold(session, 4, "X", reason="kraid_return_door_shot")
        hold(session, 14, reason="kraid_return_door_fuse")

    # Jump-enter through the elevated trigger band. Floor walk/spin pins at
    # x≈37–85 with door_transition=0 even after the shell is open.
    for index in range(900):
        phase = index % 30
        if phase < 4:
            state = hold(session, 1, "LEFT", "A", reason="kraid_return_jump")
        elif phase < 10:
            state = hold(
                session, 1, "LEFT", "A", "B", reason="kraid_return_jump_spin"
            )
        elif phase < 14:
            state = hold(session, 1, "X", reason="kraid_return_reshot")
        else:
            state = hold(session, 1, "LEFT", "B", reason="kraid_return_run")
        if state.room_id == ROOM_KRAID_EYE:
            break
        if state.door_transition:
            for _ in range(80):
                state = hold(session, 1, reason="kraid_return_transition")
                if (
                    state.room_id == ROOM_KRAID_EYE
                    and state.door_transition == 0
                ):
                    break
            if state.room_id == ROOM_KRAID_EYE:
                break
        if state.pose in (137, 138) and state.samus_x <= 80:
            hold(session, 4, reason="kraid_return_lip_release")
            hold(session, 4, "RIGHT", reason="kraid_return_lip_backoff")
            hold(session, 4, "X", reason="kraid_return_lip_reshot")
            hold(session, 12, reason="kraid_return_lip_fuse")
    else:
        raise TimeoutError(
            f"kraid_to_eye_return: left eye-door exit timed out: {session.state}"
        )
    return wait_ordinary_room(
        session,
        ROOM_KRAID_EYE,
        settle_frames=340,
        label="kraid_to_eye_return",
        x_range=(300, 560),
        y_range=(300, 450),
        min_settle_frame=12,
    )
